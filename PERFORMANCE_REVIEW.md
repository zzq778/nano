# r36 Jetson 视觉性能与质量审查

日期：2026-11-21。基于上传的 `jetson_vision_full_20261121.tar.gz`，原始配置为 r35，未修改模型、相机外参、阈值、状态机或 230400 波特率 UART 协议。

## 已合入的优化

- **`frontends.py`**：六类 YOLO 后的蓝色上下文补检、橙色可见色块补检复用一份整帧 HSV；不再重复在不同大小的区域里做颜色空间转换。橙色 `R/G`、`R/B` 比例检查只遍历满足原 HSV 门槛的像素，不再为整帧生成 `int16` 和浮点中间数组。原始的 HSV、比例、形态学、连通域、去重门槛不变。
- **`zones.py`**：粉色底面掩模用 OpenCV 的饱和减法与阈值替代全图 `int16` 算术，正差值阈值逐像素等价；用 `cv2.findNonZero` 直接给凸包输入像素坐标，避免 `np.nonzero` 后的 `column_stack` 和 `astype`。
- **`geometry.py`**：新增批量地面投影 `project_many`，一次投影整个安置区轮廓，保持原来的标定检查、有效射线与交点判定；原 `project` API 不变。
- **诊断**：`YoloFrontend.last_diagnostics.timings_ms` 显示原生检测、HSV/蓝色补检、橙色补检分段耗时；绿色顶面融合诊断包含底层 YOLO 的计时。更新的 `src/detector_bridge.cpp` 能在 Nano 上重新编译后提供原生检测的预处理、TensorRT 推理及传输、后处理三段耗时。`native.py` 对旧版 `libvision_detector.so` 保持兼容；若未重编译，`native_stages_ms` 不会出现。

## 同机 A/B 基准

在 **x86_64、Python 3.13.5、OpenCV 4.13.0、OpenCV 线程数=1** 的同一进程中，读取上传的 6 张 Astra 图像，采用相同的假检测框，交错调用优化前/后各 100 次并取中位数。未执行 TensorRT 推理与摄像头采集。

| 阶段 | r35 中位耗时 | r36 中位耗时 | 改善 |
|---|---:|---:|---:|
| 六类检测后的 CPU 颜色补检 | 8.123 ms | 6.260 ms | 22.9% |
| 蓝色平台轮廓和坐标 | 2.235 ms | 2.063 ms | 7.7% |
| 红/紫色平台轮廓和坐标 | 4.823 ms | 4.313 ms | 10.6% |

这些阶段可能按帧是否有对应目标而有条件执行，**不可直接累加**，更不能把上述百分比称为完整视觉系统的 FPS 提升。原归档 `astra_priority_bench` 的历史运行记录显示检测约 69 ms、细化约 34 ms、整帧计算约 115 ms；这些是早期配置的记录，**不是 r35/r36 在实际 Nano 上的新测值**。Nano 的 CPU、TensorRT、内存带宽和后台任务差异可显著改变收益。

## 正确性验证

- 6 张 Astra 图像、固定假检测框：r35 与 r36 的补检对象列表相同（类别、包围框、来源、中心、种子等）。
- 蓝/紫平台 × 有/无中间分隔 × 左右镜像共 8 种合成场景：r35 与 r36 的区域诊断、中心和朝向相同（浮点比较允许 5e-6）。
- 同一批真实图像和随机像素：新橙色比例筛选掩模与旧版逐像素相等。
- 新增 3 项有明确回归风险的测试。当前环境 `unittest discover` 共 112 项，**105 通过、7 因 ARM64 `.so` 在 x86 环境无法加载而出错**；无新的功能断言失败。老库仍为上传的 ARM64 预编译版本，本次未伪称 Nano 实测或重新编译成功。
- 未改变类别映射、安置区优先顺序、type2 角度和 `AA55 | N | 7字节×N | CRC16` 串口格式。

## Nano 上的完整验证命令

在 Jetson 上进入 `jetson_vision` 工程根目录并停止同时运行的生产服务后，检查下列项目：

```bash
# 1. 验证源码和随附图像身份
python3 tests/verify_sources.py
python3 -m unittest discover -s tests -v

# 2. 原生检测 C++ 分段计时（仅编译 detector，不触碰相机库）
cmake -S . -B build_perf -DWITH_TENSORRT=ON
cmake --build build_perf --target vision_detector -j2
# 验证通过后，再将 build_perf/libvision_detector.so 放在配置引用的路径下。

# 3. 真实 Astra 与 UART 的端到端性能记录（单进程，避免同时运行旧服务）
python3 app.py --camera astra --uart-device /dev/ttyTHS1 \
  --warmup 3 --max-frames 240 --quiet --output /tmp/vision_r36_bench
```

其中 `frames.jsonl` 的 `timings_ms` 和 `frontend_diagnostics` 可观察实帧耗时，`run_summary.json` 的 `observed_loop_hz` 包括采集、推理、串口、日志以及偶发的系统调度，不等同于纯推理 FPS。`native_stages_ms` 需要服务实际加载新编译的检测库才会出现。上面命令会对物理 UART 发包，务必在安全的台架环境、车辆不运动时运行；如果不需要物理串口，可改用 `--uart-loopback`。

## 仍存在的限制

本次环境不是 Jetson Nano，未运行 TensorRT 8.2、Astra 真实采集或重新编译 ARM64 库。完整 FPS、95/99 分位时延、模型检出率与坐标精度必须以上述板上测试为准。视觉有可能因多物块、背景复杂度、ROS 图像订阅、网页 JPEG 编码和每 5 秒的同步 PNG 抽样产生不同负载。正式服务的采样、监视和 ROS 功能均保留；没有通过关闭功能或变更置信度来制造更高 FPS。


---

# r37 第二轮视觉性能与质量审查（2026-11-21）

## 修改说明

- `frontends.py`、`green_detection.py`、`green_proposals.py`、`green_regions.py`、`zones.py` 和 `pipeline.py`：将原本已计算的全帧 HSV 向绿色候选、黑/蓝中心、黑色尖顶及蓝/紫安置区几何复用；需要兼容离线参考调用时仍接受 `hsv=None`。不删补检、不调整任何阈值。
- `pipeline.py`：没有需拟合的绿色/蓝色完整立方体时，避免二次调用 `metric_prepare()` 创建整帧颜色校正上下文；有物块需做原生姿态拟合时保持原流程。
- `sampling.py`、`app.py`：生产采样从主线程同步 PNG 编码/磁盘写入，改为单线程、单待写槽位的异步存档。主线程只在满足 5 秒间隔且容量允许时复制一次当前图像和 UART 结果，然后立即继续处理下一帧。磁盘写入失败信息与最终 saved 数写入 `sample_writer_summary`；不可把 `sample_queued` 误当已保存成功。保持 `SampleRecorder` 同步 API 不变以便测试和离线使用。
- `live_monitor.py`：监控不再额外拷贝已经由 `LatestCamera.get()` 独立拥有、后续只读的图像。监控仍按配置限频 JPEG 编码。
- `src/detection_postprocessor.cpp`：NMS 原来在全局 score 排序后，逐候选比较所有已接受类别；现在每个候选只比较同类索引，输出仍依原始稳定 score 顺序追加。`src/detector_bridge.cpp` 复用原始候选和恢复框缓冲。

## 可复验数据与边界

在相同 x86_64 环境与 6 张 Astra 原始帧上，`YoloFrontend`（模拟固定六类框）+ 中心/绿色候选的 CPU 路径交错各 75 次：r36 中位 **8.921 ms**，r37 **8.565 ms**，降低 **4.0%**。蓝色安置区路径交错各 75 次：r36 **2.755 ms**，r37 **2.653 ms**，降低 **3.7%**。这一对比不含 TensorRT 和原生 metric 拟合，不是整个 Nano FPS；两组基准彼此有重叠阶段，不应直接相加。

用 `g++ -O3` 编译真正的新版 `src/detection_postprocessor.cpp` 与独立原算法参考，在 50 组随机候选（包含高重叠、类 200 和 5,000 候选框场景）逐框比较，**50/50 完全相同**；参考总耗时 **130.295 ms**、新版 **57.557 ms**，该批合成 NMS 工作量加速 **2.26 倍**。这只是 NMS 子步骤的独立主机测试；Nano TensorRT 推理及算力限制仍然存在。

7 张实际 PNG 的固定假框完整 Python 流水线对照，r36/r37 非计时结果（目标、几何、区域、完整 UART 字节）全部一致。原 112 项 + 新增 6 项单元测试共 118 项，本机 **111 通过，7 项因 ARM64 原生库不能在 x86 加载而报错**；无回归断言失败。基于 `tests/nms_equivalence.cpp` 的跨平台独立 C++ 检验可不需要 OpenCV/CUDA 编译。

定时 PNG 测试人为令写盘耗时 120 ms：主循环提交约小于 60 ms（测试门槛），保存后图像/包/序号完全一致；这个测试验证了写盘未占用主处理线程，未声称这台机器能代表 Nano 的真实写入延迟。

## Nano 板端应用与完整验证

```bash
cd /home/rikibot/vision_research_20261026/full_project_20261027/jetson_vision
python3 tests/verify_sources.py
python3 -m unittest discover -s tests -v

# 单独编译/验证无 TensorRT 依赖的真实 NMS 源码
g++ -std=c++14 -O3 -Iinclude src/detection_postprocessor.cpp tests/nms_equivalence.cpp -o /tmp/nms_equivalence
/tmp/nms_equivalence

# 编译新的 TensorRT 库，否则仍使用附带的旧 ARM64 .so
cmake -S . -B build_perf -DWITH_TENSORRT=ON
cmake --build build_perf --target vision_detector -j2
# 检查新编译文件后，再受控地替换 build/libvision_detector.so

# 真实 Astra/YOLO/PTY 端到端运行（不发送物理车体串口）
python3 app.py --camera astra --uart-loopback --max-frames 240 \
  --warmup 3 --quiet --no-frame-log --sample-interval 5 \
  --output /tmp/vision_r37_test
```

需要真正证明整机 FPS 收益，必须记录上述板上 `run_summary.json`、分阶段延迟和实际 GPU 占用，并与同配置旧版本比较。目前未接触真实 Nano、Astra 和 STM32，不能假装板上实测。若更换主机编译的 `.so`、降低阈值或禁用默认 ROS、监控功能以换 FPS，会影响可比性，均未这样做。

## r39 独立二次审查结果（2026-11-21）

重读 `app.py` 至 `runtime.SerialSink.write` 的真实路径，并检查上版测试是否误把 Python 回退算作生产 native metric 收益。r39 的新可测热点为 `Geometry.set_position` 针孔条件下每目标 `cv2.projectPoints` 诊断投影：以同样的 K 内参代数公式替代，500 坐标和 28 组输入输出对照通过，UART 帧完全一致。x86 200 次投影构造 P50 14.297→0.890ms；七张 Astra 图加固定 15/32 模拟检测框，50 次交错回放 P50 分别 39.560→37.724ms、75.026→73.917ms；**不含 TensorRT 或 native metric，也不代表 Nano FPS**。0/1 框部分轮次反而变慢，不宣称统一的整机或尾延迟改善。监控、逐帧 JSON、抽样提交的 CPU 代价现在可以在 `run_summary.json` 单独观察，关闭时返回 null。具体复现实验与未测试设备项见 docs 下四份 r39 追加记录。
