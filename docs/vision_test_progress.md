# r37 → r39 优化与独立复审测试进展

日期：2026-11-21。主机 x86_64 / Python 3.13.5 / OpenCV 4.13.0。所有基准区分开发机 CPU、模拟模型框回放和真实 Nano 指标。

| 轮次 | 变更 / 依据 | 验证与结论 |
|---|---|---|
| R0：冻结基线 | 保留 r37 完整副本及本地 Git 初始提交；检查 `app.py`、`pipeline.py`、`frontends.py`、`runtime.py`、`src/` 以及生产 service 和模型映射 | r37 `unittest discover`：118 项，111 通过、7 项因加载 ARM64 `.so` 在 x86 上报错。r37 模型/库哈希记录并封存 |
| R1：UART 热点 | `uart.crc16` 的 8 次/字节位运算改为同多项式的一次查表；`Pipeline.process` 单目标仅执行现有 `_pack_target` 校验，不再生成独立完整帧与 CRC；全部合法目标的诊断预览直接复用已编码帧 | `tools/bench_wire_r38.py` 交错 A/B，0/1/15/32 条线包字节相同；15 条 P50 83.007→22.243 μs，32 条 176.986→45.847 μs。CRC golden+270 个随机负载、0/1/15/32/33、invalid 字段通过 |
| R2：可选 Python 几何回退 | `shape_center.cube_bbox_center` 针孔路径使用直接投影及 2×2 Gram 方程替代反复 `cv2.projectPoints`、`cond`、`lstsq`；带畸变路径保留原数值求解 | 198 个随机姿态生成的合法空间盒：198/198 几何对照通过；P50 9.761→3.823 ms（子步骤下降 60.84%）。新增 r37 golden 样本：33 合法+8 非法。**正式配置原生 metric_cube 路径不使用此 Python 回退** |
| R3：局部 ROI 内存 | `pipeline.top_quad` 使用坐标轴广播代替 `np.mgrid` 两张完整坐标图 | 合成橙色块四顶点逐点一致；独立局部 P50 414→250 μs，不能外推整体 FPS |
| R4：端到端观测 | `app.py` 采集/处理/应用层写出延迟、新鲜度、跳帧和各阶段 P50/P95；仅保存最近 1024 个样本，不引入新线程或无界队列 | 无逐帧 JSON 模式的应用集成测试通过；禁用 UART 时明确显示未测，不填 0；后端不提供原生计时时显示 null |
| R5：测试环境 | 仅当 ELF 的 `e_machine` 与主机架构不符时跳过原有 7 个 ARM64 ABI 用例，实际 Nano 仍执行它们 | 完整 `unittest discover` **126 项 / 119 通过 / 7 跳过 / 0 失败**；`python -m compileall -q app.py python` 通过 |
| R6：独立二次审查 | 核对模型与五个预编译 `.so` SHA-256、输出/协议合同；重新 g++ 编译并执行 C++ NMS 对照 | 模型及全部五个库 r37/r38 哈希一致；NMS 50/50 对照一致（本机 r37 NMS 优化后与未优化算法参考，116.437→54.879 ms；并非 r38 新贡献）；7 张图 × 0/1/15/32 模拟 YOLO 框离线处理输出与 r37 一致 |

## R6 离线 CPU 链验证

命令：`python3 tools/bench_r38_pipeline.py --baseline-root /path/to/r37/jetson_vision --iterations 12`。每个目标数在同一 7 张 Astra 图片上交错执行 r37 与 r38，并在计时前逐项比较完整处理结果（去除实时计时字段）。前端使用**固定模拟检测框**；既不加载 TensorRT，也不执行原生 metric 模型拟合。这是 Python 回退路径的端到端 CPU 压力测试，不是实际 Nano 服务性能：

| 检测框数 | r37 P50 / P95（ms） | r38 P50 / P95（ms） | P50变化 |
|---|---|---|---|
| 0 | 4.012 / 4.497 | 4.231 / 7.026 | -5.46%，微小波动/无收益 |
| 1 | 14.510 / 17.245 | 8.389 / 9.164 | +42.19% |
| 15 | 75.629 / 84.603 | 38.782 / 42.674 | +48.72% |
| 32 | 166.415 / 256.906 | 74.579 / 87.024 | +55.18% |

样本数每组只有 12 次，P95 仅供定位长尾，不能据此推断部署时 P95。旧实现的 Python 几何回退在本基准里承担绝大部分开销；此实验不能用来宣称正式配置提升了 48% 或 55%。

## 曾提出但未实施 / 回退的想法

- 未通过跳过 HSV 漏检补检、放宽检测阈值、删安置区几何或降分辨率刷 FPS，正确性风险不可接受。
- 未改原生 `metric_cube` 的全目标调度：其实际耗时、相对近景收益尚无 Nano 测试，不能按模拟 Python 回退的时间做决定。
- 未添加采集/推理线程池、无界队列、半包回滚、自动 GPU 参数改动；原工程已经使用有界最新帧采集与后台样本写盘。
- 未把 0 目标离线路径的细微耗时反向差异当作成果；没有移除必要功能来优化它。

## 运行记录与真实阻塞

- 本机 `g++ -std=c++14 -O3 -Iinclude src/detection_postprocessor.cpp tests/nms_equivalence.cpp` 编译并运行成功。
- TensorRT 依赖的 detector、Astra SDK 和带 OpenCV 的本机 C++ 原生链接未执行：当前 x86 环境缺少 Jetson 对应 SDK/头文件；用户提供的 `.so` 为 ARM64。
- 尚无真实相机、物理 UART 到 STM32 解包、GPU 使用率/热节流、摄像头硬件时间戳；板端与全链联调未测试。
- 当前阶段无明确能在 x86 主机上验证的**生产原生**高收益改动。继续修改 `metric_cube` 或 TensorRT/CUDA 时须先在 Nano 取得分段数据、编译并做等价性/精度比对。

## r39：按用户提示词第二部分执行的独立复审

| 轮次 | 修改或重新检查 | 真实结果、保持或回退决定 |
|---|---|---|
| R7 独立事实核对 | 重读 `app.py`、`runtime.py`、`continuous_uart.py`、`pipeline.py`、`geometry.py`、`green_detection.py`、`frontends.py`、`zones.py`、C++ TensorRT/NMS/metric；确认 r38 的 15/32 目标回退 CPU 基准**不含**正式 native metric/YOLO | 复测 r38 **126 / 119 PASS / 7 SKIP / 0 FAIL**。原生调用不可在 x86 链接，未宣称已测 GPU 或 STM32 接收 |
| R8 生产几何 CPU 热点 | `geometry.py:Geometry.set_position` 在 `use_pinhole` 时用相同 K 的透视方程代替逐目标 `cv2.projectPoints`；非针孔/畸变分支保持原始 OpenCV 调用 | 500 个随机坐标在 `atol=1e-10px` 通过；开发机交错 200 坐标/批，40 批最终复验 P50 **14.297→0.890ms**（-93.80%），不是整机性能 |
| R9 ROI 复用 | `pipeline.py:top_quad` 允许复用属于相同未模糊图像的已有 HSV；未做 preprocess 相等假设，模糊分支保持独立 ROI 转换 | 测试真实六图、人工凸四边形、靠边/越界框：返回角点一致；该路径在正式 native metric 模式下不是主热点，不单独声称 FPS 收益 |
| R10 端到端诊断独立核查 | `app.py` 在现有固定 1024 滑窗里新增 `monitor_submit`、`frame_json_log`、`sample_submit` 三个 P50/P95，用于找到已启用 ROS/网页/采样对下一帧的尾部阻塞；关闭时为 null | 注入 2/3ms 观察者延迟，阶段 P50 正确检出；`app_write_result_age` 不被错算入串口写入耗时，关闭逐帧 JSON 时不伪造 JSON 写盘耗时；没有新增后台线程或无界队列 |
| R11 独立回放 A/B | 七张 Astra 图、固定 0/1/15/32 框，r38/r39 的稳定处理结果深度比对。仅允许 `ground_center_projection_px` `≤1e-9px` 浮点舍入，其他字段必须精确相等；UART 每帧要求完全相同 | **28/28 比对一致**；x86 50 次交错基准见 `docs/r39_x86_benchmark.json`，50 次交错代表性基准：15 框 P50 39.560→37.724ms，32 框 75.026→73.917ms。0/1 框 P50 反而变慢；另两次交错重跑 15/32 框均获 4%–5% P50 改善。离线测试的不稳定性如实保留。该模拟路径没有 TensorRT / native metric |
| R12 拒绝的无收益尝试 | 针对橙色稀疏候选 RGB 比率，尝试把浮点阈值改为完全等价的整数交叉相乘（全部 256×256 值对一致） | 100–2,000 个候选的典型小区域在本机更慢（例如 2,000 候选 11.66→12.88μs），**未合入**生产代码；其余 CUDA/线程池/模型分辨率改动缺少实机验证，未乱改 |

## r39 补充兼容性与应用级验证

| 轮次 | 修改或重新检查 | 真实结果、保持或回退决定 |
|---|---|---|
| R13 最小环境兼容性 | 新增回归测试的随机场景改用 `numpy.random.RandomState`（Nano 旧 NumPy 可用）；原 r38 UART 随机向量测试的 `Random.randbytes`（Python 3.9 才有）改成 `getrandbits`（Python 3.6 可用） | 不改变任何生产源码/CRC，**131 项中 124 通过，7 项架构不符跳过，0 失败**；且文件校验通过 |
| R14 真实图像应用入口烟测 | `app.py --mode color --calibration configs/calibration.json --manifest data/approved_six/manifest.json --max-frames 2 --uart-loopback --quiet --no-frame-log` | x86 可运行色彩入口 2 帧、PTY 应用写入 2 帧，退出 0；这是 **color 模式 / 失效标定** 烟测，不冒充 TensorRT 或真实 STM32 接收 |
| R15 C++ 与数学复核 | `g++ -std=c++14 -O3 -Iinclude src/detection_postprocessor.cpp tests/nms_equivalence.cpp`，同类 NMS 50 组；橙色 ROI `top_quad` 可复用同帧 HSV | **50/50 同输出**（既有 r37 优化），r39 ROI 局部示例 646.6→423.9μs P50，四角不变；该 ROI 在生产 native metric 路径下通常不会执行，不能外推 FPS |

完整独立审查的第二次/第三次本机交错重复结果保存于 `docs/r39_repeat_benchmarks.json`。0、1 目标时的时间并不稳定，本轮不以该项计算整体加速，也不擅自删除必要的颜色补检。

## R16 交付包独立复验

- x86 完整测试：**131 项，124 通过，7 项仅因 ARM64 ELF 与 x86_64 架构不同跳过，0 失败**；`compileall` 与 `tests/verify_sources.py` 均通过。
- 将 r39 完整包解压至独立目录，重新运行来源与六图像素身份校验及全部 131 项测试，结果一致。
- 将 r38→r39 Git 二进制补丁应用到另一份独立 r38 工作副本：**19/19 个差异文件字节一致**，来源身份校验通过。
- 原模型、五个 ARM64 `.so` 以及提供的图片素材：共 18 个文件与 r38 逐字节一致。
- 独立 `g++ -std=c++14 -O3` 编译后复测 NMS：50/50 同输出；属于 r37 已有优化，不计作 r39 的新增性能。
- 尚未执行的设备验证：Nano TensorRT/Astra、生产 native metric、完整物理 UART 链与 STM32 实际解包。

## r40：本机真实像素 ROI 热点优化与基准

1. 原始 r39 完整工程独立解压，原生模型及旧库只读保存。本机 r39 单测基线：124 passed、7 skipped、45 subtests passed。
2. `frontends.orange_visible_proposals`：HSV、颜色阈值及面积/长宽过滤不变，形态学之后利用 `cv2.boundingRect` 缩小 8 连通域标号范围。通过全局像素整数矩恢复中心坐标，确保完整检测字典逐项一致；密集橙色覆盖全画面时不宣称加速。
3. `frontends._blue_context_region`：同样只排除零像素外边界；内部裁剪是否触边仍依据原始上下文 ROI，而不是新的最小包围框。原候选 bbox、context_indices、中心和顺序均保持一致。全帧候选/单候选、随机颜色覆盖和边缘切割做了 A/B。
4. `zones._color_hull`：使用所有外部轮廓边界点求凸包，代替扫描所有掩模像素。对凸包起点按 `x` 最大、`y` 最大旋转，与原 `cv2.convexHull(cv2.findNonZero(mask))` 的输出顺序一致；真实蓝区、合成蓝/紫区、碎片/孔洞、随机 160 个掩模逐元素相等。
5. 新增 `native_metric_prepare`、`native_metric_orange_fit`、`native_metric_cube_fit` 耗时分段，加入主应用最近 1024 帧统计；以模拟原生函数调用验证 prepare 一次、橙块 fit 一次、绿/蓝 cube 两次，不改变实际调用次数。未连接原生库时不伪造实机计时。
6. 完整离线流水线 A/B：七张真实 Astra 像素图 × 0/1/15/32 固定模拟检测数量；`tools/bench_r38_pipeline.py` 同输入逐项对比字典（排除时间字段），UART 逐字节相等。50 次交错 P50/P95 见 `docs/r40_hotpaths_ab.json`。
7. 未采用的改动：未删除所谓重复立方体拟合——静态追踪证明 `pose_order` 只为橙块排序，绿/蓝走另一条 cube 路径，并非同目标重复。没有在无 TensorRT 的 x86 上重写 CUDA/GPU 内核或改采集/任务门槛。

最终自动化测试与交付包校验记录见 r40 审查报告。以上均非 Nano 实测有效输出 FPS。

## r41：生产原生 C++ 离线复审与优化

1. 重新剖析 0/15/32 固定框离线 CPU：Python `cube_bbox_center` 在无 native solver 的压力模式下主导耗时；该模块是参考/实验回退，**不能把 15 框 CPU 减少时间直接当作正式 Nano 的瓶颈**。放弃只针对 Python 回退刷分的尝试。
2. 在纯 `vendor_model/src/model_geometry.cpp` 中合并遮挡和边界距离共同像素取整/索引；在重叠率计算时按候选凸包外接框跳过不可能在凸包内的背景点，保留所有采样门限与接受流程。
3. 从源代码编译 r40/r41 两版原生模型，实现无 TensorRT/Jetson 的 C++14 测试运行器，使用 42 真实 Astra 像素但模拟检测框 + 80 合成模型/遮挡/噪声场景；严格比较拟合输出。
4. 122 场景、30 轮交错，P50 0.6946→0.5657ms，P95 1.3033→1.0467ms，77 场景拟合有效且状态/位置/yaw/cost/诊断一致。
5. 在 80 合成场景运行 ASan+UBSan 通过；独立原生 C++ 编译和 Python 3.6 语法检查通过；原始 ARM64 二进制未触碰。
6. 完整测试 127 passed / 7 skipped / 45 subtests，详情 `docs/r41_offline_optimization_report.md`。后续 Nano 构建/真实 FPS 不在本机伪造。

- r41 完整离线七图 × 0/1/15/32 检测框共 28 种输出逐项相等；color 模式应用入口 2 帧 UART PTY 回环无 transport_error，非 TensorRT/STM32 物理链实测。详细记录见 `docs/r41_offline_optimization_report.md`。
