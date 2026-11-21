# r37 → r39 视觉性能报告（不将离线 CPU 等同 Jetson FPS）

版本：r37→r38，2026-11-21。测量设备 x86_64 / Python 3.13.5 / OpenCV 4.13.0（OpenCV 单线程）；其他设备指标均标注为未测。仓库的旧历史记录并不是新测基线。

## 指标定义与时钟

- `run_summary.json.observed_loop_hz`：从运行循环开始到停止时应用完成帧数 / 时间；包含取帧等待、计算、串口/采样/限速等应用开销，不等同模型 FPS 或 STM32 接收 FPS。
- `latency_ms.capture_handoff_age`：`read_done`－采集子进程 SDK `read()` 返回时刻；没有曝光时刻，不得称传感器延迟。离线视频/静态图同样属于软件交付起点。
- `latency_ms.app_write_result_age`：应用层 `sink.write()` 返回－采集子进程 SDK `read()` 返回时刻；**不含物理 UART 移位寄存器清空和 STM32 解码**。
- `latency_ms.source_wait`、`pipeline_compute`、`app_serial_write`、`frame_to_write`：分别对应源等待、`Pipeline.process`、Sink 应用写入、循环开始到写入结束。断开 UART 时写入类指标为 `null`，不伪造 0 ms 结果。
- `latency_stage_ms`：检测、中心提取、近景细化、metric 验证、UART 编码、橙/蓝 HSV 补检、绿色局部处理与可用的 C++ 原生预处理/TensorRT&传输/NMS 后处理计时。C++ 三段仅在板上编译新版库后可用。
- P50=中位数，P95=向上取整第 95% 秩次。仅保留**最近最多 1024 帧**，`samples` 为实际窗口内样本数；不是无限历史量化数据，也不保证低频异常帧被长期保留。

## 数据主表

| 指标 | r37 | r38 | 证据范围 |
|---|---:|---:|---|
| 真实有效 UART 结果 FPS | 未测 | 未测 | 无 Nano 硬件 |
| 相机读出→应用写出 P50/P95 | 未测 | 未测 | r38 已加入板上测量字段 |
| 相机帧龄 P95 | 未测 | 未测 | r38 已加入测量字段，非曝光时间 |
| TensorRT+传输 P50/P95 | 未测 | 未测 | 需 Nano 重编译原生检测库 |
| 原生 metric_cube 耗时 P50/P95 | 未测 | 未测 | 需要板上单独热点剖析/验证 |
| 串口编码，N=15 P50 | **83.007 μs** | **22.243 μs** | 同主机交错 A/B，输出字节相同 |
| 串口编码，N=32 P50 | **176.986 μs** | **45.847 μs** | 同上 |
| Python 立方体回退单次拟合 P50 | **9.761 ms** | **3.823 ms** | 198 个合成几何、不是生产 native metric |
| Python 模拟流水线，N=15 P50 | **75.629 ms** | **38.782 ms** | 7 张真实像素+模拟框，禁用 TensorRT/native metric |
| Python 模拟流水线，N=32 P50 | **166.415 ms** | **74.579 ms** | 同上，不能代表整机 FPS |
| CPU/GPU/峰值内存/温度 | 未测 | 未测 | 缺少板上监测条件 |
| 输出语义 | r37 | 相同 | 7 图×4 种压力配置逐项相等；单测/几何 golden |
| 串口格式与 CRC | r37 | 相同 | 0/1/15/32 记录完整帧逐字节相等，随机 CRC/golden 向量 |

## CPU 的可复现基准

```bash
python3 tools/bench_wire_r38.py --baseline-root /path/to/r37/jetson_vision --iterations 700
python3 tools/check_shape_center_r38.py --baseline-root /path/to/r37/jetson_vision
python3 tools/bench_r38_pipeline.py --baseline-root /path/to/r37/jetson_vision --iterations 12
python3 -m unittest discover -s tests -v
```

- 模拟流水线测试里真实 Astra 图像只提供像素，`Detector.run` 返回固定假框，**不能评价模型检出率/识别精度**；7 图各含 0/1/15/32 模拟输入。回退 1/15/32 项收益主要来自针孔投影与线性求解代替 Python 调用 SVD；0 输入波动不具有实质收益。
- UART P50 分别下降 73.20%（15）和 74.10%（32），但绝对节省约 61μs / 131μs；整机 100 ms 级别时不应夸大其 FPS 贡献。40/50/60%的模拟流水线收益不是生产态结论。
- `top_quad` 橙色 ROI 广播在一张人工构造合法色块上 P50 414→250μs，输出四角相同。该步骤只在 `edge_candidate` 且后续几何验证时运行；未建立其生产频率，不能累加到每帧节省。
- C++ NMS 50/50 稳定相等，本机参考/当前为 116.437/54.879ms（合成候选集总耗时）；这是**r37 已有**的优化，本轮只复验，不纳入 r38 增益。

## 瓶颈排序与风险

| 优先级 | 源码路径 / 函数 | 证据 | 处理决策 |
|---|---|---|---|
| 1（待实机确认） | `src/tensorrt_engine.cpp:infer`, `src/detector_bridge.cpp:run_detector` | 早期记录检测约 69ms；每次同步 H2D/推理/D2H。旧记录并非 r37 新实测 | 保留 engine/context 复用、模型和精度；用原生分段时延判断真实比例，再决定是否更改同步/缓冲。未在 x86 胡乱改 CUDA |
| 2（待实机确认） | `src/metric_bridge.cpp:metric_prepare`, `metric_fit_json`, `metric_cube_json` | 早期细化约 34ms；本机没法加载 ARM64 metric lib | 不跳过近景定位；需在 Nano 用 0/1/15/32、不同距离和稠密场景定位耗时 |
| 3（局部已优化） | `python/rescuevision/pipeline.py` 目标遍历 → `uart.encode`；`shape_center.py` | 逐目标完整编码和回退求解可在 x86 上复验 | 已减少编码重复工作与 CPU 回退代价；保留 ABI 与角度/类别语义 |

## 对比、回退及限制

将 **同一** r37/r38 模型、相机、功耗模式、ROS/监控/采样设置分别运行等量实机帧。观测 `run_summary.json` 新增的 `latency_ms`、`latency_stage_ms`，旧 r37 没有这些聚合字段，比较 r37 时应在同样配置下读取旧逐帧 JSON（不能强行填不存在的值）。需要曝光起点时由 Astra SDK 可信时间戳另行标定。物理串口不能仅靠 `write()` 返回当作真正 STM32 接收成功；该项需要 STM32 实际解析记录。

当前不存在被证据证明适合直接投入生产的更多 x86 高收益原生优化；未测试项不填估计值。无独立场景泛化评估，不能宣称比赛检测准确率提高。

## r39 第二部分：独立审查与最新 A/B（x86，不是 Nano FPS）

基线 r38 与 r39 共享相同 Astra 七图、配置、YOLO **模拟检测框**，OpenCV 线程数=1；x86_64 Python 3.13.5 / OpenCV 4.13.0，交错 A/B。运行：

```bash
python3 tools/bench_r39_audit.py --baseline-root /path/to/r38 --iterations 50
```

代表性完整结果保存为 `docs/r39_x86_benchmark.json`，多轮复测见 `docs/r39_repeat_benchmarks.json`。后者用于显示小目标数不稳定，**不择优挑选单次最快成绩**。

| 指标 | r38 P50 / P95 | r39 P50 / P95 | 备注 |
|---|---:|---:|---|
| `Geometry.set_position`，200 次/批 | 14.297 / 15.837 ms | 0.890 / 1.166 ms | 40 批，P50 -93.78%，生产可调用的 Python 子步骤 |
| 0 个模拟检测框流水线 | 3.959 / 5.136 ms | 4.451 / 5.769 ms | 可能仍有 HSV 橙色补检，**本次退化**；多轮中波动明显 |
| 1 个模拟检测框流水线 | 8.077 / 9.990 ms | 9.112 / 11.791 ms | **本次退化**；另两次重复结果为改善，不能认定稳定收益 |
| 15 个模拟检测框流水线 | 39.560 / 42.362 ms | 37.724 / 41.035 ms | P50 -4.64%；其他重复约 -4.4% 至 -4.6% |
| 32 个模拟检测框流水线 | 75.026 / 97.458 ms | 73.917 / 84.819 ms | P50 -1.48%；另两次重复约 -4.9% 至 -5.2% |
| 真实 Nano 有效输出 FPS / 帧龄 P95 | **未测** | **未测** | 无 Nano 和 Astra 物理设备 |
| TensorRT/原生 metric 性能、GPU/温度/节流 | **未测** | **未测** | ARM64 `.so` 在 x86 不可运行 |
| 物理 UART drain / STM32 实际解包 | **未测** | **未测** | PTY 回环不是 MCU 联调 |

新增 `run_summary.json.latency_stage_ms.monitor_submit`、`frame_json_log`、`sample_submit`，分别统计主线程提交监控、逐帧 JSON、异步样本提交的耗时，按最近最多 1024 帧计 P50/P95；禁用功能时为 `null`，不伪造 0ms。注入各 2ms/3ms 的虚拟观察者阻塞后，可独立检出真实提交成本，同时不改变 `app_write_result_age`（仍以串口应用写出为终点）。

**输出与几何等价范围**：`Geometry.set_position` 的针孔诊断重投影 500 组随机位置 `atol ≤1e-10 px`；畸变模型仍调用原 OpenCV；七张 Astra 图片 × 0/1/15/32 模拟检测框，共 28 个完整稳定输出逐项比对，除了诊断 `ground_center_projection_px` 的 `≤1e-9px` 浮点末位，其他字段严格相等，`uart_hex` **逐字节一致**。本次没有运行 YOLO、原生 metric 或做真实标注泛化评估，不能拿输出对照证明识别准确率提高。

**局部 ROI**：在有预先计算 HSV 且进入 `top_quad` 轮廓判定的合成橙色候选上，原路径 646.6μs、复用路径 423.9μs P50（约 -34%）；角点完全一致。此路径属于非 native/实验细化分支，非当前生产每帧必经步骤。

**回退的尝试**：把橙色 RGB 比率门槛改成整数交叉相乘，全 256×256 分量值对逐项语义等价，但 100–2,000 候选区间的 x86 计时更慢，未合入；不改变 TensorRT 精度、模型分辨率、近景几何准入、蓝/橙/绿色兜底、CRC、波特率和串口数量限制。

**当前停止条件**：已完成当前 x86 可证明有收益且兼容的局部优化和可复现测试。剩下真正可能支配比赛实时性的 TensorRT 同步传输与原生 metric 需要 Jetson 新编译/分段计时；本机缺少 CUDA、TensorRT 和 SDK，未经测量不把任何采集线程/模型变更投入生产。要完成设备侧最后验收，保持模型、功耗模式、输入场景一致，在 Nano 上运行 240 帧正式 UART-PTY 试验并读取 `run_summary.json` 的 FPS、帧龄及各 P95；要验证接收正确还需真实 STM32 解析记录。

## r40：独立 CPU 热点测试（不包含 TensorRT 和原生 metric）

同一台 x86_64 Linux、Python 3.13.5 / OpenCV 4.13.0、单线程、7 张实际 Astra BGR 图片，等价固定检测框，两版交错执行 50 次。数据见 `r40_hotpaths_ab.json`、`r40_full_pipeline_ab.json`。

| 阶段或负载 | r39 P50 | r40 P50 | P50 下降 | 口径 |
|---|---:|---:|---:|---|
| 橙色补检（Astra 01） | 3.421 ms | 1.564 ms | 54.28% | 单个 Python/OpenCV 阶段 |
| 蓝色补检（Astra 01, 15 框） | 2.848 ms | 1.871 ms | 34.29% | 同上 |
| 蓝色区域几何（Astra 现场帧） | 1.224 ms | 0.355 ms | 70.99% | 安置区处理阶段 |
| 0 目标 CPU 离线流水线 | 3.584 ms | 2.132 ms | 40.51% | 七图、模拟 YOLO 框、无原生几何 |
| 1 目标 CPU 离线流水线 | 6.872 ms | 4.418 ms | 35.70% | 同上 |
| 15 目标 CPU 离线流水线 | 22.342 ms | 19.771 ms | 11.51% | 同上 |
| 32 目标 CPU 离线流水线 | 42.351 ms | 39.356 ms | 7.07% | 同上 |

**不能将上述 CPU 加速比例换算为 Nano 实际 FPS**。真实 TensorRT 传输与算子耗时、原生 metric 耗时、相机时间戳/帧龄、UART 物理发送、CPU/GPU 温度和时钟频率仍未在本轮测量。原有校验与串口报文字节不变；新增 native metric 子阶段计时需在 Nano 上运行才能收集数值。全画面高密度橙色区域几乎没有 ROI 裁剪收益，相关优化未声称全面有效。

## r41：生产原生 RGB 模型拟合纯 C++ 本机 A/B

在相同 C++14/O3 编译器、同一 x86_64 CPU，以交错顺序运行 122 个场景 30 次（42 实拍像素/固定模拟框，80 几何生成 RGB 场景），严格核对完整拟合输出：

| 项目 | r40 P50 (ms) | r41 P50 (ms) | 减少 |
|---|---:|---:|---:|
| 全部场景 | 0.6946 | 0.5657 | **18.56%** |
| 成功拟合 77 场景 | 0.9114 | 0.6870 | **24.62%** |
| 拒绝拟合 45 场景 | 0.4804 | 0.4196 | **12.66%** |

全体 P95 1.3033→1.0467ms。这是 `fitGroundModel` 原生内核局部运行时间，尚未在 Nano 编译/激活，不能推断整机 FPS；TensorRT 和 `metric_cube` 没有纳入本轮基准。`docs/r41_native_model_ab.json` 有机器可读原始概要，完整说明见 `docs/r41_offline_optimization_report.md`。
