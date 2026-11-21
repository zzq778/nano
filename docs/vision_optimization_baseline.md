# Jetson 视觉侧性能优化基线（r37 → r38）

日期：2026-11-21。工作副本：以用户此前上传并已交付的 `jetson_vision_r37_deploy.tar.gz` 内容为原始基线，在独立目录复制后使用本地 Git 首次提交封存 r37。未覆写用户原件、未上传或公开模型。比对根目录为 `jetson_vision/`。

## 运行链及证据位置

生产服务 `jetson-vision.service` 运行 `python3 app.py --camera astra --sample-interval 5 --sample-limit 120 --ros-image-topic /camera/rgb/image_raw --ros-image-hz 5 --warmup 1 --uart-device /dev/ttyTHS1 --quiet --no-frame-log ...`，并且 `configs/runtime.json` 的 `frontend=yolo`、`geometry_backend=native_known_size_rgb`、`green_crowd_evidence=true`、`placement_zone_color=blue`。

1. `app.py:open_camera` → `runtime.LatestCamera` → `_camera_reader` → `astra.NativeAstra.read`：Astra SDK 属于独立采集进程，进程间共享一个 `640×480×3` 的最新 BGR 缓冲。`LatestCamera.get(after_sequence)` 在锁内复制最新图像，未使用无界旧帧队列。该帧的时间戳来自采集进程 `read()` 返回后，**不是曝光时间戳**。
2. `app.py` → `pipeline.Pipeline.process(image)` → `green_detection.GreenCrowdFrontend.detect` → `frontends.YoloFrontend.detect` → `native.NativeDetector.run` → `src/detector_bridge.cpp:run_detector` → `tensorrt_engine.cpp:TensorRtEngine::infer`：模型创建和暖机一次；运行时复用 engine、context 和 CUDA buffer。原生前处理包含 `resize`、`copyMakeBorder`、BGR→RGB、`split` 与 CHW 浮点转换，TensorRT 通过同步 `cudaMemcpy` 和 `executeV2` 完成推理，结果经过 `YoloRawAdapter::adapt`、坐标恢复及同类 NMS。
3. `YoloFrontend.detect` 在同帧生成一份 HSV：蓝色上下文补检、橙色可见区补检；`GreenCrowdFrontend` 进一步执行绿色面提取、局部原生边缘细化和合并。`Pipeline.process` 再对黑/蓝色中心、黑色尖顶进行局部处理。
4. `Pipeline.process` 在近景优先级调度后对模型可用的候选使用 `NativeMetric.prepare/fit/cube`，应用 `Geometry.estimate`、`best_effort_position`、`complete_orientation`；另用 `zones.estimate_zone_regions` 计算己方放置区入口与两子区中心。**正式 r37/r38 配置中的绿色/蓝色立方体优先走原生 `metric_cube`**，可选 Python `shape_center.cube_bbox_center` 主要用于没有原生立方体拟合的回退/实验路径。
5. `Pipeline.process` → `uart._pack_target` 校验目标 → `uart.encode` → `runtime.packet_for_age` → `continuous_uart.ContinuousSink.write` → `runtime.SerialSink.write` 发送应用层完整帧。固定长度规则 `AA55 | N | 7×N | CRC16LE`，N=0..32，CRC16/CCITT-FALSE 覆盖从 N 开始的字节。
6. 可选 ROS 图像发布、网页监控及五秒一次 PNG/结果配对采样仍然启用；PNG 写盘由 r37 引入的有界后台线程处理，不占用主推理线程的同步磁盘 I/O。

## 输入、构建及身份

- 硬件目标：Jetson Nano 4 GB、Ubuntu 18.04、CUDA 10.2、TensorRT 8.2.1.8、Astra RGB、STM32F407VET6 UART 230400。
- 正式模型 `models/blocks_and_zones_astra_v2.engine` SHA-256：`39624a89792c1be22f7c2bcc847685c6b9bd83288c77a1ee42fc7e07ee1c303a`，`configs/runtime.json` 的六类别通道→业务类型映射 `[2,3,1,0,4,5]`；阈值不变。
- 现有 ARM64 原生库位于 `build/libvision_{astra,detector,green,metric,orange}.so`。本次仅改 Python，不修改本轮 C++ 源码；原生二进制仍为用户提供的 ARM64 版本。上轮 r37 C++ NMS/原生计时代码需要在 Nano 编译才会生效。
- 当前主机：x86_64，Python 3.13.5，OpenCV 4.13.0。没有 Jetson、Astra、真实串口、TensorRT/CUDA 和 C++ OpenCV 开发头文件；不能重新链接 ARM64 动态库。
- 可用固定图像：`data/approved_six/raw/01.png` 至 `06.png`、`data/zone_contract/astra_frame_000000.png`，测试仅本地读取。可用于回归，不能证明独立数据泛化准确率。
- 基线源码/素材验签：`python3 tests/verify_sources.py`；指纹：`BASELINE.json`、`source_identity.json`；完整 STM32 合同：`VISION_WIRE_CONTRACT.md`。

## 基线观察与未测边界

旧档案里的近似 **检测 69 ms、细化 34 ms、计算总 115 ms** 属于早期运行记录，不是当前 Nano r37/r38 的新实测，不能当作公平 A/B 的 P50。真实当前负载下，TensorRT/CPU-GPU 同步、原生几何拟合、相机缓冲新鲜度仍需板上分段测定；本次不能给出 Jetson FPS、端到端曝光时延、GPU 负载、温度或 STM32 解包成功率。

## r39 独立复审基线（2026-11-21）

以已经可独立解压/回滚的 **r38** 工程为比对基线（并非以最初 r35 或早期 Nano 历史耗时冒充新基准）。`r38` 源目录保留原样；`r39` 是独立副本，版本元数据使用 `rgb-unified-20261121-r39-independent-audit`。服务入口、六类 engine、默认蓝色己方安置区、波特率、零目标状态帧和 `MAX_TARGETS=32` 不变。

重新检查了以下真实调用链：`app.main` → `LatestCamera.get`（采集进程共享最新帧，消费端加锁复制）→ `Pipeline.process` → `GreenCrowdFrontend.detect` → `YoloFrontend.detect` → `NativeDetector.run` → `src/detector_bridge.cpp` → `TensorRtEngine::infer`（同步 H2D/executeV2/D2H）→ HSV 蓝/橙/绿色补检 → `NativeMetric.prepare/fit/cube` → `Geometry.set_position` → `zones.estimate_zone_regions` → `uart.encode` → `packet_for_age` → `ContinuousSink.write` → `SerialSink.write`（应用层）。后台 ROS 发布、网页监控、异步 PNG 样本为独立可选路径，但共享 CPU/GIL/内存带宽。

**独立发现的生产 CPU 热点：** `Geometry.set_position` 在针孔模式下每次位置有效时都调用 `cv2.projectPoints`，只为生成地面中心的诊断像素字段。该标定的针孔模型允许用同样内参的直接透视公式替代。新路径保留输入世界坐标、毫米量化、输出键和畸变相机分支；500 随机点对照 `≤1e-10 px`。代表性 x86 200 次函数批处理由 14.297ms 降至 0.890ms P50。该局部测试不能等价为 Nano FPS。

**复核范围限制：** 当前机器没有 ARM64 Jetson/Astra/STM32 物理串口和 TensorRT/SDK 开发链；`build/*.so` 仍是上一版原始 ARM64 文件且没有重新编译。可检验项目是 Python、纯 C++ NMS、PTS/PTY 模拟串口、固定真实 Astra 像素与模拟检测框。GPU 同步、原生 metric 的实机阶段耗时与物理 UART 排空未测。保留旧服务/模型及编译链，不因 CPU 模拟跑分改变模型精度或采集结构。

## r40 本地优化基线（2026-11-21）

- 基线：`jetson_vision_r39_mobile_full.tar.gz`，保留独立原件；未更改旧版 ARM64 `build/*.so`、TensorRT engine、标定数据及 UART。
- 本地环境：Linux x86_64、Python 3.13.5、OpenCV 4.13.0、OpenCV 单线程；Astra 七张 640×480 原始图；YOLO 为固定模拟框，未运行真实 TensorRT/Jetson 摄像头。
- 真正运行链未变：Astra 最新帧 → YOLO 六类 → 蓝/橙颜色补检 → 近景中心/姿态 → 已知尺寸原生几何 → 己方安置区 → UART CRC16/CCITT-FALSE。
- 本轮仅优化 `frontends.py` 蓝/橙补检连通域的像素占据边界，`zones.py` 安置区轮廓凸包，并增补原生 metric 子阶段耗时记录；需要 Nano 原生库才能测该阶段实测值。
- 见 `docs/r40_hotpaths_ab.json`、`docs/r40_full_pipeline_ab.json`、`docs/vision_test_progress.md`；所有数据均为本机 CPU，不是 Jetson FPS。

## r41 原生 C++ RGB model fit 本机基线（2026-11-21）

- 在 r40 完整独立副本中开发，生产路径仍以 `app.py`→`NativeMetric`→`metric_fit_json`→`fitGroundModel` 为准；主模型、类别映射、标定、UART、近景策略及旧 ARM64 二进制未变。
- 由于 x86 没有 TensorRT/CUDA/OpenCV C++ 开发库，仅以独立 `g++` 编译实际生产模型拟合 `vendor_model/src/model_geometry.cpp`（纯 C++14），与同条件 r40 建立输入输出 A/B。
- 122 场景 / 30 次交错；原生处理 P50 0.6946→0.5657ms，成功拟合 77 场景，准确输出严格一致。`docs/r41_native_model_ab.json`。
- 性能必须与正式 Nano `metric_prepare`、`metric_cube`、TensorRT 和传输耗时分开。优化在本轮**未在 ARM64 动态库中激活**。
