# r38→r39 契约兼容性与部署回退

## 明确保持不变

- `configs/runtime.json` 保留 engine `models/blocks_and_zones_astra_v2.engine`、模型 SHA-256、模型通道→业务类型 `[2,3,1,0,4,5]`、逐类阈值 `[0.25,0.5,0.12,0.65,0.1,0.35]`、NMS IoU=0.45、近景调度、默认配置 `placement_zone_color=blue`、Astra 标定与 230400 波特率。r38 仅将 `baseline_id` 更新为 r38，r39 继续更新为 `rgb-unified-20261121-r39-independent-audit`；**没有改比赛逻辑、类别或目标类型**。
- UART 仍是 `AA 55 | N:u8 | N × <Bhhh (7 bytes) | CRC16/CCITT-FALSE little-endian`，N∈[0,32]。CRC 覆盖 `N + N×record`，poly=0x1021、init=0xFFFF、无反射、xorout=0。0/1/15/32 条与 r37 完整字节相同，>32 仍明确拒绝/由上层裁剪，不允许缓冲越界。
- 线上绿色 0、黑色 1、蓝色 3 yaw=0；橙色 2 使用既有模型朝向，180°周期；己方物资/伤员 4/5 保留 300×300mm 子区中心和指向区内法线的 `yaw_cdeg`。模型类 4/5 是模型语义，不等同通讯类型 4/5 的内部含义；视觉已有映射不变。
- 黑色物块仍不依赖 Astra depth；蓝/橙 HSV 补检、绿色拥挤区域处理、安置区优先两条目标、近景几何与有效性门槛不变。
- 模型、五个 ARM64 `.so` 和相机标定内容的 SHA-256 均与 r37 相同；没有要求修改 STM32 端源码/接线/波特率。`ContinuousSink` 和 `SerialSink` 的写入语义不变。

## 本轮行为变化（仅内部性能/诊断）

- `uart.py:crc16` 查表取代 bit-by-bit，同计算结果；`pipeline.py` 使用同模块已有单条记录校验 `_pack_target`，且仅在全部目标已按同序进入协议帧时复用预览字节。
- `shape_center.py:cube_bbox_center` 只改变针孔模型的数学实现，空间边界、候选选择、尺寸和返回字段不变。带畸变/非针孔求解仍沿用旧数值方法。198 个候选几何与 r37 等价，已保存 41 组 golden 测试向量。
- `pipeline.py:top_quad` 只改内存布局，返回的四顶点顺序、条件不变。
- `app.py` 仅追加端到端测量字段，原有 `timings_ms`、`uart_hex`、`transport`、`run_summary` 已存在字段保持；`run_summary.json` 新增固定窗口分位数；关闭物理/PTY 串口时不填伪造的 `write_ms`。
- `tests/native_support.py` 让开发机原生 ABI 不匹配测试显示 `skipped`；Nano/ARM64 环境原有测试仍会执行。这是测试报告准确性修复，**没有移除任何 ARM64 测试**。

## 复验范围

- 可在 x86 完成：Python 语法、完整可运行单元测试、虚拟 PTY、6 张 Astra 图 + 1 张区图对应 0/1/15/32 模拟框输出对照、golden CRC 字节、合法/非法几何、50 组 C++ 纯 NMS。
- 未在当前环境验证：Astra 的真实帧率与硬件曝光时间戳，TensorRT 8.2 实际耗时，重新编译原生动态库后 ABI，真实 ROS 订阅与网页并发负载、真实 UART 物理波形和 STM32 接收端解包。

## 构建、运行、回退

**测试环境**：`python3 tests/verify_sources.py && python3 -m unittest discover -s tests -v`。打包原生库为 ARM64，x86 开发机无需/不能加载。若要板上重新构建 **r37 已有** 的 detector C++ 优化，可在 Nano 的原生工具链中执行：

```bash
cmake -S . -B build_perf -DWITH_TENSORRT=ON
cmake --build build_perf --target vision_detector -j2
# 对新产出的 .so 进行独立 smoke/等价测试后，再在维护窗口切换引用；不要先覆盖唯一旧库。
```

安全的 Nano 采集与 UART PTY 测试（PTY 不向机器人电机控制器发物理数据）：

```bash
python3 app.py --camera astra --uart-loopback --warmup 3 --max-frames 240 \
  --quiet --no-frame-log --sample-interval 5 --output /tmp/vision_r38_bench
```

正式环境仍使用原 `jetson-vision.service` 的物理 `/dev/ttyTHS1` 及 ROS 发布/监控等功能。验证前应先在**安全、不运动**的台架上分别运行 r37/r38；不能仅因 PTY 成功就宣称 STM32 解析成功。对比服务时保持相同功耗、系统负载、阈值、模型和输入场景。

**回退**：保留完整 r37 目录；在已停止服务且备份配置的维护窗口，切回原 r37 目录/文件并重启服务。不要同时运行两个争用 Astra/UART 的进程。若只覆盖了 Python 文件，可用 `r37_to_r38.patch` 反向恢复；原 `models/`、`build/`、`configs/` 无语义改动。

## r39 独立二次审查的兼容性结论

- 未更改 `uart.py`、模型权重、模型 SHA-256、类别映射、检测阈值、配置相机内/外参、己方安置区选择、量化单位、长度上限、CRC、零目标包，亦未修改 ARM64 `.so`、C++ ABI 或 `jetson-vision.service`。
- `geometry.py` 只改变**针孔模型诊断像素投影**的计算方式；标定空间中心与 `x_mm/y_mm` 严格一致。畸变相机仍调用 OpenCV，所有返回字段保持不变；投影像素最后一位浮点数值可能相差 `1e-13`，测试采用明确 `1e-9px` 容差。
- `pipeline.py` 的 `top_quad` 增加尾部可选参数 `hsv=None`，既有调用方式仍受支持；只在 HSV 与传入 image 对应同一原始帧时复用。输出角点、安置区法线、橙色 yaw 及全部协议帧不变。
- `app.py` 新增的三个阶段统计仅增加可选诊断字段；未改变 `--quiet --no-frame-log` 正式模式的图像处理、UART 发送、监控/采样开关与启动命令。
- 仅 x86 可运行单测和仿真 PTY/固定像素等价性通过；**未**在 Jetson Nano 实测 TensorRT/原生几何性能、真实相机连续启动、真实串口物理 drain 和 STM32 现场解包。r38 可随时以保留的完整目录切回；不需要修改 STM32 固件。

## r40 兼容核验

- 代码变更限蓝/橙补检 ROI 裁剪、安置区轮廓构建和原生 metric 子阶段计时。`configs/runtime.json` 仅调整版本标识 `baseline_id`，不改变类别、阈值、尺寸、近景参数或串口设置。
- 原 YOLO 六类映射 `[2,3,1,0,4,5]`、己方安置区语义、UART 0/1/15/32 容量、CRC16/CCITT-FALSE、yaw 量化和字段序列完全保留。
- r39/r40 七张 Astra 像素 × 0/1/15/32 模拟模型框离线流水线中的稳定对象和 UART bytes 完全相等；真实蓝/紫安置区及随机像素轮廓顶点逐元素相等。
- r39 原有模型权重/引擎和 ARM64 `.so` 字节身份不变；不保证未经本轮实机验证的 Nano FPS、视觉泛化精度或真实 STM32 解析结果。`timings_ms` 仅新增若干诊断字段，老字段/原协议保持兼容。

## r41 原生优化兼容性声明

新增原生 C++ `fitGroundModel` 内部循环优化，不改类映射/位置/角度/输出 JSON 字段/CRC16/CCITT-FALSE/STM32 230400 和 32 目标上限。122 场景比较的状态、位置、yaw、候选数量和相关诊断与 r40 严格一致。`configs/runtime.json` 仅更新版本标识，model、标定和原始 ARM64 动态库完全不动。未经 ARM64 编译，不能冒称已经在 Nano 验证 native ABI、整体吞吐和 UART 接收。

- r41 完整离线七图 × 0/1/15/32 检测框共 28 种输出逐项相等；color 模式应用入口 2 帧 UART PTY 回环无 transport_error，非 TensorRT/STM32 物理链实测。详细记录见 `docs/r41_offline_optimization_report.md`。
