# r41：生产原生 C++ 几何拟合 CPU 优化与离线完整审查

日期：2026-11-21。**全部在本机 x86_64 完成，未连接或修改 Jetson。**
冻结基线是 `jetson_vision_r40_deploy.tar.gz` 的完整副本；r40 原件保留。

## 关键事实

实际 Nano 生产配置使用 `geometry_backend=native_known_size_rgb`，绿色/蓝色立方体优先由原生 `metric_cube_json` 求解，橙色由原生 `metric_fit_json` 完成。r40 离线 Python 参考版 `cube_bbox_center()` 在 15/32 框压力测试中占了大量时间，但不代表真正部署到 Nano 的主要热点。r41 因而没有继续“优化 Python 回退做跑分”，而直接针对生产 `vendor_model/src/model_geometry.cpp` 的 RGB 模型拟合主路径。

## 真实修改

- `Evidence.visibleEdgeDistance(x, y, *occluded)`：合并 `isHidden()` / `at()` 对同一轮廓采样点的像素取整与索引。保留原来“浮点坐标略出 ROI 但四舍五入到内部像素”的遮挡规则，以及距离越界返回 `20 px` 的规则。内部采样点只取整、取索引一次。删去不再调用的 `Evidence.at()`。
- `Fitter::overlap`：预先计算拟合投影凸包的轴对齐外接范围，跳过不可能在多边形中的背景格点 `inside()` 边缘判断；候选凸包内部仍使用原 `inside()`。IoU 的分子、分母与原算法相同。
- 其余 C++ 模型代价函数、颜色阈值、立方体顶点、姿态搜索次数、收敛阈值、遮挡规则、定位/朝向含义、TensorRT 和 UART 均未更改。无新增线程、框架、跨帧追踪或动态配置。

## 纯原生 C++ 模拟测试，不依赖 Jetson

直接用 `g++ -std=c++14 -O3` 编译 r40 和 r41 的 `vendor_model/src/model_geometry.cpp`、`ground_geometry.cpp`，通过 `tests/model_fit_probe_r41.cpp` 运行 **122 种相同场景**：

- 7 张真实 Astra 像素 × 6 类别/位置框（42 种）：框是固定模拟检测结果，**不是 YOLO 实测输出**。
- 80 个按当前真实标定投影生成的合成物块场景（四类、五个距离/横向位置），包含原始投影、1～3 px 检测框偏移、可见碎片遮挡以及像素噪声。
- 原生拟合状态：43 `FITTED`、34 `ENVELOPE`、45 `REJECTED`。前后版本的每个场景：有效射线、状态/原因、搜索评价次数、米制 XY、yaw、拟合误差、可见率、best-effort、候选数量与 footprint 数量严格一致（结果比较排除本次耗时）。
- 30 次交错运行，两套同一构建参数，统计全部有效射线拟合 P50/P95：

| 比较 | r40 | r41 | P50 耗时下降 |
|---|---:|---:|---:|
| 所有 122 场景 | 0.6946 ms | 0.5657 ms | **18.56%** |
| 77 个成功拟合场景 | 0.9114 ms | 0.6870 ms | **24.62%**（按全体样本中位数） |
| 45 个拒绝拟合场景 | 0.4804 ms | 0.4196 ms | **12.66%**（按全体样本中位数） |
| 所有场景 P95 | 1.3033 ms | 1.0467 ms | **19.69%** |

77 个成功场景按**每个场景自己配对**的中位加速为 21.68%；拒绝场景按场景配对为 8.21%。这里的数值是开发机运行“独立 RGB model fit C++ 内核”耗时，不包括 OpenCV `metric_prepare`、白平衡、`metric_cube_json`、TensorRT、真实模型检测、相机与 UART。不同 CPU、编译器优化及 Jetson 温度下不保证相同比例；无需 Nano 的部分已经完成。

原始 JSON：`docs/r41_native_model_ab.json`。复现：

```bash
python3 tools/bench_r41_native.py --baseline-root /path/to/r40/jetson_vision --iterations 30
python3 tools/bench_r41_native.py --baseline-root /path/to/r40/jetson_vision --synthetic-only --iterations 30
```

## 回归与安全

- Linux x86_64 完整单元测试：127 passed / 7 skipped / 45 subtests passed；新 C++ 自编译合成 RGB 样本测试通过。
- `-fsanitize=address,undefined -fno-omit-frame-pointer` 编译并运行 80 个合成场景：退出码 0、无 sanitizer 诊断。
- 源码静态 C++14 编译通过，NMS 独立构建测试通过。Python 3.6 语法解析、单元测试通过。
- 没有修改 `app.py`、`python/rescuevision/*.py`、`src/metric_bridge.cpp`、`src/tensorrt_engine.cpp`、标定数值、模型 engine、UART 字节规则及 ARM64 旧 `.so`；`configs/runtime.json` 仅更新 `baseline_id`。
- r40/r41 完整离线 Python 流水线输出与 UART 报文需要逐字节保持不变；由于生产修改的是未重建的 C++ 模型库，**此回放通过只能验证 Python/配置没有回归，无法证明 Nano 构建后 native ABI/耗时。**

## 实际可用性和严格边界

**r41 包中的 `build/libvision_metric.so` 仍为旧版 ARM64 二进制。** r41 的原生算法优化只存在于 `vendor_model/src/model_geometry.cpp`，必须在 Nano 上将新版源码于独立构建目录重新编译并做原生精度/性能验证后才会体现在运行中。这一轮没有连接 Nano，未激活 r41 Native 路径。

本机没有 TensorRT/CUDA、Astra SDK、Jetson aarch64 链接与真实 STM32，因此无法确认真实 FPS、热节流、UART 完成/解包、模型精度或毫秒级端到端延迟；这些内容保留为后续实机验证，而不是“未完成的本地工作”。

对能离线证明的低风险等价改动，目前已完成。其它可能收益较大的 GPU 预处理和模型参数改动需要真实 TensorRT 环境/精度测量，不凭猜测纳入本轮。

## 交付前应用级核验（补充）

- 7 张 Astra 图片 × 0/1/15/32 固定框：r40/r41 离线 Python 流水线逐项比对 **28/28 完全一致**，包括每帧 UART 十六进制字节；20 次交错计时 P50 (0/1/15/32) r40 分别 1.946/4.255/20.113/40.878ms，r41 分别 1.913/4.302/20.187/41.415ms，符合**本轮没有修改 Python 处理性能**的预期。不能把这些波动算作 native 的收益。
- 应用入口 `python3 app.py --mode color --calibration configs/calibration.json --manifest data/approved_six/manifest.json --max-frames 2 --uart-loopback --quiet --no-frame-log --output /path/to/output`：退出码 0，`run_summary.json` 记录处理帧数 2、应用结果 2、`transport_error=null`。最后一帧字节为 `AA 55 00 F0 E1`，是按既有 CRC16 发出的零目标状态帧；**不是 YOLO 模型识别结果或 STM32 物理收包证明**。
- 应用测试初次由于漏填必需 `--output` 参数而得到 CLI 退出码 2；补全参数后真实执行成功，日志没有隐藏这一步。
- 完整 Python 测试和输入像素/源码身份校验已经通过。完整 Python A/B 记录：`docs/r41_offline_pipeline_ab.json`。
