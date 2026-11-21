# r41 Source optimization baseline — native RGB fitting

Based on an independent full copy of r40. Only native C++ RGB fitting and host test/benchmark tools were changed; configuration changed only `baseline_id`. The existing ARM64 native libraries and model weights are byte-identical to r40. Evidence: `docs/r41_native_model_ab.json` and `docs/r41_offline_optimization_report.md`. No Jetson runtime result claimed.

---

# r40 Local CPU hot-path baseline

Source: independently preserved r39 complete deployment archive. Changes: exact blue/orange occupied-ROI CCL, exact contour-boundary zone hulls, and separately measured native metric stages. Six-class model and binary/STM32 protocol unchanged. Local benchmark results in `docs/r40_hotpaths_ab.json`; Jetson end-to-end throughput not measured this round.

# 当前工程基线：r39（2026-11-21）

当前基线标识：`rgb-unified-20261121-r39-independent-audit`。

## r38 历史基线

原 r38 标识：`rgb-unified-20261121-r38-wire-fallback-latency`。

在 r37 基线上保留原生 YOLO/NMS、六类模型和 ASM/CUDA 依赖不变，进一步减少 UART 目标校验与 CRC 重复开销、优化 Python 针孔立方体几何回退与橙色 ROI 坐标生成，加入有界的应用层端到端时延 P50/P95 分段记录。

正式配置仍优先使用 ARM64 `metric_cube`，Python 回退路径大幅加速不等价于正式 Nano FPS 提升。未改 STM32 通讯格式、类别/角度语义、阈值、标定或安置区。

详见 `docs/vision_optimization_baseline.md`、`docs/vision_test_progress.md`、`docs/vision_performance_report.md`、`docs/vision_compatibility.md`。原 r36/r37 的审查报告保留在 `PERFORMANCE_REVIEW.md`。

## r39 本轮独立复审

r39 (2026-11-21) 是在完整封存的 r38 上进行的独立二次审查：生产针孔诊断投影 `Geometry.set_position` 去掉逐目标 `cv2.projectPoints`；未模糊的顶面 ROI 复用已有 HSV；`app.py` 附加监控、逐帧 JSON、采样提交 P50/P95。模型/类别/STM32 协议/正式服务命令未改变。完整证据见 `docs/vision_optimization_baseline.md`、`docs/vision_test_progress.md`、`docs/vision_performance_report.md`、`docs/vision_compatibility.md`，性能来自 x86 主机而非 Nano。
