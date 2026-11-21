## r41 Native RGB model-fit CPU optimization (source-only, 2026-11-21)

r41 keeps the r40 image-processing and UART behavior and reduces repeated indexing and polygon point-inclusion work in the production C++ known-size fitter (`vendor_model/src/model_geometry.cpp`). On a host, 122 unchanged RGB fitting cases ran 18.56% faster by pooled P50, with exact result parity; the full report is `docs/r41_offline_optimization_report.md`.

**Important:** the bundled `build/libvision_metric.so` is the unchanged original ARM64 binary; this release does not activate native performance improvements on Nano until rebuilt and verified there. No production deployment or public GitHub push was performed.

---

## r40 CPU hot-path optimization (2026-11-21)

This revision retains the r39 vision baseline and native model, and reduces CPU work in the orange/blue color supplements and placement-zone hull extraction. It also adds native metric substage timers. See `docs/vision_performance_report.md` for reproducible r39/r40 comparisons. No Jetson runtime FPS claim is made. The local A/B command is:

```bash
python3 tools/bench_r40_hotpaths.py --baseline-root /path/to/r39/jetson_vision --iterations 50
```

---

# 独立 Astra 视觉工程 r39（六类视觉与两子区通信合同）

唯一入口为app.py，生产配置configs/runtime.json，开机单元jetson-vision.service。实际源码身份以BASELINE.json和source_identity.json为准；旧版本说明已归档，不作当前规则。

## 当前流程

Astra SDK直接采集640×480 BGR最新帧 → 六类普通YOLO检测 → 绿色顶面/双边缘补检及已证实群父框剔除、蓝色上下文补检 → 中心/姿态计算 → 普通UART目标帧。没有强制数量，群框不冒充单块。

绿色与蓝色只求中心：优先用已知4cm立方体外框求中心，旋转只作内部形状变量，不输出绿色姿态。黑色优先用顶点和已知四面体高度求中心。橙色继续已知尺寸模型拟合与角度求解。精确位置失败后保留有限的模型最佳解、绿色顶面高度估计，或基于可见中心/已知高度与占地范围的粗略估计。position_valid表示有可用估计，不表示精度验收通过；position_quality、prior_position_reason、accuracy_verified等保存在日志。未知标定或射线无有效地面交点时不凭空补坐标。

坐标是物块三维几何中心的垂直地面投影，相对轮轴中心的X/Y，单位mm；+X车头、+Y车左。当前Astra内参fx493.3689、fy494.2357、cx327.3957、cy234.8487；光心(27,0,129)mm，roll0/pitch_down31.6/yaw0，地面Z=-36mm。畸变和独立测距精度未验收，保留估计标记，不因此停止发包。旧六图用其独立绑定标定；外部多物块照片仅作显式虚拟相机压力测试。

## UART与区域

/dev/ttyTHS1，230400/8N1，AA55｜N｜N条7字节记录｜CRC16/CCITT-FALSE，小端。仅N=0..32普通帧；故障帧生成/识别已移除。启动、预热、无目标、采集或单帧处理异常时发送N=0（AA5500F0E1），空闲守护间隔1秒，不重复旧目标。单体位置不可用或越界只省略该目标，区域两条记录优先成对保留，剩余容量按现有顺序保留物块，最多32项。200ms仅为延迟诊断，不丢弃刚完成的有效结果。

物块type0绿、1黑、2橙、3蓝，绿色/黑色/蓝色线上yaw保持0，橙色发送模型角度。己方默认蓝色，runtime.json的placement_zone_color可改purple。识别660×360mm外轮廓，推算内部物资/伤员各300×300mm子区中心，分别发送type4/5及朝区内法线yaw_cdeg。面对区内始终物资在左、伤员在右，颜色切换不交换语义。详见VISION_WIRE_CONTRACT.md。

## 启动、图像与抽样

相机与模型并行初始化；采集进程终止后自动重开相机，保留模型和串口。只有一个相机设备所有者。ROS为可选诊断输出：--ros-image-topic /camera/rgb/image_raw --ros-image-hz 5，有订阅者才序列化，后台只持有最新帧。sensor_msgs/Image、bgr8；时间戳为主机交接时间，不是曝光。RViz使用Image/raw，不是依赖CameraInfo的Camera显示。ROS master未上线不阻塞视觉/UART，本服务不另起ROS master。Melodic需可共存的python3-rospkg-modules/python3-catkin-pkg-modules，不安装会冲突的python3-rospkg或python3-catkin-pkg元包；包安装先模拟并使用--no-remove。

正式service已启用--sample-interval 5 --sample-limit 120：每5秒最多120组无损输入PNG、同帧完整输出和实际发送UART包，保存在/home/rikibot/vision_logs/current/samples/时间_PID。新进程开新目录，在 UART 完成后将帧与实际发送包配对，拷贝图像后交给单独的有界写入线程保存无损 PNG；写盘失败停抽样、不停视觉，关闭时等待至多一个排队样本及正在写入的样本完成。采满后视觉继续运行。`frames.jsonl` 中 `sample_queued` 是排队事件，`sample_writer_summary` 记录最终保存数和错误；准确的逐帧内容在样本目录同名 JSON 文件中。

python3 -m unittest discover -s tests -v

r23：94项Nano契约通过。保存的两绿一橙样本改前8次均只发2项，改后8次均发3项，绿色角度均为空。旧六图按独立标定回放，已检出但位置不精确的目标可输出估计；这些不证明毫米精度。实际相机12帧所有检测均形成发送目标，物理串口写完不等于STM32正确接收。当前办公桌背景有蓝黑误检，识别准确率未验收。没有在本版本重新断电验收。

## r24相机卡顿恢复
连续三次取帧失败时重建采集进程，恢复后序号重新开始；保留模型、主进程与串口。95项板载测试通过，真实受控卡顿11.57秒恢复新帧，UART TX增加108字节。不是硬件所有故障的恢复保证。

## r25 收尾验证
正式模型models/blocks_and_zones_astra_v2.engine。103项Jetson测试通过；99张Astra回归中86帧输出两子区、12帧方向歧义、1帧尺寸拒绝，PTY回环全部通过。107组线上字节经STM32真实Comm/Pose/World/Field主机联测通过。该批图像部分参与训练，回归覆盖不等于独立泛化或毫米精度。

2026-11-19：橙色type2的yaw_cdeg不再强制清零；发送当前模型角度，按180度周期归一化，无姿态估计时0。live_monitor网页区分模型姿态质量和实际线上值。

2026-11-19 近处/移动物块补检：语义类别green/black/orange/blue/zone/zone置信度阈值为0.25/0.50/0.12/0.25/0.35/0.35。57帧真实采集审查中，5帧橙色置信度0.13–0.17被旧阈值0.20过滤；20帧同帧整链PTY验证全部通过，补回5帧橙色位置与角度。低阈值可能增加误检，黑色阈值保持0.50以排除地面黑条和桌椅。该批为现场回归，未构成连续运动精度真值测试。

2026-11-19 r28：橙色物块可见颜色连通域补检。YOLO遗漏或边界裁切时，使用高饱和橙色、颜色比例、面积和填充度筛选；已有YOLO橙框覆盖则不重复增加。候选继续走现有已标定地面中心与角度拟合、正常UART。180帧现场图像抽查，20帧无YOLO橙框样本完整流程均发出橙色坐标和非零角度、PTY回环通过。局部遮挡的角度可能为best-effort，需看监视器质量标记。

2026-11-19 r29：蓝色物块YOLO门槛从0.25提高至0.65。现场无蓝块画面中固定顶边背景产生0.33–0.57置信度的蓝色误检；历史真蓝块样本有0.9以上置信度。其余类别阈值和颜色补检逻辑未改。

2026-11-19 r30：红色/粉色实体平台以六类YOLO外框投影地面求整个平台及两个子区；原HSV颜色区分割在当前Astra低饱和画面中仅2/238帧发送两区。按外框投影和红色阈值0.20，238帧真实采集离线回放174帧形成物资/伤员完整成对目标（73.1%）；默认己方颜色仍为blue，不向串口发送红色对方区。全局CLAHE和白平衡未采用，均未稳定提升检出。样本有大量相邻帧，比例为回归覆盖率，不代表独立泛化准确率。

2026-11-19 r31：红色实体平台阈值0.10，支持红色/紫色多块可见轮廓合并；仅当颜色轮廓缺失或尺寸不合理时回退模型外框。238帧Astra实际采集离线回放193帧得到物资/伤员成对子区，含原先失败的120、160帧。回退外框的朝向不含斜边证据，精度需另行标定；默认己方blue未改。

2026-11-19 r32：红色实体平台完整可见时，按红色地面相对周围地面的RGB色差并合并紫色边缘像素形成轮廓，投影后求两个子区中心与法线；颜色证据不足才使用轴对齐YOLO外框回退。238帧同场景红区回放198帧输出成对子区，其中175帧采用颜色轮廓、24帧采用外框回退。蓝色平台路径保持原算法。placement_zone_color默认blue，可改purple；两模式分别用真实图像20帧PTY发包回环验证，blue19对、purple17对。色彩轮廓给出的是可见范围，不代表外尺寸测量真值，仍需独立角点/尺寸标定。

2026-11-19 r33：扩大红区紫色边缘HSV范围到H100..179/S>=42/V35..205，恢复远距离画面中偏蓝的边缘。被指出的0帧原可见颜色轮廓右端缺失，新轮廓覆盖实体右边缘；238帧回放198帧成对子区。默认blue；蓝/红选择分别20帧PTY验证19/17对。轮廓贴合为视觉检查，毫米坐标仍缺独立实测真值。

2026-11-19 r34：红色平台颜色轮廓在检测框邻近图像边界一侧扩展搜索，左右镜像帧均能形成完整子区对，避免固定向右扩展的偏置；保留YOLO候选约束及几何门槛。238帧同次真实回放198帧成对发出（83.2%），开头静止60帧物资区Y标准差约23mm，较r32约69mm下降。默认己方blue，配置purple亦经真实帧PTY验证。完整精度仍受未独立验证的外参、畸变和实体加工偏差限制。

2026-11-19 r35：完整红色平台优先用内部粉色底面估计两子区中心，整体可见轮廓继续决定朝向；内侧底面尺寸偏离600×300mm超过25%时，仍用原整体轮廓回退。238帧Astra回放198帧成对输出；用户手工标的4帧8点，画面平均偏差由约22.1降到17.4像素；开头静止60帧物资区Y标准差由22.3降到1.9mm。四帧完整Jetson模型与UART编码回放4/4成对。以上手工点位为近似像素参考，不等于独立测量的毫米精度；默认己方颜色仍为blue。


## 2026-11-21 r36 性能与代码质量改进

当前基线 `rgb-unified-20261121-r36-fast-color-pipeline`：对后处理颜色空间转换、橙色稀疏颜色门槛、红区掩模与轮廓投影进行语义等价优化。六类 YOLO 模型、阈值、标定、UART 协议、抽样、ROS/监视功能不变。原生检测分段计时需要在 Jetson 上重编译检测库；旧 ARM64 库继续可用。基准数据、验证结论和板上复测命令见 `PERFORMANCE_REVIEW.md`。

## 2026-11-21 r37 视觉速度优化

r37 继续消除跨阶段重复 HSV 转换、空帧不必要的 native metric 初始化，并将 5 秒 PNG 抽样搬到后台（单个待写队列，无无限积压）。监控进程复用不可变的输入帧引用。C++ 同类 NMS 经过 50 组候选对比与原算法逐框一致，候选密集时局部收益明显；原生 `.so` 必须在 Nano 上编译后生效。所有检测、几何和串口阈值均未改变。详细数据见 `PERFORMANCE_REVIEW.md`。

## r38 本轮优化与复验（2026-11-21）

最新审查与基准见 `docs/vision_optimization_baseline.md`、`docs/vision_test_progress.md`、`docs/vision_performance_report.md` 和 `docs/vision_compatibility.md`。r38 不修改模型、UART 协议与比赛类别语义；新增应用层 P50/P95 统计均为有限窗口，不能替代板上曝光/物理串口测量。

```bash
python3 tests/verify_sources.py
python3 -m unittest discover -s tests -v
# 原生 C++ NMS（不需 CUDA、OpenCV 头文件）
g++ -std=c++14 -O3 -Iinclude src/detection_postprocessor.cpp tests/nms_equivalence.cpp -o /tmp/nms_equivalence
/tmp/nms_equivalence
```

与原 r37 的开发机对照脚本在 `tools/bench_wire_r38.py`、`tools/check_shape_center_r38.py` 和 `tools/bench_r38_pipeline.py`；须分别提供完整 r37 基线目录。正式服务保持原启动方式；测试时请避免同时启动两个争用 Astra/UART 的实例。

## r39 独立二次复审（2026-11-21）

本轮不是对上一轮结论背书：重新检查完整 `Astra → YOLO → HSV/绿色 → native metric → UART` 调用链，确认 r38 模拟回退测试不能充当 Jetson 实机吞吐率。r39 保持所有外部契约，减少 `Geometry.set_position` 针孔模型重复 `cv2.projectPoints`，复用未模糊 ROI 的已有 HSV，并为生产运行增加监控/JSON/样本提交耗时 P50/P95（无对应功能时为 null）。x86 局部基准和各目标数完整模拟对照见 `docs/vision_performance_report.md` 与 `docs/r39_x86_benchmark.json`；所有未知 Nano 指标标为未测。

```bash
python3 tests/verify_sources.py
python3 -m unittest discover -s tests -q
python3 tools/bench_r39_audit.py --baseline-root /path/to/r38 --iterations 50
# 没有 CUDA/OpenCV C++ 开发头文件的 PC 可独立构建并运行 NMS 检查：
g++ -std=c++14 -O3 -Iinclude src/detection_postprocessor.cpp tests/nms_equivalence.cpp -o /tmp/nms_equivalence
/tmp/nms_equivalence
```

正式服务命令及串口格式不变。请勿直接把 PC 的合成检测框成绩代入 Jetson Nano 的实际 FPS；本机不涉及系统刷机、模型更换和对 STM32 的写入。
