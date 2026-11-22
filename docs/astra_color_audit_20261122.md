# Astra 偶发偏绿／偏灰：源码链路审查（2026-11-22）

## 结论与证据范围

对 `chat/vision-r41` 分支进行静态源码审查。**没有读到偏色发生当时的相机原始 PNG，因此未定位具体硬件或 SDK 根因。** 当前源码未在显示监控输出前对完整原始图施加灰度化、绿色滤镜、gamma、饱和度或自动亮度补偿。

| 步骤 | 代码位置 | 实际操作与数据所有权 |
|---|---|---|
| OpenNI 彩色流 | `src/astra_camera.cpp:59-73` | 选择 640×480、30 Hz、`PIXEL_FORMAT_RGB888`；禁用镜像。没有显式设置自动白平衡／自动曝光、手动增益、色温 |
| SDK → BGR | `src/astra_camera.cpp:80-110` | 按 stride 从 SDK RGB888 拷贝到独立输出缓冲，交换 R/B 通道；不做颜色增益 |
| 进程之间交付 | `python/rescuevision/runtime.py:75-90,125-136` | 带锁 `np.copyto`，消费者获取独立 `.copy()`；没有颜色转换 |
| YOLO 输入 | `src/detector_bridge.cpp:55-70` | 缩放／letterbox 后 `BGR2RGB`、`1/255` 归一化，只构建神经网络输入，不改原始帧 |
| HSV 补检 | `python/rescuevision/frontends.py:231-249` | `cv2.cvtColor(image, COLOR_BGR2HSV)` 新缓冲，仅做颜色掩膜和候选生成 |
| 原生几何颜色补偿 | `configs/runtime.json` 的 `metric_white_reference: true`；`src/metric_bridge.cpp:40-43`；`vendor_model/include/rescue_vision/ground_white_reference.hpp` | 创建独立 RGB 图、估算地面白参考的红／蓝增益，写入 `Session.balanced` 临时缓冲用于模型拟合。**不会回写原始 BGR 图或 YOLO 输入** |
| 本地预览和网页 | `python/rescuevision/drawing.py:6`；`python/rescuevision/live_monitor.py:68-93` | 前者在新图副本上标记；后者直接将原始 BGR 编码 JPEG，叠加标记由网页 Canvas 绘制 |
| ROS 图像 | `python/rescuevision/ros_image.py:41-48` | 直接发送 BGR，消息编码标记 `bgr8` |

## 能解释和不能解释的现象

- 若网页实时画面**整体**发绿或发灰，几何的地面白参考校正不是直接原因，因为修改的是独立拟合缓冲。
- 橙／蓝／绿检测掩膜可能受实际偏色影响，造成识别变化，但其正常处理不会造成原始摄像头颜色改变。
- Astra 流创建时并未明确固定或记录自动白平衡、自动曝光状态。**设备或驱动的自动调节**是待验证假设，不是已确定根因。现场照明光谱/闪烁、供电或采集流异常也需结合原始帧区分。
- `BGR2RGB` 和 `RGB888→BGR` 方向在代码中彼此一致；静态审查没有发现重复交换通道或未按 stride 读取的问题。

## 最小化实机诊断（不改变算法）

1. 同一光照、同一景物、尽量固定位置，采集颜色正常与偏色时各数张**原始 PNG**，不要用手机对显示屏拍照代替原始图。
2. 对比相机刚打开与预热数秒后的画面，并保存序列号、采集时间；观察是否存在颜色跳变。
3. 对白色地面背景区域统计 RGB 三通道中位值、HSV 饱和度和灰度亮度，并对比正常／异常帧；在分析前不要做自动色彩增强。
4. 如原始 PNG 已偏色，下一步再查看 SDK 摄像头控制属性是否支持读取/固定自动白平衡与曝光；如果原始 PNG 正常而网页偏色，则审查浏览器显示和 JPEG 解码路径。

工程已提供可选原始帧采样：`app.py --sample-interval <秒> --sample-limit <张数>` 经 `python/rescuevision/sampling.py` 保存原始 PNG 和同帧结果。采样会消耗存储空间；**当前 Nano 根分区仅剩约 1.8 GiB，不应无上限采样。** 摄像头由单一进程独占，诊断前确认生产服务状态；无需也不应通过任务代理同时打开第二个相机会话。

本审查未修改 Astra 驱动、视觉算法、色彩阈值、模型权重或 UART。
