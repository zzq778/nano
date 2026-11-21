# 新相机的测距、测角与物理中心

目标点始终是物块三维几何中心沿Z投影到地面的点。输出x/y相对轮轴中点，X前/Y左。日志同时给出水平距离、该地面点到轮轴中点的三维直线距离，以及橙/绿物块的地面姿态角。像素框中心、颜色质心、黑块顶点都是观测证据，不直接当作此物理目标点。

绿色/蓝色为40mm正方体，橙色80×40×40mm，黑色正四面体各棱40mm、高32.660mm。方体顶面中心的XY等于其地面中心；正四面体顶点的垂直投影等于底面重心。黑色使用可见顶点的几何证据估计中心，蓝色使用已知立方体尺寸和框投影约束估计中心；明显碎片只允许远处输出带不确定度的粗中心。黑蓝不输出朝向。所有中心算法仍需新相机标定和实物误差验收。

原六幅640×480照片已由用户确认属于旧Astra及对应安装，可以使用所给实测K与外参作明确标记的针孔估算，畸变仍未确认。`astra_six_estimate.json`仅按六图RGB SHA绑定，用于离线回放，不允许套入替换后的USB实时相机。

当前为原Astra 2bc5:0402，使用配套SDK直接取流。用户指定K为fx493.3689、fy494.2357、cx327.3957、cy234.8487；当前光心(+27,0,+129)mm，roll0、yaw0、地面z=-36mm。新帧近1/2号参考拟合给出俯角31.6°工程估计，独立位置精度与畸变仍未验证。旧六图仍使用其历史外参，不与当前安装混用。

新增multi_block外部相机图片可按用户授权使用虚拟Astra作压力测试，配置为`multi_block_assumed_astra.json`，明确记录分辨率缩放假设。它不提供实测距离/角度真值，不计入下述误差验收，也不允许打开物理UART。

## 1. 采集与求内参

打印同目录board_A4_18mm.pdf，选实际大小100%，用尺确认100mm标尺及实际格长。棋盘含9×6内角点。固定相机安装、分辨率和焦距；手持棋盘改变远近/倾角/画面位置，保留至少12个清晰且有倾角变化的视图。以下capture命令会明确打开所指定相机，本轮六图实验没有运行它。

```sh
python3 tools/calibrate_current_camera.py capture --device astra --output calibration_views --count 20
python3 tools/calibrate_current_camera.py intrinsics --images calibration_views --square-mm 18 --base-config configs/astra_current.json --output configs/astra_measured.json
```

方格若实测并非18mm，应把--square-mm改成实测值。工具记录设备VID/PID、数据来源和重投影误差，拒绝无棋盘、视角不足、倾角变化不足或超限RMS，不写猜测内参。没有可靠设备绑定时不会自动宣布匹配当前相机。

## 2. 独立地面测距/测角

另拍一张未参与标定的地面棋盘图。量出A0（纸上左上第一个内角点）相对轮轴中点的地面x/y和棋盘+X相对车头的角度。棋盘+X沿列递增、+Y沿行递减，确认OpenCV角点顺序；如检测顺序反向则加--reverse。下面三个MEASURED参数必须由实物测量替换。

```sh
python3 tools/calibrate_current_camera.py validate-ground --image measured_board.png --calibration configs/astra_measured.json --square-mm 18 --origin-x-mm MEASURED_X --origin-y-mm MEASURED_Y --board-yaw-deg MEASURED_ANGLE --corner-order-confirmed --output ground_validation.json
```

输出每个角点的XY误差、中心误差、距离和方向角误差。它验证相机地面投影，不能替代物块边缘/中心提取的整链验收。工具拒绝把标定训练图重新当独立测量真值。

## 3. 真实物块整链验证

在若干已知位置/角度放置标准物块，记录其地面中心而非近边或顶点。量出中心相对轮轴的x/y；橙色量80mm长边方向，绿色按90°对称比较。黑蓝只标中心。将独立测量写为JSON的measurements数组，各项包含image_id、candidate_index、type、x_mm、y_mm，橙绿另含yaw_deg，并在source说明尺具和测量方法。物块匹配用结果图的候选编号，仅用于评分，不输入检测器。

```sh
python3 tools/score_block_pose.py --frames output/measured_scene/frames.jsonl --measurements physical_measurements.json --output block_pose_errors.json
```

用户验收目标为：近场中心误差≤10mm，远场≤20mm，橙色角度误差≤10°，绿色≤15°。无有效米制观测或独立实物真值时，误差为null、测量数为0，不伪造零误差。原六图可作绑定标定的估算和检测/计数回归；新标定资料与实物测量资料按来源单独冻结，不能用重投影效果代替实物真值。

## 4. 新UART

230400/8N1，AA55|N|type+x_int16_mm+y_int16_mm+yaw_int16|CRC16。线上x/y必须来自上述物理地面中心。当前协议要求全部yaw=0；真实测算的角度保存在yaw_world_deg，不把保留的0当成测角结果。以后启用线上角度须两端同步验证；帧格式本身不变。

当前Astra畸变未知，工程保留原因并输出故障包。工具产生的新测量配置不会自动替换用户指定K或当前运行配置；使用前仍需独立验证。实测D输出明确为未去畸变图像模型，不继承忽略D的针孔估算策略。地面验证工具可使用这份D；当前native_known_size_rgb后端只接收针孔或已矫正的图像坐标，不能直接把新测出的未去畸变配置切入该后端，也不能仅修改rectification_status来冒充矫正。实际去畸变链需在有测量数据后单独验证。

虚拟Astra压力结果也保留明确来源。本轮仅要求四类物块，安置区后续接入，编码器合法类型仍为0..5。PTY回环可验证字节和CRC，尚未替代STM32实收和实物测距/测角验收。
