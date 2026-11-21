# Jetson—STM32 放置区通信合同（2026-11-17）

## 坐标与左右语义

坐标原点为车轴中点，X向前、Y向左；x_mm/y_mm为毫米int16小端。yaw_cdeg为车体X轴到“场内指向区内法线”的逆时针角，单位0.01度，归一化到[-18000,18000)。区域中心是对应子区几何中心的地面坐标。

两个放置区镜像布置。用户已确认：无论己方在哪侧、选哪种颜色，机器人从场内面对区内，始终物资在左、伤员在右。颜色只用于选己方，绝不交换两区。

外轮廓尺寸660×360mm；内部有效区域600×300mm，物资和伤员各300×300mm。Jetson从外轮廓求平台中心C及朝区内单位向量n。令左向量l=(-n.y,n.x)，物资中心=C+150mm·l，伤员中心=C−150mm·l；两者发送相同法线yaw。左右两区不用整个平台中心替代。

## 线上合同

AA55 | N:u8 | N×(type:u8,x_mm:i16LE,y_mm:i16LE,yaw_cdeg:i16LE) | CRC16LE。
CRC-16/CCITT-FALSE覆盖N及记录，poly=0x1021，init=0xFFFF，xorout=0，无反射。

- 0绿色、1黑色、2橙色、3蓝色物块：保持当前发送规则，绿色/黑色/蓝色线上yaw=0，橙色type2发送模型角度。
- 4/6=己方物资子区，5/7=己方伤员子区；Jetson实际生成4和5，STM32兼容6/7别名。
- 正常N=0..32，N=0是普通空目标帧。STM32另兼容N=FF故障帧；本版Jetson仍只生成正常帧，启动、预热、采集/处理异常和空闲状态沿用N=0，不重发旧目标。Jetson的decode只解正常目标帧，不把FF解成空目标。
- 一个平台产生一对4/5记录。两区优先占用2个容量位，其他物块按现有优先级填满剩余30位。两区中任一坐标越过int16范围时整对省略，避免只发送半对地标。

## 己方选择与实现

Jetson configs/runtime.json 的 placement_zone_color 默认为blue，改为purple可切换己方；修改后须重启视觉服务。仅选己方颜色；重复检测只取一处最高置信度且几何有效的平台，不发送对方平台。

YOLO框按外尺寸使用。蓝色轮廓采用HSV H=90..145、S>=85、V>=30；先3×3开运算去噪，再7×7闭运算连接内部细分隔线，选最大连通平台，最终凸包只取原有有效颜色像素。避免墙面/地面杂色把凸包撑成YOLO矩形。

外轮廓投影到地面后拟合矩形。按用户要求使用宽松尺寸范围：短边162..648mm、长边297..1188mm、长短比1.2..3.5。超出名义尺寸±25%但通过宽松门槛的结果标为relaxed_size_estimate。

入口取靠近相机的长边，依赖机器人位于场地内。两长边距离差<50mm时不发布区域位姿；没有可靠区域方向时不编造yaw。畸变、围边高度和外参误差仍会影响毫米估计。

## 与STM32场地定义一致

Field上侧物资(-150,1350,+90°)、伤员(150,1350,+90°)；下侧物资(150,-1350,-90°)、伤员(-150,-1350,-90°)，单位mm/deg。正负方向及镜像左右关系与上述公式一致。Pose按4/6与5/7映射到己方对应地标，接收侧不重复应用相机外参。

## 模型与验证

普通六类检测模型为models/blocks_and_zones_astra_v2.engine，TensorRT SHA256=39624a89792c1be22f7c2bcc847685c6b9bd83288c77a1ee42fc7e07ee1c303a。类别通道顺序orange/light_blue/black/green/pink_zone/blue_zone，模型通道转协议type映射仍[2,3,1,0,4,5]；模型颜色类别5不等于线上伤员type5，两者在pipeline中明确转换。

99帧同次Astra采集回归：99帧检出平台，86帧发送两区，12帧方向歧义，1帧尺寸拒绝；99帧PTY发送回环均通过。该批数据部分参与模型训练，属于回归覆盖率，不是独立泛化精度。整链计算均值约116ms，实时进程同时运行时测得，非曝光到接收端的延迟。

本机test/vision_contract_check.c及vision_contract_vectors.h验证107个真实Jetson生成包（99个回放+8个上下镜像/不同车头角度案例），经实际comm_nano.c→Pose→World→Field。逐字段核对type/坐标/yaw，并验证分包、CRC拒绝和地标投影。测试不操作硬件串口。

复现：python test/run_vision_contract_check.py --cc D:/DEV/Dev-Cpp/MinGW64/bin/gcc.exe；UB检查可改GCC11.2并加--ubtrap。Jetson运行python3 -m unittest discover -s tests。

## 文件与部署

Jetson根目录：/home/rikibot/vision_research_20261026/full_project_20261027/jetson_vision。
修改文件备份：/home/rikibot/zone_contract_candidate_20261117/before_install/。
运行进程的模型和配置身份以新进程输出的metadata.json及配对样本为准；磁盘文件替换不会自动改变旧进程。

2026-11-19：橙色type2的yaw_cdeg不再强制清零；发送当前模型角度，按180度周期归一化，无姿态估计时0。live_monitor网页区分模型姿态质量和实际线上值。
