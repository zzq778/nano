# r39 手机 ARM64/SSH 与 GitHub 发布准备核验记录

日期：2026-11-21。环境：x86_64 Linux，非 Jetson，不在用户设备的局域网。

## 已执行

- 读取并确认 r39 完整源码、`CMakeLists.txt`、UART 契约、测试工具及服务启动配置。
- GitHub 连接身份核实为 `zzq778`；`zzq778/jetson` 的仓库元信息与目录读取均返回 404，连接可见仓库列表中没有目标仓库。**未向任何其他仓库上传**。
- 用 `g++ -std=c++14 -O2` 在 x86 上构建 C++ NMS 测试：50 组旧新实现对照通过；这是开发机 CPU 验证，不是 Nano ARM64 运行验证。
- 发布版源码以下测试运行通过：`test_contracts.py`（28项）、`test_r38_wire_latency.py`（4项）、`test_r39_pinhole_and_topquad.py`（3项）、`test_r39_runtime_overheads.py`（2项）。合计 37 项通过。
- 手机 SSH / Nano 构建 / CPU-only NMS 三份 Bash 脚本通过 `bash -n` 静态检查。
- 使用 `clang -target aarch64-linux-gnu` 编译单独的纯 C ABI 目标，`readelf` 显示 AArch64 ELF；**未链接 Qt/OpenCV/TensorRT，也没有编译主工程的 ARM64 `.so`**。
- 从隔离容器尝试访问曾用的 Jetson 局域网 TCP/22，连接被拒绝。该结果不能判定 Nano 关机或 SSH 未启用，因为此环境不在同一局域网；本容器也没有 SSH 客户端可执行程序。
- 去除了 GitHub 源码分发版里的模型权重、现场图像、原生动态库和原 Git 历史。完整部署版本另外保留全部本地资源，供私有传输。

## 待用户设备验证

- Android 手机上 Termux 的 SSH 登录；目标网络必须可达、认证由用户本人完成。
- Nano 上使用对应 SDK/工具链的 CMake 构建、ABI 检查、服务启停与实际 UART 联调。
- 远端仓库在 GitHub 应用授权后实际推送。

## 限制

GitHub 源码快照不包含模型、测试图像或 ARM64 编译库，不能单独执行完整视觉推理。完整工程与该源码快照分别交付；所有生产算法和 UART 编码实现均保持 r39 内容不变。
