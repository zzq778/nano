# 从 Android 手机访问 Jetson Nano / ARM64 构建（r39）

## 首选：Termux SSH → Nano 本机构建

1. 从 F-Droid 或 Termux 官方 GitHub 发布页安装 Termux，不要混装不同签名源的 Termux 插件。
2. 在 Termux 中执行：

```sh
pkg update
pkg install openssh git
ssh user@JETSON_LAN_IP
```

首次登录确认服务端主机指纹（应与已知 Jetson 主机指纹核对），输入 Jetson 账户密码。此时编译**实际发生在 Nano**，完全匹配原有 Ubuntu 18.04 ARM64 的 CUDA/TensorRT/OpenCV 链接环境。

在已获得工程源码的 Nano 上，进入工程目录：

```sh
uname -m                     # 预期 aarch64
cmake --version
ls /usr/local/cuda
python3 tests/verify_sources.py
bash scripts/phone_nms_smoke.sh
bash scripts/build_native_on_jetson.sh
```

`build_native_on_jetson.sh` 在 `build_phone_verify/` 单独构建，并不会覆盖生产 `build/*.so`。需要 Astra SDK 时，先设 `ASTRA_SDK_ROOT` 指向 SDK 的根目录。若 CMake 找不到 TensorRT、OpenCV 4 或 CUDA，确认 Nano 的开发头文件与原工程使用的库是否已安装；**不要随意升级 JetPack**。

终端断线不会代表已完成编译。请在稳定连接后运行，上面脚本未启用后台自动运行。测试通过后才能单独部署动态库。

## 手机本机 ARM64 Linux 编译（CPU-only）

Android 的 Termux 原生 clang 面向 Android Bionic，不是 Jetson 的 Ubuntu glibc ABI；直接构建的 Android `.so` **不能**代替 Jetson 的 `.so`。

可以通过 Ubuntu ARM64 用户态运行只依赖标准 C++ 的 NMS 验证：

```sh
pkg install proot-distro
proot-distro install ubuntu:18.04
proot-distro login ubuntu
apt update
apt install -y build-essential git file
# 将源码文件夹带入 Ubuntu 用户态后，cd 到工程根目录
bash scripts/phone_nms_smoke.sh
```

Ubuntu 18.04 的软件源可能过期；若安装失败可选择当前 Ubuntu 镜像做 CPU 练习，但其较新的 glibc 不能保证能部署到 Nano。此流程不提供 Jetson GPU 驱动、CUDA 或 TensorRT，**不适合完整检测库**。`proot` 也不等于另一套内核，编译性能受手机散热/内存限制。

## 不在同一个 Wi-Fi / 局域网

手机与 Nano 若不在同一个局域网，`192.168.x.x` 私有地址通常不可达。推荐在两端建立 Tailscale 网络，再从 Termux 连接 Nano 的 Tailscale IP：

```sh
ssh user@TAILSCALE_IP
```

仅启用网络互通不足以 SSH：Nano 必须已有 SSH 服务，且你的账户有登录权限。不建议把 TCP/22 直接暴露给公网。

## 检查 SSH 服务（在 Nano 的本地终端执行）

```sh
sudo systemctl status ssh
sudo systemctl enable --now ssh
hostname -I
```

Ubuntu 默认 `openssh-server` 若缺失，先在 Nano 用现有安装源安装；不要远程盲改 firewall 或认证规则，以免失去现有登录方式。
