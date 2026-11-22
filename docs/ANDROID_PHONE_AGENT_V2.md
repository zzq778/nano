# 手机 ChatGPT → Termux → Android：v2 实机部署

## 已实现

使用 GitHub 私有仓库 `zzq778/jetson` 的 `phone_requests/` → `phone_results/` 作为出站轮询队列，每 30 秒检查。没有开放任何手机入站端口。与 Jetson `requests/` 和 `results/` 完全分离。

- 基础：`ping`、`device_info`、工作区文件列表/读取/原子写入。
- 开发：显式 `--enable-dev` 后运行白名单程序（`python`、`git`、编译工具等）。**不是沙盒**：Python 参数允许执行代码，因此等于授权对 Termux 权限范围内的任意操作。
- Android：显式 `--enable-adb`（已有无线 ADB 配对）或 `--enable-root`（设备已经有 `su`）后，可执行 `android_status`、`android_tap`、`android_swipe`、`android_keyevent`、`android_input_text`（ASCII）、`android_launch_app`、`android_open_url`、`android_ui_dump`、`android_screenshot`、`android_batch`。
- 截图通过 Pillow 转成缩小的 JPEG 放到私有结果仓库中；这个仓库会持久保留截图，**不要在密码、验证码、银行等敏感界面执行截图**。
- `android_batch` 最多 12 个步骤；每步可设 `after_ms`。失败则停止，不是通用网页浏览器或能理解 UI 的本地 AI。
- Root 选项**不会 Root 手机**、不会解锁或刷机。只是检测并使用已经安装的 `su` 做受限 UI 操作。

## 1. 切回手机本地 Termux

如果提示符是 `rikibot@rikibot:...$`，说明当前在 Jetson Nano 的 SSH shell，需要输入 `exit` 退出 SSH。然后检查：

```sh
echo "$PREFIX"
```

正常 Termux 应输出类似 `/data/data/com.termux/files/usr`，而不是空值。若仍在 Jetson 上，再运行一次 `exit`。

## 2. 安装

无需下载 ZIP。手机本地 Termux 输入：

```sh
pkg update -y
pkg install -y python curl android-tools python-pillow
mkdir -p ~/chat-mobile-agent && cd ~/chat-mobile-agent
curl -fL --retry 3 -o phone_agent.py 'https://raw.githubusercontent.com/zzq778/nano/feature/mobile-agent-v2/tools/phone_agent.py'
python -m py_compile phone_agent.py
python phone_agent.py setup
python phone_agent.py once
```

若 `python-pillow` 未可用，可先安装其他三个包，截图需要 Pillow 后再单独补装。`setup` 将要求 GitHub fine-grained PAT：仅授权私有仓库 `zzq778/jetson`，**Contents: Read and write**。Token 仅在 Termux 的隐藏输入框中输入，永不发给 ChatGPT，文件模式 `0600`。

已经预置 `phone_requests/phone_bootstrap_20261122.json` 的 `ping` 请求。`once` 成功后，会出现 `phone_bootstrap_20261122 ok`，ChatGPT 可检查私有仓库里的结果。

## 3. 获得 Android 界面控制（二选一）

### A. 已 Root 的手机

在手机本地 Termux 运行 `su -c id`。如果成功且显示 `uid=0(root)`，可通过已安装的 `su` 获得界面控制，无须无线 ADB 配对：

```sh
cd ~/chat-mobile-agent
termux-wake-lock
python phone_agent.py loop --enable-dev --enable-root
```

如果弹出 Root 授权，用户必须在本机手动批准。**不要在没有备份的情况下解锁 Bootloader 或刷机**。

### B. 未 Root 的手机：无线 ADB

在手机「设置 → 开发者选项 → 无线调试」打开功能，选择「使用配对码配对设备」。保持设置界面可见，通过手机分屏或通知栏切回 Termux：

```sh
adb pair <配对地址IP>:<配对端口>
# 按提示输入六位配对码，切勿把配对码发到聊天
adb connect <无线调试主页显示的IP>:<连接端口>
adb devices -l
```

注意：**配对端口和连接端口不同**。Android 11+ 支持无线调试，但通常需要手机连接 Wi-Fi；若手机只在开热点、未连接 Wi-Fi，可能无法开启。此时可以临时用 Wi-Fi，或改用已 Root 模式；不能绕过 Android 的授权机制。

然后运行：

```sh
cd ~/chat-mobile-agent
termux-wake-lock
python phone_agent.py loop --enable-dev --enable-adb
```

## 4. 后台运行（可选）

确定稳定后可将上述 `loop` 用 `nohup ... > ~/chat-mobile-agent/agent.log 2>&1 &` 后台运行。到系统设置将 Termux 的电池优化设为「无限制」，仍不能保证 Android 不会杀死进程。需要开机自启时另装官方 Termux:Boot，并单独配置。

**安全范围**：私有 GitHub 队列写入权限=手机代理下发权限，尤其是 `--enable-dev`；其他人有仓库写权限即可能操控你的 Termux。无需 Root 的 ADB 也属于高权限，远程执行动作前要关注屏幕内容与数据安全。提供一键停止方式：在 Termux 执行 Ctrl+C，或找到代理 PID 后停止进程；撤销 GitHub PAT 可永久终止队列认证。

## 5. 功能边界

完成代码和离线测试不等于已经连接真机。当前尚未验证这部手机的 PAT 访问、Android 16 无线调试、Root 情况、完整 UI 层可见性和截图压缩速度。即使 UI 控制可用，部分 App 的安全窗口不允许截屏、无障碍读取或 ADB 注入。用户授权并不意味着系统能越过这些硬件/系统限制。
