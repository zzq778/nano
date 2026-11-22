# r41 Jetson Nano 实机验证进展（2026-11-22）

## 环境与任务回执

通过私有 `zzq778/jetson` 的 GitHub 代理读取实机回执，不是本机模拟。

- 设备：`rikibot`、`aarch64`、Ubuntu 18.04.6 / Linux 4.9.337-tegra、Python 3.6.9。
- 代理：`bridge_version=2`，开始检查时 `~/nano` HEAD `a8435df`。
- `df -h /home/rikibot`：59G 总量、55G 已使用、**1.8G 可用，97% 使用率**。
- `free -m`：物理内存 3963 MB，可用约 1419 MB；交换空间约 10 GB。
- `nvcc` 当前 PATH 未检测到；`g++`、`cmake`、`git`、`python3` 存在。

## 首次 r41 同步为什么失败

任务 `r41_20261122_01_sync` 返回 `Tracked changes exist in development workspace; refusing checkout`。独立工作目录 `~/.local/share/jetson-chat-dev` 刚执行 `git clone --no-checkout`，真实工作树只有 `.git`，但 Git 的 `status` 会将全部已跟踪源码显示为 `D`，旧代理错误地当作用户修改而拒绝检出。

后续旧代理继续执行的 `compileall` 返回码 0 但报告 `Can't list 'python' / 'tests' / 'tools'`，**不能记为通过**。`pytest` 报 `tests` 不存在，原生编译和 NMS 脚本也不存在；这些失败是空检出目录造成的，**不构成 r41 算法失败或 C++ 构建结论**。

## 修复与安全处理

- `tools/dev_agent.py` 新逻辑在确认 `origin` 正确后，若工作区**仅包含 `.git`**，允许完成首次检出；否则仍拒绝覆盖已修改的跟踪文件。
- 始终要求 `chat/*` 分支和 GitHub 完整 40 位提交哈希匹配后才检出。
- `run_check` 对仅包含 `.git` 的空工作区明确拒绝，防止 `compileall` 假通过。
- 新增真实 `main` HEAD 指向的 `git clone --no-checkout` 集成回归及空目录检测测试；本机复现和验证了首次检出逻辑。**新增完整测试仍需 Nano 运行**。
- 修复分别提交到 `main` 和 `chat/vision-r41`；部署需要 Nano `~/nano` 拉取后重启桥接进程，不会自动热更新。
- 磁盘剩余 1.8G，**暂缓可能大量生成构建缓存的 TensorRT/CMake 原生构建**，先完成成功检出、Python 测试和 NMS 小规模编译。

## Astra 颜色链路

详见 `docs/astra_color_audit_20261122.md`。当前代码未修改预览整帧色相或饱和度；`metric_white_reference` 的 R/B 增益仅作用于独立原生几何临时缓冲。偶发整帧发绿／发灰原因仍待采集当时原始 PNG 排查。

## 尚需实机验证

1. Nano 执行 `cd ~/nano && bash scripts/upgrade_bridge.sh`，确保新代理测试通过并替换后台进程。
2. 重新下发**新的任务 ID**：同步当时最新 `chat/vision-r41` 提交哈希、核对工作区 HEAD，再执行测试；之前已失败任务不会自动重跑。
3. 根据新的设备空间余量决定是否独立 ARM64 构建；不覆盖生产 `.so`、模型、标定和 UART。
