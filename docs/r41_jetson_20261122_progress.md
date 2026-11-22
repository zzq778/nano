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

## 2026-11-22 上午追加实机记录

- Jetson 上代理 v2 通过健康检查；提交记录显示测试时 `~/nano` 仍在 `104d90f`。r41 检出依旧失败，所以后续 `compileall` 的“Can't list ...”不能算成功。
- 根分区的**最新回执**为 59G 总量、32G 已用、25G 可用、56% 使用率（比之前 97% 使用率明显改善），物理内存可用约 1334 MB。没有自动执行删除磁盘文件的操作。
- 本机复现：`git clone --no-checkout` 之后 `git ls-files` 可为空，而 `git ls-tree -r --name-only HEAD` 可列出源文件。未检出工作树与额外 `.pytest_cache` 共存时，旧代理错误地视为用户变更。修复改为检查 `HEAD` 已跟踪文件是否真实存在，并且**仅当全部不存在**时恢复初始文件，再安全切换到指定哈希；部分已有源码仍拒绝覆盖。
- 独立 Git 模拟环境已验证：可从带测试缓存的空检出目录恢复到目标分支，未改变测试缓存；其他情况下仍保留拒绝覆盖保护。此项新修复仍待 Nano 执行升级后的专项测试。
- 更新后的代理主分支修复提交 `b32c49c`，开发分支含同一修复。由于常驻进程不会热更新，需要 Nano 在 `~/nano` 运行 `bash scripts/upgrade_bridge.sh`。r41 源码与 ARM64 测试需在这次升级成功后以**新任务 ID**重跑。

## 2026-11-22 后续：开发代理兼容性修复

- Nano 的代理升级测试截图确认三项失败，均在代理自身测试：测试 mock 未提供完整执行结果；测试仓库没有可解析的 HEAD；旧同步逻辑过早依赖 HEAD。**r41 算法和 ARM64 编译尚未执行，不属于算法回归失败。**
- 调整同步顺序：先获取并确认 chat/* 分支的精确 40 位提交哈希，再根据目标树与可用的旧 HEAD 树确认文件是否已经存在；仅在完全无已跟踪源码文件落盘时使用强制首次检出。未出生 HEAD、旧索引、测试缓存均不会阻塞安全初始化；本地源码已存在时禁止强制覆盖。
- 对缺失目录会假成功的 compileall 增加实际源码关键文件检查；修复 mock 和测试缓存断言。增加无效 HEAD、错误提交哈希与未跟踪目录冲突测试。
- 本地用真实临时 Git 仓库执行 10 个初始化、重复同步、脏工作区、哈希核验与冲突场景测试全部通过；附加缓存保护验证通过。**尚未宣称 Nano 上全部测试通过**，需在 Nano 从 GitHub 拉取最新主分支后重新运行代理测试。
- 代理正式升级使用 `cd ~/nano && bash scripts/upgrade_bridge.sh`，测试全部通过时自动替换旧常驻代理；若任何测试失败则保留旧代理。
