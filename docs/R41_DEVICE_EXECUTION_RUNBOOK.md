# Jetson Nano · r41 独立实机验证执行顺序

版本日期：2026-11-22。此文档是未来操作手册，**截至编写时未执行 r41 Jetson 新轮验证，更未部署生产视觉服务**。

## 0. 前提

- Nano 通过热点能够访问 `api.github.com` 和 `github.com`；Termux 能 SSH 到用户自己的 Nano。
- `~/nano` 是 bridge 启动目录；远程 GitHub PAT 仅限私有任务队列仓库，保存在 Nano 本地，不应显示或上传。
- 源码和性能测试来自 `zzq778/nano` 的独立分支 `chat/r41-preflight`；`main` 只同步代理固定任务与自身回归测试，不合并视觉 r41 实验代码。
- 任何操作遇到 `git` 本地修改、代理测试失败、SHA 不匹配、磁盘不足，**停止并保留现场，不执行清理或强制重置**。

## 1. 唯一必要的首次用户操作

仅在用户决定启动设备测试后，在 Termux SSH 登录 Nano 并执行：

```bash
cd ~/nano
bash scripts/upgrade_bridge.sh
```

升级脚本先使用 HTTP/1.1 快进拉取 `main`，运行 `test_remote_bridge.py` 和 `test_dev_agent.py`，测试通过才重启普通用户权限的代理。没有 `sudo`、没有清理目录，也不会替换生产 `.so`。如果检测到 systemd 服务，会提示用户单独重启，而不是偷偷修改系统服务。**GitHub 仓库提交成功不等于这一步已成功。**

## 2. Chat 模式可执行的受限自动任务

后续使用每次新的任务 ID，经 GitHub 私有队列下发：

1. `health`：核对代理实际源版本、架构及工具链。
2. `sync_code`：指定 `branch=chat/r41-preflight` 和本次核对的完整 40 位提交 SHA，只同步到 `~/.local/share/jetson-chat-dev`；拒绝意外源文件覆盖。
3. `repo_status`：确认独立工作区实际 HEAD 与预期一致且已跟踪文件没有本地修改。
4. `run_check` / `r41_readiness`：只读检查 RAM、磁盘、ARM64 和原生编译工具。磁盘少于 4 GiB 或可用 RAM 少于 750 MiB 时暂停后续重型构建。
5. `run_check` / `r41_preflight_quick`：隔离目录的 Python 回归、NMS、哈希锁定 r40/r41 C++ 80 场景 A/B；缺少私人实拍帧时显式 `SKIP`，不推算整机 FPS。
6. 在以上工作无异常且系统资源足够时，`run_check` / `native_build`：固定在 `build_chat_verify/` 构建 ARM64 动态库；检查是否构建成功和原有 ABI，不能替换生产 `build/*.so`。
7. 复核状态与日志，形成“通过、失败、跳过”的逐项报告。不把失败任务的退出码 0、`Can't list` 或空测试目录视为通过。

## 3. 压测数据和采集限制

- C++ A/B 仅比较纯几何内核；TensorRT 推理要单独记录耗时/输入传输，不能与离线 Python 参考路径相加推导真实 FPS。
- 同一功耗模式、散热、模型、输入、光照和热机状态下，统计 0/1/15/32 目标及 300+ 帧的 P50/P95/P99，至少 3 轮；持续压力至少 10 分钟、600 帧。
- 用既有运行 JSONL/summary 和 tegrastats 文本分别交给 `r41_runtime_report.py`、`r41_tegrastats_report.py`；它们是**日志解析器，不自动采集**。
- 偶发偏绿/偏灰的诊断必须取得正常和异常的真实原始 PNG。静态源码检查只证明显示链路未主动对整帧调色，不能认定自动白平衡就是根因。
- **任何真实 Astra 独占采集、接入 `/dev/tty*`、替换生产动态库、重启生产视觉服务或切换功耗模式**，在之后单独确认；当前测试代理不具有这些操作权限。

## 4. 故障与安全边界

- `git pull` 网络错误：保留原目录和 Token，不重克隆、不 `reset --hard`；检查网络后重试相同非破坏性命令。
- 代理升级测试失败：升级脚本在启动新代理前退出，保留旧进程和现场；提供失败日志用于修正。
- Git 工作区脏或 SHA 不匹配：停止，不尝试绕过 SHA 检查或强制覆盖源码。
- 结果仅有 `PASS`、`FAIL`、`SKIP` 的实际区分；未运行硬件测试的项目标记 `NOT_RUN`，不得推断已达到目标。

本阶段的完成定义是**执行能力、测试工具和可回退边界准备完毕**，不是承诺软件绝无缺陷。
