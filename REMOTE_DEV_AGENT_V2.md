# Jetson Nano Chat 模式远程开发代理 v2

**目的**：让手机 ChatGPT Chat 模式通过 GitHub 完成多数代码开发回合：查看代码 → 修改 GitHub 开发分支 → Nano 同步 → 编译/测试 → 回读日志 → 继续修改。它是异步 GitHub 桥接，**不是直接 SSH，也不等于完整 Codex/Work 计算机操作**。

## 实际工作方式

- **公开源码仓库**：`zzq778/nano`，只放源代码、编译脚本和测试；模型、现场数据和设备密码不公开。
- **私有任务仓库**：`zzq778/jetson`，用 `requests/<id>.json` 和 `results/<id>.json` 传递任务/结果。
- **Jetson 常驻桥接**：`~/nano/tools/remote_bridge.py loop`，每约 30 秒轮询待执行任务。
- **独立开发目录**：`~/.local/share/jetson-chat-dev`。首次同步时克隆公开仓库，随后切到指定 `chat/xxx` 分支的远端提交（detached HEAD）。该目录可以重新构建，不代表生产目录。
- 原有四个动作 `health`、`nms_smoke`、`python_tests`、`native_build` 继续兼容。v2 另外支持如下能力。

| 任务 `action` | 参数 | 功能 |
|---|---|---|
| `sync_code` | `branch`（如 `chat/vision-r40`）、`expected_sha`（经审查的完整 40 位提交哈希） | 拉取并切换到指定开发分支；若开发区有已修改的跟踪文件则拒绝覆盖 |
| `repo_status` | 无 | 返回开发区 HEAD、Git 状态及来源 |
| `list_code` | 可选 `prefix` | 列出开发仓库已跟踪源码文件（最多返回 150 条） |
| `read_code` | `path`、可选 `start_line`、`max_lines` | 查看源代码/文档片段，拒绝 `.env`、`.git`、上级目录和符号链接 |
| `search_code` | `query` | 在已跟踪的源码中搜索文本，最多返回 60 个匹配 |
| `run_check` | `check`、仅 pytest 可用 `test_file` | 在独立开发目录进行固定类型测试/编译，含超时和输出大小限制 |
| `device_info` | 无 | 读取 `free`、`df`、`uptime` 与开发工具存在情况 |

`run_check.check` 支持 `pytest`、`nms_smoke`、`compileall`、`native_build`、`cmake_configure`、`cmake_build`、`diff_check`。`test_file` 只接受 `tests/test_*.py`。所有执行均使用参数数组而非任意 shell 文本；不包含 sudo、固件操作、生产动态库覆盖或硬件执行。

**安全边界说明**：这并非容器或 Linux 权限隔离。运行 GitHub 分支里的 Python 测试或构建脚本，本质上仍会执行该仓库代码，并拥有 Linux 用户 `rikibot` 的文件权限。只能同步自己信任的分支；正式生产部署和系统级配置需要独立授权。令牌仍只保存在 Jetson 上，GitHub 仓库权限应仅允许私有 `jetson` 的 Contents 读写。

## 例子：由 Chat 模式下发任务

先在 `zzq778/nano` 中创建并修改 `chat/vision-r40` 分支。然后由 ChatGPT GitHub 连接在私有 `jetson` 仓库创建：

```json
{
  "id": "sync_r40_001",
  "action": "sync_code",
  "branch": "chat/vision-r40",
  "expected_sha": "<这里填开发分支真实的40位提交SHA>"
}
```

在 `results/sync_r40_001.json` 检查 `ok: true`、`head` 及 `branch`，再创建：

```json
{
  "id": "test_r40_001",
  "action": "run_check",
  "check": "pytest"
}
```

ChatGPT 可读取结果、修复公开开发分支、用新的唯一 ID 再次 `sync_code`、重跑测试。每次由正在进行的 Chat 对话推进；Chat 本身不会无限期自主思考或持续运转。周期性任务另需自动化调度。

## Nano 上一次性升级（不需要重新生成 Token）

当前 `~/nano` 的 Python3/pytest 已配置好时，先在 Termux SSH 中执行：

```bash
cd ~/nano
 git pull --ff-only
python3 -m pytest -q tests/test_remote_bridge.py tests/test_dev_agent.py
```

然后重启现有后台代理，使 v2 代码生效：

```bash
pgrep -f '^python3 -u tools/remote_bridge.py loop$' | xargs -r kill
nohup python3 -u tools/remote_bridge.py loop > "$HOME/jetson-bridge.log" 2>&1 < /dev/null &
sleep 2
tail -n 12 "$HOME/jetson-bridge.log"
```

若想开机自启，先预览再由你明确选择安装：

```bash
bash scripts/install_bridge_service.sh --preview
bash scripts/install_bridge_service.sh --install
```

`--install` 需要本机 sudo 确认，把 unit 写入 `/etc/systemd/system/`，**服务以普通 `rikibot` 用户执行**，有重启策略；不会安装系统 Python/CUDA，也不操作生产程序。只选择一种常驻方式（`nohup` 或 `systemd`），避免双实例锁冲突。停止服务：`sudo systemctl disable --now jetson-chat-bridge.service`。

## 关键边界

1. GitHub 通道不是实时终端；单次任务可能有轮询、网络及构建延迟。
2. 远端编译的成功不代表模型/相机/UART 已经完成实机联合验证。
3. 私有模型、传感器现场数据只保存在 Nano/私有存储，公开仓库不能部署完整工程。
4. 新代理只操作独立开发检出；正式部署、修改服务、写串口/刷机必须由用户另行批准和使用独立工具。
5. 如果需要跨多轮自动持续迭代，请另外配置定时/条件任务；Chat 对话不会在用户离开时自发继续执行。