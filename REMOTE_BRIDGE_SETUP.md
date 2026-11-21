# 手机 Chat 模式 → Jetson Nano：受限 GitHub 远程桥接

本方案**不是直接 SSH**：Jetson 自己以 HTTPS 主动查询私人仓库 `zzq778/jetson` 的 `requests/*.json`，只执行预先批准的四种操作，并写入 `results/*.json`。ChatGPT 通过连接的 GitHub 应用创建任务、读取结果。Nano 无需开放 SSH 到公网。

## 你需要做一次的操作（在手机 Termux 已 SSH 登录的 Nano 终端）

### 1. 创建仅限一个仓库的令牌

在 GitHub 浏览器打开 [Fine-grained personal access tokens](https://github.com/settings/personal-access-tokens/new)：

- **Resource owner**：`zzq778`
- **Repository access**：`Only select repositories` → 仅选择私有 `zzq778/jetson`（不是公开 `nano`）
- **Repository permissions** → `Contents`: `Read and write`（Metadata Read 通常自动附带）
- 其他权限不要额外开启。设置适当到期时间；到期后重新配置。

**不要把 Token 发到 ChatGPT、GitHub issue 或公开仓库里。**

### 2. Nano 上下载代理并输入令牌

在已登录 Jetson 的 SSH 会话执行：

```bash
cd ~
git clone https://github.com/zzq778/nano.git
cd nano
python3 tools/remote_bridge.py setup
```

如果 `~/nano` 之前已经存在，别删：执行 `cd ~/nano && git pull --ff-only` 即可。`setup` 会在终端无回显地输入 GitHub Token；它只存到 Nano 本地 `~/.config/jetson-remote-bridge/token`，文件权限 `0600`。无需 sudo、pip 安装或系统升级。

### 3. 先执行一次健康检查

```bash
python3 tools/remote_bridge.py once
```

首次运行显示 `Completed ... new job(s)` 即说明能访问任务队列；有待处理健康检查任务时，会写入私有仓库 `results/`。告诉 ChatGPT“代理已运行一次”即可，不要发 Token。

### 4. 如需持续保持 Chat 模式可用，再运行后台轮询

```bash
nohup python3 -u tools/remote_bridge.py loop > "$HOME/jetson-bridge.log" 2>&1 < /dev/null &
```

每 30 秒查询一次。有任务才执行；出错会打印到 Nano 本地日志，**没有自动任意命令执行能力**。查看状态：

```bash
pgrep -af 'remote_bridge.py loop'
tail -n 30 ~/jetson-bridge.log
```

停止时，可在已登录的 Nano 会话用 `pgrep -af` 找到进程号再 `kill <PID>`；无需改 SSH 配置，也没有设置开机自启动。

## 可接受的任务

| action | Nano 上实际执行 | 是否需要 CUDA/TensorRT |
|---|---|---|
| `health` | 架构、Python 版本、g++/CMake/nvcc 可用性、源码版本 | 否 |
| `nms_smoke` | `bash scripts/phone_nms_smoke.sh`，在临时目录构建 CPU NMS 测试 | 否 |
| `python_tests` | `python3 -m pytest -q tests`，运行测试而不启动机器人 | 否（原生测试除外） |
| `native_build` | `bash scripts/build_native_on_jetson.sh`，在独立 `build_phone_verify` 目录编译 | 是 |

无法由任务 JSON 自行指定 `bash` 命令、执行路径、环境变量或 sudo。运行日志最多返回 16 KiB，结果只写入私有仓库。代理默认单实例，最长执行时间有界。它**不会**刷新硬件、改变串口协议、覆盖生产 `.so` 或上传模型。

## 常见问题

- `GitHub HTTP 401` / `403`：检查 Token 的过期时间、仅选 `jetson` 的仓库权限及 Contents Read/Write。
- `GitHub HTTP 404`：通常是 Token 无权读私有仓库或仓库名错误。
- `No module named pytest`：不影响 `health` / `nms_smoke`；测试所需依赖需在 Nano 的现有环境中安装，不要盲目升级 JetPack。
- `Token file is not private`：执行 `chmod 600 ~/.config/jetson-remote-bridge/token`。
- `Agent already running`：不要启动第二个实例。使用已运行的后台代理。
- 有连接，但没有任务：可继续正常操作；请求队列还没有新的 JSON 文件。

**安全注意：** 仓库权限或 Token 被盗会允许他人提交上述受限任务。请用短期令牌，及时撤销不再使用的 Token，不要在未经审查的第三方电脑上保留它。结果仓库也属于私有资料，避免向公开仓库转存日志。