# Chat 开发代理 v2 测试与验证记录（2026-11-21）

## 范围与基线

源代码：`zzq778/nano`；任务队列：私有 `zzq778/jetson`。该升级**只修改远程代理和测试/文档**，没有修改模型、视觉阈值、UART 协议、运行主链，也没有自动部署到 Nano 生产程序。

## 开发机测试

- 本机环境：Linux x86_64、Python 3.13.5、pytest 9.0.2。用本地 Git 仓库模拟开发分支同步，不需要连接实际 Nano。
- 代理专项：`python -m pytest -q tests/test_remote_bridge.py tests/test_dev_agent.py` → **22 passed**（另有 12 个子用例通过）。
- 与完整 r39 安全源码合并：`python3 -m pytest -q tests` → **141 passed、12 skipped、0 failed**。12 跳过涉及未随公开源码发布的私有图像/几何样本及需要设备原生库的验证条件。
- 运行 `ast.parse(..., feature_version=(3,6))` 解析代理和测试的 Python 语法：通过。`bash -n` 检查服务脚本和升级脚本：通过。
- 实测本地 Git 分支同步、切到固定提交、拒绝有未提交修改时的覆盖、拒绝路径穿越、拒绝符号链接、限制读取扩展名、过滤完成任务避免旧的前十名卡死。

## 对权限的界定

- `run_check` 不能接受任意命令；只接受约定的单测、编译、NMS、diff 检查。
- 分支必须是 `chat/xxx` 且必须提供经过核对的完整 SHA-1，防止分支在测试前被替换。
- 开发工作区与原生产工程分离；读取源码限定为开发工作区里的一般源码/文档格式。
- **不宣称 Linux 沙箱安全**：执行分支代码具有 Jetson 登录用户可读写文件的权限。不要测试来源不可信的 Git 分支；严格保持 Token 私有。
- 新版未在 Nano 上完成新动作联调，需设备侧升级代理后再验证 `sync_code` 和 `run_check` 的实际执行结果。

## 回退

若代理升级后出现问题：`git -C ~/nano log -3 --oneline` 找到上一个工作版本，并将旧版本 `tools/remote_bridge.py`、`tools/dev_agent.py` 与相关文件在单独分支恢复；不要通过强制清空工作目录处理。私有任务与历史结果不会被清除。