# r41 性能测试准备最终报告 · 2026-11-22

本报告仅包含**本机可重复执行的测试结论**。截至本提交，没有通过新代理运行 Jetson r41 ARM64 构建、TensorRT 实际推理、摄像头独占测试或串口联调。正式生产服务没有被修改。

## 源码与测试结果

| 验收内容 | 实测 |
|---|---|
| 原始 r40 与 r41 完整树、full tier | 6 PASS、0 FAIL、0 SKIP |
| 完整树 Python 回归（含新增预检单测） | 140 passed、7 skipped、45 subtests passed |
| 不含私人图像/模型/动态库的公开源码副本，quick tier | 4 PASS、0 FAIL、2 SKIP |
| 不依赖外部 r40 工作树的哈希锁定 C++ A/B | 80 合成场景输出一致，P50 0.8420 → 0.6542 ms（本机耗时下降 22.3%） |
| r40 C++ 参考源码 SHA-256 | `de74ef99b87f4b2e081d7c3bc9fff809a48ce81dd840a8f544ec57eb94c8f430` |
| Python 3.6 | 新脚本经语法兼容检查；Nano 上运行时兼容**未验证** |

注意：r41 原生拟合的 22.3% 为独立 x86 主机测量结果，**不能作为 Jetson 整机 FPS 提升**。完整树和公开树间跳过项不同，不应相加或将 SKIP 视为通过。

## 已落地的安全边界

- `main` 仅更新远程开发代理及对应回归测试，未合入 r41 视觉生产逻辑。
- 新动作 `r41_readiness` 和 `r41_preflight_quick` 为**固定命令**；不接受提交任意 Shell 或额外命令参数。
- 开发工作区和固定构建输出目录拒绝符号链接；ARM64 原生编译输出严格位于隔离检出目录的 `build_chat_verify/`，不再读取 `BUILD_DIR` 环境变量，最多双并行。
- 代码同步核对完整 Git SHA，修复空工作区、无效 HEAD、暂存删除与缓存共存的初始化情况；存在真实本地源码冲突时不强制覆盖。
- `upgrade_bridge.sh` 采用单次 HTTP/1.1 快进拉取，测试全部通过才替换普通用户代理；不执行 `sudo`、`reset --hard` 或自动清理设备文件。
- 性能日志统计单独记录 `frame_error`，失败帧不会进入正常计算耗时分布，避免异常快帧伪造优化。

## 仍需设备操作的验收

见 [设备执行手册](R41_DEVICE_EXECUTION_RUNBOOK.md)：第一步在手机 Termux 登录 Nano 后运行一次 `cd ~/nano && bash scripts/upgrade_bridge.sh`。实际升级结果需由 Nano 测试确认，而不是由 GitHub 文件存在来推断。

随后 Chat 模式按固定提交 SHA 依次执行在线健康检查、独立检出、只读资源预检、Python/NMS/C++ 快速验收和隔离 ARM64 构建。真实摄像头、TensorRT 全链路运行、温度记录和生产替换需另外评估与授权。
