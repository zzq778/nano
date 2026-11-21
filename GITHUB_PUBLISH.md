# r39 GitHub 发布说明

这份目录是安全优先的**源码分发版**，不是完整部署包。

为避免把训练权重和现场图像上传到可能公开的仓库，这里刻意未包含：
`models/`（engine/ONNX）、`data/`（真实相机图像/测试数据）、`build/`（AArch64 动态库）以及原有 `.git` 历史。

完整模型、真实样本和已在 Nano 上使用的动态库，继续以用户自己持有的 `jetson_vision_r39_deploy.tar.gz` 为准。仅源码发布版无法直接启动检测，也无法执行所有依赖数据/模型/原生库的测试；这**不是**源码功能失败。

## 创建目标仓库并上传

目标仓库 `zzq778/nano` 已建立；本源码包只包含公开发布适宜的源码和文档，不含训练权重、现场图片及设备原生编译产物。

如果在手机 Termux 中自己上传，确认仓库属于本人且访问已配置：

```sh
pkg install git openssh
cd path/to/this/source-folder
git init -b main
git add .
git diff --cached --check
git commit -m 'Import Jetson Nano vision r39 source and mobile build docs'
git remote add origin https://github.com/zzq778/nano.git
# 仅当远端为空仓库时：
git push -u origin main
```

如果仓库已经有文件或提交，**不要直接强推**，而是先 clone 该仓库，建功能分支，再导入此源码并通过 PR 合并。HTTPS 认证请使用 GitHub 账号授权的凭据或凭据管理器，**不要把 token 写进源码、脚本或聊天消息**。
