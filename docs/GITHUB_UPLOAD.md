# 上传到 GitHub

本项目已改为通过 Git 上传完整原目录，包括代码、训练数据、模型检查点和运行结果。目标仓库：<https://github.com/15755633090-ai/coarse_graining>。

`github_upload` 精简副本已按要求删除，不需要重新生成。`.gitignore` 只排除 Python 缓存、虚拟环境、凭据文件和重复上传副本；数据集与输出目录不再被排除。

## 后续更新完整项目

在项目根目录执行：

```powershell
git status
git add .
git commit -m "Update coarse-graining project"
git push origin main
```

GitHub 普通 Git 上传单文件上限为 100 MiB，目前本项目最大单文件约 18.94 MiB，不需要 Git LFS。规则见 [GitHub 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

`scripts/maintenance/prepare_github_upload.py` 是此前保留的可选精简副本工具，完整 Git 上传不需要运行它。如需使用，在项目根目录执行 `python -m scripts.maintenance.prepare_github_upload`。项目运行与检查命令见 [根目录 README](../README.md)。
