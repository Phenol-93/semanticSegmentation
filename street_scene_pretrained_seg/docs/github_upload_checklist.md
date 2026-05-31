# GitHub 上传检查清单

上传前建议只提交代码、配置和文档，不提交本地数据、模型权重和推理结果。

## 建议提交

```text
README_GITHUB.md
.gitignore
requirements_runtime.txt
configs/class_mapping.yaml
docs/
scripts/
input_images/.gitkeep
checkpoints/.gitkeep
outputs/**/.gitkeep
```

## 不建议提交

```text
checkpoints/*.pth
checkpoints/active_model_info.json
input_images/*.jpg
input_images/*.jpeg
input_images/*.png
outputs/**/*.png
outputs/**/*.csv
outputs/**/*.md
scripts/__pycache__/
```

## 上传前检查命令

查看 Git 将会跟踪哪些文件：

```bash
git status
```

查看是否还有大文件会被提交：

```bash
git status --short
```

如果已经误添加了模型权重或输出结果，可以取消暂存：

```bash
git restore --staged checkpoints outputs input_images
```

然后重新添加需要提交的代码和文档：

```bash
git add README_GITHUB.md .gitignore requirements_runtime.txt configs docs scripts
git add input_images/.gitkeep checkpoints/.gitkeep outputs/.gitkeep
git add outputs/cityscapes_vis/.gitkeep outputs/cityscapes_pred/.gitkeep
git add outputs/project_pred/.gitkeep outputs/project_vis/.gitkeep outputs/project_overlay/.gitkeep outputs/reports/.gitkeep
```

## README 处理建议

如果这个目录要作为独立 GitHub 仓库，建议上传前将：

```text
README_GITHUB.md
```

复制或重命名为：

```text
README.md
```

当前本地 `README.md` 可以保留给自己使用，里面可以写本机路径和本地环境细节；GitHub 版 README 应避免出现本机绝对路径。
