# 模型显式下载说明

本项目采用“显式下载 config + checkpoint 到本地，然后固定使用本地文件推理”的方式。

## 为什么要显式下载模型

MMSegmentation 可以在某些推理入口中根据模型名自动下载权重，但自动下载通常会把文件放到缓存目录。缓存目录不一定直观，也不方便把实验交付给别人复现。

本项目把预训练模型文件明确下载到 `checkpoints/`，后续推理脚本只读取本地文件：

```text
checkpoints/*.py
checkpoints/*.pth
checkpoints/active_model_info.json
```

这样可以清楚知道当前实验使用的是哪一个模型配置和哪一个权重文件。

## config 文件是什么

`config` 文件通常是一个 `.py` 文件，来自 MMSegmentation 的模型配置。它记录模型结构、数据预处理、推理流程、类别信息等内容。

在本项目中，推理阶段会读取本地 config 文件，而不是只依赖模型名称。

## checkpoint 文件是什么

`checkpoint` 文件通常是一个 `.pth` 文件，保存已经训练好的模型参数。这里使用的是 MMSegmentation 在 Cityscapes 数据集上训练好的预训练权重。

本项目不训练模型，只使用这个 `.pth` 文件做街景图片推理。

## 为什么更适合复现

显式下载后，项目会生成 `checkpoints/active_model_info.json`，记录：

```text
model_config_name
config_file
checkpoint_file
download_time
download_command
```

之后的推理脚本可以固定读取这份 JSON 中记录的本地文件。只要 `checkpoints/` 中的 config 和 checkpoint 没有被删除，后续推理一般不需要再次联网下载模型。

## 下载命令

在项目根目录运行：

```bash
python scripts/00_download_pretrained_model.py
```

默认模型为：

```text
pspnet_r50-d8_4xb2-40k_cityscapes-512x1024
```

也可以手动指定模型名和输出目录：

```bash
python scripts/00_download_pretrained_model.py --model-config-name pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest checkpoints
```

如果提示找不到 `mim` 命令，需要先在当前 OpenMMLab 环境中安装或激活 OpenMIM。
