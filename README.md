# 街景语义分割预训练模型推理工具

这是一个基于 **OpenMMLab / MMSegmentation** 的街景语义分割推理项目。项目使用 Cityscapes 预训练模型，对输入街景图片进行批量语义分割，并生成类别 mask、彩色可视化、overlay 图和类别占比报告。

当前版本只做推理：

- 不训练模型
- 不微调模型
- 不需要人工标注 mask
- 不计算 mIoU 或准确率
- 推理阶段固定使用本地 `config.py` 和 `.pth checkpoint`

转载需标注来源。

## 功能

- 显式下载 MMSegmentation 预训练模型到 `checkpoints/`
- 使用本地 config + checkpoint 批量推理
- 保存 Cityscapes 19 类原始预测 mask
- 将 Cityscapes 类别重映射为项目 9 类
- 生成项目类别彩色 mask
- 生成原图叠加 overlay 图
- 统计每张图的类别像素占比

## 项目结构

```text
street_scene_pretrained_seg/
  input_images/          # 放原始街景图片，GitHub 中默认只保留目录
  checkpoints/           # 本地模型文件，GitHub 中默认只保留目录
  outputs/               # 推理输出，GitHub 中默认只保留目录
  configs/
    class_mapping.yaml   # Cityscapes 到项目类别的映射配置
  docs/
    class_definition.md
    model_download.md
    usage.md
  scripts/
    00_download_pretrained_model.py
    infer_cityscapes_local.py
    remap_to_project_classes.py
    visualize_project_masks.py
    make_class_statistics.py
    run_pipeline.py
  requirements_runtime.txt
  README_GITHUB.md
```

## 环境准备

建议使用已配置好的 OpenMMLab 环境。核心依赖包括：

```text
PyTorch
MMCV
MMEngine
MMSegmentation
OpenMIM
Pillow
NumPy
PyYAML
```

依赖清单见：

```text
requirements_runtime.txt
```

OpenMMLab 相关包与 CUDA、PyTorch、MMCV 的版本需要匹配。建议优先参考 OpenMMLab 官方安装方式配置环境。

## 快速开始

进入项目目录：

```bash
cd street_scene_pretrained_seg
```

下载默认 Cityscapes 预训练模型：

```bash
python scripts/00_download_pretrained_model.py
```

默认模型为：

```text
pspnet_r50-d8_4xb2-40k_cityscapes-512x1024
```

下载成功后会生成：

```text
checkpoints/active_model_info.json
checkpoints/*.py
checkpoints/*.pth
```

将待处理街景图片放入：

```text
input_images/
```

支持：

```text
.jpg
.jpeg
.png
```

运行完整 pipeline：

```bash
python scripts/run_pipeline.py
```

指定设备：

```bash
python scripts/run_pipeline.py --device cuda:0
```

CPU 推理：

```bash
python scripts/run_pipeline.py --device cpu
```

## 输出结果

运行完成后，结果会写入 `outputs/`：

```text
outputs/cityscapes_pred/     # Cityscapes 19 类单通道预测 mask
outputs/cityscapes_vis/      # Cityscapes 原始可视化结果
outputs/project_pred/        # 项目 9 类单通道 mask
outputs/project_vis/         # 项目 9 类彩色 mask
outputs/project_overlay/     # 原图 + 半透明分割结果
outputs/reports/             # 类别统计和失败文件记录
```

统计报告包括：

```text
outputs/reports/class_statistics.csv
outputs/reports/class_statistics_summary.md
outputs/reports/failed_files.md
```

## 项目类别

项目输出类别固定为：

```text
0 sky
1 road
2 lane_marking
3 vehicle
4 vegetation
5 building
6 pole
7 sidewalk
8 person
255 ignore
```

Cityscapes 到项目类别的映射见：

```text
configs/class_mapping.yaml
```

## 关于车道线

`lane_marking` 保留为项目类别 ID `2`，但 Cityscapes 预训练语义分割模型通常没有独立的车道线类别。当前路线下，车道线往往会被预测为 `road`，因此 `lane_marking` 通常不会稳定出现。

如果后续必须单独识别车道线，需要考虑：

- 使用专门的车道线检测或分割模型
- 准备少量车道线 mask 做微调
- 在道路区域上做颜色、边缘或几何后处理

## 不建议提交到 GitHub 的内容

以下内容通常较大或属于本地数据，已通过 `.gitignore` 排除：

```text
checkpoints/*.pth
input_images/*
outputs/*
__pycache__/
*.pyc
```

也就是说，GitHub 仓库保存的是代码、配置和文档；模型权重、输入图片和推理输出由使用者在本地生成。

## 常用命令

查看完整 pipeline 参数：

```bash
python scripts/run_pipeline.py --help
```

只做本地 checkpoint 推理：

```bash
python scripts/infer_cityscapes_local.py
```

只做类别重映射：

```bash
python scripts/remap_to_project_classes.py
```

只生成彩色 mask 和 overlay：

```bash
python scripts/visualize_project_masks.py
```

只生成类别统计：

```bash
python scripts/make_class_statistics.py
```

## 说明

本项目是一个可复现的预训练模型推理工具，不是训练项目。没有人工 ground truth mask 时，不能严谨计算 mIoU、Pixel Accuracy 等指标。本项目提供的是预测结果、可视化结果和类别占比统计。
