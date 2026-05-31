# 类别定义与重映射说明

本项目当前阶段只使用 MMSegmentation 的 Cityscapes 预训练模型进行推理，不训练模型、不微调模型，也不需要人工标注 mask。

## 原始 Cityscapes 输出是什么

Cityscapes 预训练语义分割模型通常输出 19 个城市街景类别。推理得到的原始预测 mask 是单通道类别索引图，每个像素值表示一个 Cityscapes 类别 ID：

```text
0 road
1 sidewalk
2 building
3 wall
4 fence
5 pole
6 traffic light
7 traffic sign
8 vegetation
9 terrain
10 sky
11 person
12 rider
13 car
14 truck
15 bus
16 train
17 motorcycle
18 bicycle
```

这份原始输出会保存在 `outputs/cityscapes_pred/`，用于保留模型最初的预测结果，方便后续检查、重新映射或对比不同模型。

## 本项目输出是什么

本项目将 Cityscapes 19 类预测结果统一重映射为 9 个项目类别，并保留 `255 ignore` 作为忽略区域：

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

重映射后的项目 mask 会保存在 `outputs/project_pred/`。它仍然是单通道类别索引图，像素值只应包含 `0-8` 或 `255`。

## 为什么要做类别重映射

Cityscapes 的 19 类比本项目需要的类别更细，也有一些类别在本项目第一版中可以合并。例如 `car`、`truck`、`bus`、`train`、`motorcycle`、`bicycle` 都统一归入 `vehicle`；`building`、`wall`、`fence` 统一归入 `building`。

这样做有三个目的：

1. 让输出类别更贴合当前街景分析任务。
2. 降低第一版结果解释和统计的复杂度。
3. 保留 Cityscapes 原始预测，同时生成项目自己的标准输出格式。

具体映射关系写在 `configs/class_mapping.yaml` 中，后续脚本应直接读取该配置，避免在代码里重复硬编码。

## lane_marking 的限制

`lane_marking` 在本项目中固定保留为类别 ID `2`，但 Cityscapes 预训练语义分割模型通常没有独立的车道线类别。因此，仅靠 Cityscapes 预训练模型推理时，车道线大概率会被预测为 `road`，重映射后通常不会稳定出现 `lane_marking`。

如果后续必须单独识别车道线，需要考虑额外方案，例如：

1. 使用专门的车道线检测或车道线分割模型。
2. 准备少量人工标注的车道线 mask 做微调。
3. 在 `road` 区域内做颜色、边缘或几何后处理。

这些都属于后续改进方向，不属于当前“不训练、只推理”的阶段。

## 为什么只做推理时不需要人工 mask

当前流程使用的是已经训练好的 Cityscapes 预训练模型。推理阶段只需要输入原始街景图片，模型会直接输出预测 mask。

人工标注 mask 主要用于训练、微调或计算 mIoU、Pixel Accuracy 等定量指标。由于本阶段不训练、不微调，也不计算需要 ground truth 的指标，所以不需要用户准备人工标注 mask。

本阶段可以做的是：

1. 生成预测 mask。
2. 生成彩色可视化和 overlay 图。
3. 统计预测结果中的类别占比。
4. 做定性质量检查。
