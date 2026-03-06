# SF-YOLO for YOLOv26 (Ultralytics) 使用文档

本文档介绍如何使用 SF-YOLO（无源域自适应）方法对 YOLOv26 模型进行域自适应训练。本实现基于 Ultralytics 框架，支持双域验证和教师-学生模型同时保存。

## 概述

SF-YOLO 是一种面向目标检测的域自适应方法，能够在**无需目标域标签**的情况下，将源域训练好的模型自适应到目标域。本实现专门针对 YOLOv26 模型，并提供了以下增强功能：

- ✅ **双域同时验证**：同时监控目标域和源域的性能
- ✅ **双模型保存**：分别保存学生模型和教师模型
- ✅ **域差距监控**：实时跟踪域迁移过程中的性能变化

## 前置条件

1. **源域预训练模型**：在源域（如 Cityscapes）上训练好的 YOLOv26 模型
2. **目标域增强模块 (TAM)**：预训练的风格迁移模型，用于域适应
3. **目标域数据集**：无标签的目标域图像（如 Foggy Cityscapes）
4. **源域数据集**（可选）：用于监控域迁移过程中的灾难性遗忘

## 环境配置

```bash
# 使用 uv 安装依赖
uv sync
```

## 训练流程

### 第一步：训练目标域增强模块 (TAM)

```bash
cd TargetAugment_train

# 提取目标域训练数据
python extract_data.py \
    --scenario_name city2foggy \
    --images_folder ../datasets/cityscape_foggy_yolo/images \
    --image_suffix png

# 训练 TAM 模块
python train.py \
    --scenario_name city2foggy \
    --content_dir data/city2foggy \
    --style_dir data/meanfoggy \
    --vgg pre_trained/vgg16_ori.pth \
    --save_dir models/city2foggy \
    --n_threads=8 \
    --device 0
```

### 第二步：SF-YOLO 域自适应训练

使用 `train_sf-yolo-ultralytics.py` 脚本进行域自适应训练：

```bash
uv run train_sf-yolo-ultralytics.py \
    --weights source_weights/yolov26l_cityscapes.pt \
    --data datasets/cityscape_foggy_yolo/cityscapes.yaml \
    --source_data datasets/cityscape_yolo/cityscapes.yaml \
    --epochs 60 \
    --batch 8 \
    --imgsz 960 \
    --device 0 \
    --decoder_path TargetAugment_train/models/city2foggy/decoder_iter_160000.pth \
    --encoder_path TargetAugment_train/pre_trained/vgg16_ori.pth \
    --fc1 TargetAugment_train/models/city2foggy/fc1_iter_160000.pth \
    --fc2 TargetAugment_train/models/city2foggy/fc2_iter_160000.pth \
    --style_path TargetAugment_train/data/meanfoggy/meanfoggy.jpg \
    --style_add_alpha 0.4 \
    --SSM_alpha 0.5 \
    --conf_thres 0.4 \
    --iou_thres 0.3 \
    --val_source \
    --project runs/sf-yolo \
    --name city2foggy_yolo26
```

**关键参数说明**：
- `--source_data`：源域数据集配置（用于监控域迁移性能）
- `--val_source`：启用源域验证
- `--SSM_alpha`：稳定学生动量系数（0.5 表示每轮将教师模型权重的 50% 转移到学生模型）

## 命令行参数详解

### 模型与数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | `yolo26n.pt` | 源域预训练模型路径 |
| `--data` | 必填 | 目标域数据集 YAML 配置文件 |
| `--source_data` | None | 源域数据集 YAML 配置文件（用于监控） |
| `--val_source` | True | 是否在源域验证集上进行验证 |
| `--epochs` | 60 | 训练总轮数 |
| `--imgsz` | 960 | 训练和验证的图像尺寸 |
| `--batch` | 16 | 批次大小 |
| `--device` | 0 | CUDA 设备 ID |
| `--workers` | 8 | 数据加载工作进程数 |
| `--project` | `sf-yolo` | 项目保存目录 |
| `--name` | `exp` | 实验名称 |

### SF-YOLO 特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--teacher_alpha` | 0.999 | 教师模型 EMA 衰减率（越接近 1，历史权重占比越高） |
| `--conf_thres` | 0.4 | 伪标签置信度阈值 |
| `--iou_thres` | 0.3 | NMS IoU 阈值 |
| `--max_det` | 20 | 每张图像最大检测数量 |
| `--SSM_alpha` | 0.0 | 稳定学生动量系数（0 表示禁用，建议 0.5） |

### TAM 模块参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `--decoder_path` | 是 | TAM 解码器权重路径 |
| `--encoder_path` | 是 | VGG 编码器权重路径 |
| `--fc1` | 是 | FC1 全连接层权重路径 |
| `--fc2` | 是 | FC2 全连接层权重路径 |
| `--style_path` | 否 | 风格图像路径（空字符串表示使用随机风格） |
| `--style_add_alpha` | 1.0 | 风格迁移强度（0-1 之间） |
| `--save_style_samples` | False | 是否保存风格增强样本图像 |

### 调试参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--debug_mode` | False | 启用调试模式，保存伪标签可视化 |
| `--debug_interval` | 50 | 每隔 N 个 batch 保存一次调试图像 |
| `--debug_save_dir` | `./check` | 调试图像保存目录 |

## Python API 使用方式

你也可以直接在 Python 代码中使用 `SFYOLOTrainer` 类：

```python
from train_sf_yolo_ultralytics import SFYOLOTrainer

# 配置参数
overrides = {
    # SF-YOLO 特有参数
    'teacher_alpha': 0.999,
    'conf_thres': 0.4,
    'iou_thres': 0.3,
    'max_det': 20,
    'SSM_alpha': 0.5,
    
    # TAM 模块参数
    'decoder_path': 'TargetAugment_train/models/city2foggy/decoder_iter_160000.pth',
    'encoder_path': 'TargetAugment_train/pre_trained/vgg16_ori.pth',
    'fc1': 'TargetAugment_train/models/city2foggy/fc1_iter_160000.pth',
    'fc2': 'TargetAugment_train/models/city2foggy/fc2_iter_160000.pth',
    'style_path': 'TargetAugment_train/data/meanfoggy/meanfoggy.jpg',
    'style_add_alpha': 0.4,
    
    # 源域验证参数
    'source_data': 'datasets/cityscape_yolo/cityscapes.yaml',
    'val_source': True,
    
    # 标准训练参数
    'model': 'source_weights/yolov26l_cityscapes.pt',
    'data': 'datasets/cityscape_foggy_yolo/cityscapes.yaml',
    'epochs': 60,
    'imgsz': 960,
    'batch': 8,
    'device': 0,
    'project': 'runs/sf-yolo',
    'name': 'city2foggy_yolo26',
}

# 创建训练器并开始训练
trainer = SFYOLOTrainer(overrides=overrides)
trainer.train()
```

## 核心特性

### 1. 双模型架构

- **学生模型 (Student)**：在风格化后的目标域图像上训练
- **教师模型 (Teacher)**：通过 EMA 机制更新，用于生成伪标签

教师模型不参与梯度更新，其权重通过以下公式更新：
```
θ_teacher = α * θ_teacher + (1-α) * θ_student
```
其中 `α` 为 `--teacher_alpha` 参数（默认 0.999）。

### 2. 目标域增强模块 (TAM)

- 基于 AdaIN 的风格迁移网络
- 将源域图像风格迁移到目标域风格
- 生成风格化图像用于学生模型训练

### 3. 伪标签生成

- 教师模型在原始图像上生成伪标签
- 使用 NMS 过滤，支持配置置信度和 IoU 阈值
- 伪标签用于监督学生模型的训练

### 4. 稳定学生动量 (SSM)

- 每轮训练开始时，将教师模型权重按一定比例转移到学生模型
- 有助于稳定训练过程，防止学生模型偏离过远
- 通过 `--SSM_alpha` 参数控制（0.5 表示 50% 权重转移）

### 5. 双域验证与监控

当启用 `--source_data` 和 `--val_source` 时，训练过程中会：

1. **目标域验证**：评估模型在目标域上的性能
2. **源域验证**：评估模型在源域上的性能（监控灾难性遗忘）
3. **域差距计算**：自动计算目标域与源域的性能差距

训练日志示例：
```
SF-YOLO: 目标域验证 fitness=0.4460, 源域 fitness=0.6270, 域差距=-0.1810
```

## 输出目录结构

训练结果保存在 `runs/sf-yolo/<实验名>/` 目录下：

```
runs/sf-yolo/
└── city2foggy_yolo26/
    ├── weights/
    │   ├── best.pt                 # 最佳学生模型
    │   ├── last.pt                 # 最新学生模型
    │   ├── best_teacher.pt         # 最佳教师模型
    │   ├── last_teacher.pt         # 最新教师模型
    │   └── epoch{N}.pt             # 定期保存的检查点
    │   └── epoch{N}_teacher.pt     # 定期保存的教师模型检查点
    ├── args.yaml                   # 训练参数配置
    ├── results.csv                 # 训练指标（包含源域和目标域）
    ├── train_batch*.jpg            # 训练批次可视化
    ├── enhance_style_samples/      # 风格增强样本（如启用）
    └── ...
```

### results.csv 字段说明

| 字段名 | 说明 |
|--------|------|
| `epoch` | 训练轮数 |
| `time` | 训练时间（秒） |
| `fitness` | 目标域 fitness 分数 |
| `source_fitness` | 源域 fitness 分数 |
| `domain_gap` | 域差距（目标域 - 源域） |
| `metrics/precision(B)` | 目标域精确率 |
| `metrics/recall(B)` | 目标域召回率 |
| `metrics/mAP50(B)` | 目标域 mAP@50 |
| `source_metrics/precision(B)` | 源域精确率 |
| `source_metrics/recall(B)` | 源域召回率 |
| `source_metrics/mAP50(B)` | 源域 mAP@50 |

## 故障排除

### CUDA 显存不足

```bash
# 减小批次大小
--batch 4

# 减小图像尺寸
--imgsz 640

# 减少工作进程数
--workers 4
```

### 域适应效果不佳

- 增加训练轮数：`--epochs 100`
- 调整风格强度：`--style_add_alpha 0.5`
- 降低伪标签阈值：`--conf_thres 0.3`
- 启用 SSM：`--SSM_alpha 0.5`
- 调整教师 EMA：`--teacher_alpha 0.995`

### 源域性能严重下降（灾难性遗忘）

- 启用 SSM 机制：`--SSM_alpha 0.5`
- 增大教师 EMA 衰减率：`--teacher_alpha 0.9995`
- 减少训练轮数：`--epochs 40`
- 降低学习率：`--lr0 0.005`

### TAM 模块加载错误

- 检查所有 TAM 权重路径是否正确
- 确认 VGG 编码器权重与模型兼容
- 检查 CUDA 是否可用

### 验证时出现文件路径错误

- 确认 `--source_data` 指向的 YAML 文件存在
- 检查 YAML 文件中的数据集路径是否正确
- 确保源域数据集的验证集图像和标签存在

## 模型选择建议

训练完成后，你可以根据需求选择不同的模型：

| 模型文件 | 适用场景 |
|----------|----------|
| `best.pt` | 目标域性能最优的学生模型 |
| `best_teacher.pt` | 目标域性能最优的教师模型（通常更稳定） |
| `last.pt` | 最后轮次的学生模型 |
| `last_teacher.pt` | 最后轮次的教师模型 |

**建议**：
- 如果关注目标域性能，优先使用 `best_teacher.pt`
- 如果需要恢复训练，使用 `last.pt`

## 参考文献

- SF-YOLO 论文：[Source-Free Domain Adaptation for YOLO Object Detection](https://arxiv.org/abs/2409.16538)
- Ultralytics 文档：[Custom Trainer Guide](https://docs.ultralytics.com/zh/guides/custom-trainer/)
- YOLOv26：[Ultralytics 最新 YOLO 模型](https://docs.ultralytics.com/models/yolov26/)

## 更新日志

### 2026-03-06
- ✨ 新增同时验证源域和目标域功能
- ✨ 新增保存教师模型功能（`best_teacher.pt`, `last_teacher.pt`）
- ✨ 新增域差距监控指标（`domain_gap`）
- ✨ CSV 结果文件现在包含源域和目标域的所有指标
