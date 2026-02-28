# AGENTS.md

本项目使用 `uv` 作为 Python 包管理器和运行环境。

## 最佳实践

+ 同步盘，也称为 work 目录，绝对路径为/home/featurize/work，是Featurize为所有用户提供的一种文件持久化的方案。用户可以将自己的代码、模型文件保存至同步盘中，这样即使在归还实例后同步盘中的文件也不会丢失。
+ 代码和模型文件始终存放在work目录中。
+ 不要在featurize/work目录中存放数据集

## 环境配置

### 设置环境变量

在运行任何命令之前，必须设置以下环境变量：

```bash
export UV_PROJECT_ENVIRONMENT=/home/featurize/venv
```

建议将此行添加到 `~/.bashrc` 或 `~/.zshrc` 中以确保持久生效：

```bash
echo 'export UV_PROJECT_ENVIRONMENT=/home/featurize/venv' >> ~/.bashrc
source ~/.bashrc
```

## 使用 uv 运行脚本

### 运行 Python 脚本

使用 `uv run` 来运行 Python 脚本，这会自动使用项目配置的虚拟环境：

```bash
# 运行训练脚本
uv run train_sf-yolo.py

# 运行验证脚本
uv run val.py

# 运行其他脚本
uv run benchmarks.py
```

### 管理依赖

所有依赖都在 `pyproject.toml` 中定义，使用以下命令管理：

```bash
# 安装项目依赖（根据 pyproject.toml 和 uv.lock）
uv sync

# 添加新依赖
uv add <package-name>

# 添加开发依赖
uv add --dev <package-name>

# 更新依赖
uv sync --upgrade

# 更新特定包
uv add --upgrade <package-name>
```

### 进入项目环境

如果需要进入交互式 Python shell 或手动执行命令：

```bash
# 进入项目的 Python 环境
uv run python

# 或者使用 uv run 直接执行命令
uv run python -c "import torch; print(torch.__version__)"
```

## 重要提示

- **不要**直接使用系统 Python 或 pip 运行脚本或安装包
- 始终确保 `UV_PROJECT_ENVIRONMENT` 环境变量已设置
- 项目使用清华镜像源加速依赖下载，并在需要时从 PyTorch 官方源安装 CUDA 版本的 torch/torchvision

## 项目依赖说明

主要依赖包括：
- PyTorch (CUDA 13.0 版本)
- ultralytics
- 其他数据处理和可视化库

详见 `pyproject.toml` 获取完整依赖列表。

## 模型训练与验证

### YOLOv26 训练（Ultralytics）

使用最新的 YOLOv26 模型在 Cityscapes 源域数据上训练：

```python
# train_yolo26.py
from ultralytics import YOLO

# 加载 YOLOv26n (Nano) 模型
model = YOLO("yolo26n.pt")

# 训练配置
results = model.train(
    data="/home/featurize/work/sf-yolo/data/cityscapes_original.yaml",
    epochs=100,
    imgsz=960,
    batch=16,
    device=0,
    workers=4,
    patience=20,
    project="runs/train_yolo26",
    name="cityscapes_baseline",
    exist_ok=True,
)
```

运行训练：
```bash
export UV_PROJECT_ENVIRONMENT=/home/featurize/venv
uv run python train_yolo26.py
```

**模型选择**：
- `yolo26n.pt` - Nano (2.4M 参数，最快)
- `yolo26s.pt` - Small
- `yolo26m.pt` - Medium
- `yolo26l.pt` - Large
- `yolo26x.pt` - Extra Large

### YOLOv26 验证

在源域或目标域上验证训练好的 YOLOv26 模型：

```python
# validate_yolo26.py
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/train_yolo26/cityscapes_baseline/weights/best.pt")

# 在源域（Cityscapes）验证
metrics_source = model.val(
    data="/home/featurize/work/sf-yolo/data/cityscapes_original.yaml",
    imgsz=960,
    batch=16,
    verbose=False
)
print(f"Source - mAP@50: {metrics_source.box.map50:.4f}")

# 在目标域（Foggy Cityscapes）验证
metrics_target = model.val(
    data="/home/featurize/work/sf-yolo/data/foggy_cityscapes.yaml",
    imgsz=960,
    batch=16,
    verbose=False
)
print(f"Target - mAP@50: {metrics_target.box.map50:.4f}")
```

运行验证：
```bash
uv run python validate_yolo26.py
```

### YOLOv5 基线模型验证

验证预训练的 YOLOv5 基线模型（`source_weights/yolov5l_cityscapes.pt`）：

**在源域（Cityscapes）验证：**
```bash
uv run val.py \
    --weights source_weights/yolov5l_cityscapes.pt \
    --data data/cityscapes_original.yaml \
    --imgsz 960 \
    --batch-size 16 \
    --task val
```

**在目标域（Foggy Cityscapes）验证：**
```bash
uv run val.py \
    --weights source_weights/yolov5l_cityscapes.pt \
    --data data/foggy_cityscapes.yaml \
    --imgsz 960 \
    --batch-size 16 \
    --task val
```

### 数据准备

#### Cityscapes 原始数据（源域）

确保原始 Cityscapes 数据已转换为 YOLO 格式：

```bash
# 原始 Cityscapes → YOLO 格式
uv run cityscape2yolo.py \
    --image_path /home/featurize/cityscape/leftImg8bit \
    --source_label_path /home/featurize/cityscape/gtFine \
    --output_dir /home/featurize/datasets/cityscape-yolo \
    --foggy_beta ""
```

#### Foggy Cityscapes（目标域）

```bash
# Foggy Cityscapes → YOLO 格式
uv run cityscape2yolo.py \
    --image_path /home/featurize/leftImg8bit_foggy \
    --source_label_path /home/featurize/cityscape/gtFine \
    --output_dir /home/featurize/datasets/cityscape-foggy-yolo \
    --foggy_beta _foggy_beta_0.02
```
