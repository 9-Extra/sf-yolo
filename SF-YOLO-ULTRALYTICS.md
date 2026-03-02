# SF-YOLO for YOLOv26 (Ultralytics)

This document describes how to use SF-YOLO (Source-Free Domain Adaptation) with YOLOv26 models from the Ultralytics library.

## Overview

SF-YOLO is a domain adaptation method for object detection that adapts a source-trained model to a target domain without requiring target domain labels. This implementation supports YOLOv26 models from the Ultralytics library.

## Prerequisites

1. **Trained Source Model**: A YOLOv26 model trained on the source domain (e.g., Cityscapes)
2. **Target Augmentation Module (TAM)**: Pre-trained style transfer models for domain adaptation
3. **Target Domain Dataset**: Unlabeled target domain images (e.g., Foggy Cityscapes)

## Installation

Ensure you have the required dependencies:

```bash
export UV_PROJECT_ENVIRONMENT=/home/featurize/venv
uv sync
```

## Training Steps

### Step 1: Train the Target Augmentation Module

```bash
cd TargetAugment_train

# Extract target training data
python extract_data.py \
    --scenario_name city2foggy \
    --images_folder ../datasets/CityScapesFoggy/yolov5_format/images \
    --image_suffix jpg

# Train TAM
python train.py \
    --scenario_name city2foggy \
    --content_dir data/city2foggy \
    --style_dir data/meanfoggy \
    --vgg pre_trained/vgg16_ori.pth \
    --save_dir models/city2foggy \
    --n_threads=8 \
    --device 0
```

### Step 2: SF-YOLO Adaptation

Use the `train_sf-yolo-ultralytics.py` script to perform domain adaptation:

```bash
export UV_PROJECT_ENVIRONMENT=/home/featurize/venv

uv run python train_sf-yolo-ultralytics.py \
    --weights runs/detect/cityscapes/train/weights/best.pt \
    --data datasets/cityscape/cityscape-foggy-yolo/cityscapes.yaml \
    --epochs 60 \
    --batch 16 \
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
    --project runs/sf-yolo \
    --name city2foggy_yolo26
```

Or use the example script:

```bash
uv run python example_train_sf_yolo.py
```

## Command Line Arguments

### Model and Data Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--weights` | `yolo26n.pt` | Path to pretrained source model |
| `--data` | Required | Path to target domain dataset YAML |
| `--epochs` | 60 | Number of training epochs |
| `--imgsz` | 960 | Image size for training |
| `--batch` | 16 | Batch size |
| `--device` | 0 | CUDA device ID |
| `--workers` | 8 | Number of dataloader workers |
| `--project` | `runs/sf-yolo` | Project directory |
| `--name` | `exp` | Experiment name |

### SF-YOLO Specific Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--teacher_alpha` | 0.999 | EMA decay rate for teacher model |
| `--conf_thres` | 0.4 | Confidence threshold for pseudo labels |
| `--iou_thres` | 0.3 | IoU threshold for NMS |
| `--max_det` | 20 | Maximum detections per image |
| `--SSM_alpha` | 0.0 | SSM momentum (0 to disable) |

### TAM Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `--decoder_path` | Yes | Path to TAM decoder weights |
| `--encoder_path` | Yes | Path to VGG encoder weights |
| `--fc1` | Yes | Path to FC1 weights |
| `--fc2` | Yes | Path to FC2 weights |
| `--style_path` | No | Path to style image (empty for random) |
| `--style_add_alpha` | 1.0 | Style transfer intensity |
| `--save_style_samples` | False | Save augmented style samples |

## Python API Usage

You can also use the `SFYOLOTrainer` class directly in Python:

```python
from ultralytics import YOLO
from train_sf_yolo_ultralytics import SFYOLOTrainer

# Prepare configuration
overrides = {
    'teacher_alpha': 0.999,
    'conf_thres': 0.4,
    'iou_thres': 0.3,
    'SSM_alpha': 0.5,
    'decoder_path': 'path/to/decoder.pth',
    'encoder_path': 'path/to/encoder.pth',
    'fc1': 'path/to/fc1.pth',
    'fc2': 'path/to/fc2.pth',
    'style_path': 'path/to/style.jpg',
    'style_add_alpha': 0.4,
    
    # Standard training args
    'model': 'yolo26n.pt',
    'data': 'target_dataset.yaml',
    'epochs': 60,
    'imgsz': 960,
    'batch': 16,
    'device': 0,
}

# Create trainer and train
trainer = SFYOLOTrainer(overrides=overrides)
trainer.train()
```

## Key Features

### 1. Dual Model Architecture
- **Student Model**: Trained on stylized target domain images
- **Teacher Model**: Generates pseudo-labels using EMA updates

### 2. Target Augmentation Module (TAM)
- Applies style transfer to adapt source domain images to target domain style
- Pre-trained AdaIN-based style transfer

### 3. Pseudo-Label Generation
- Teacher model generates pseudo-labels on original images
- NMS filtering with configurable confidence and IoU thresholds

### 4. Stable Student Momentum (SSM)
- Optional: Transfer teacher weights to student at each epoch
- Controlled by `SSM_alpha` parameter

## Output Structure

Training outputs are saved to `runs/sf-yolo/<name>/`:

```
runs/sf-yolo/
└── city2foggy_yolo26/
    ├── weights/
    │   ├── best.pt          # Best model based on validation fitness
    │   ├── last.pt          # Last epoch model
    │   └── epoch{X}.pt      # Periodic checkpoints (if save_period > 0)
    ├── args.yaml            # Training arguments
    ├── results.csv          # Training metrics
    ├── train_batch*.jpg     # Training batch visualizations
    ├── enhance_style_samples/  # Style augmentation samples (if enabled)
    └── ...
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch 8`
- Reduce image size: `--imgsz 640`
- Enable gradient checkpointing (if available)

### Poor Adaptation Performance
- Increase training epochs: `--epochs 100`
- Adjust style intensity: `--style_add_alpha 0.5`
- Lower confidence threshold: `--conf_thres 0.3`
- Enable SSM: `--SSM_alpha 0.5`

### TAM Loading Errors
- Verify all TAM model paths are correct
- Ensure VGG encoder weights are compatible
- Check CUDA availability for TAM

## References

- SF-YOLO Paper: [Source-Free Domain Adaptation for YOLO Object Detection](https://arxiv.org/abs/2409.16538)
- Ultralytics Documentation: [Custom Trainer Guide](https://docs.ultralytics.com/zh/guides/custom-trainer/)
- YOLOv26: Latest YOLO model from Ultralytics
