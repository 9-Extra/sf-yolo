# SF-YOLO: Source-Free Domain Adaptation for YOLO Object Detection

## Project Overview

This repository contains the implementation of **SF-YOLO** (Source-Free YOLO), a source-free domain adaptation method for YOLO object detection. The project was presented at the ECCV 2024 Workshop on Out-of-Distribution Generalization in Computer Vision Foundation Models.

Paper: https://arxiv.org/abs/2409.16538

SF-YOLO adapts object detection models to target domains without requiring source domain data during adaptation. It combines:
- **Target Augmentation Module**: A neural style transfer module based on AdaIN to generate style-augmented target images
- **Mean Teacher Framework**: A student-teacher architecture with exponential moving average (EMA) for stable pseudo-label learning
- **Source-Free Adaptation**: Uses only unlabeled target domain images and a pre-trained source model

## Technology Stack

- **Python**: >= 3.14
- **Deep Learning Framework**: PyTorch >= 2.10.0, torchvision >= 0.25.0
- **Base Architecture**: YOLOv5 (Ultralytics implementation)
- **Package Manager**: uv (with uv.lock)
- **Key Dependencies**:
  - ultralytics >= 8.4.15
  - numpy, pillow, pyyaml
  - tqdm, seaborn (for visualization)
  - dill, gitpython (for utilities)

## Project Structure

```
sf-yolo/
├── models/                    # YOLO model definitions
│   ├── yolo.py               # YOLO-specific modules (Detect, Segment heads)
│   ├── common.py             # Common building blocks (Conv, C3, SPPF, etc.)
│   ├── experimental.py       # Experimental features and model loading
│   ├── tf.py                 # TensorFlow model conversions
│   └── *.yaml                # Model configurations (yolov5s/m/l/x.yaml)
├── utils/                     # Utility modules
│   ├── dataloaders.py        # Dataset and DataLoader implementations
│   ├── loss.py               # Loss functions (ComputeLoss, FocalLoss, etc.)
│   ├── metrics.py            # Evaluation metrics (mAP, confusion matrix)
│   ├── torch_utils.py        # PyTorch utilities (EMA, DDP, device selection)
│   ├── general.py            # General utilities
│   ├── plots.py              # Visualization functions
│   ├── augmentations.py      # Image augmentation
│   └── loggers/              # Logging integrations (Comet, etc.)
├── data/                      # Dataset configuration files
│   ├── cityscapes.yaml       # Cityscapes dataset config
│   ├── foggy_cityscapes.yaml # Foggy Cityscapes dataset config
│   └── *.yaml                # Other dataset configs (COCO, VOC, etc.)
├── TargetAugment/             # Runtime target augmentation module
│   ├── enhance_base.py       # Base class for style transfer
│   ├── enhance_vgg16.py      # VGG16-based encoder/decoder
│   └── enhance_style.py      # Style application interface
├── TargetAugment_train/       # Training scripts for augmentation module
│   ├── train.py              # Train the style transfer network
│   ├── net.py                # Network architecture definitions
│   ├── function.py           # AdaIN functions
│   ├── extract_data.py       # Data extraction utilities
│   └── sampler.py            # Data sampling utilities
├── datasets/                  # Dataset storage (external, not in git)
├── source_weights/            # Pre-trained source model weights
├── runs/                      # Training runs and experiment outputs
├── train_sf-yolo.py          # Main SF-YOLO adaptation script ⭐
├── train_source.py           # Standard YOLOv5 training script
├── val.py                    # Validation script
├── export.py                 # Model export to various formats
├── benchmarks.py             # Performance benchmarking
├── hubconf.py                # PyTorch Hub integration
├── cityscape2yolo.py         # Cityscapes to YOLO format converter
├── pyproject.toml            # Python project configuration
└── setup.cfg                 # Code style and testing configuration
```

## Key Configuration Files

### pyproject.toml
- Defines project metadata and dependencies
- Configures PyTorch CUDA 13.0 index for GPU support
- Uses Tsinghua University PyPI mirror for faster downloads in China

### setup.cfg
- **Code Style**: flake8 with 120 character line length
- **Import Sorting**: isort with 120 character line length
- **Formatting**: yapf with PEP8 base style
- **Testing**: pytest configuration with doctest support

## Build and Setup Commands

### Environment Setup

```bash
# Create virtual environment with uv
uv venv

# Install dependencies
uv pip install -e .

# Alternative with conda (legacy)
conda create --name sf-yolo python=3.11
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Dataset Preparation

```bash
# Convert Cityscapes to YOLO format
python cityscape2yolo.py \
    --image_path leftImg8bit \
    --source_label_path gtFine \
    --output_dir yolov5_format \
    --foggy_beta _foggy_beta_0.02
```

## Training Workflow

SF-YOLO uses a **two-stage training process**:

### Stage 1: Train Target Augmentation Module

Extract target domain training data and train the style transfer network:

```bash
# Step 1: Extract target data
cd TargetAugment_train
python extract_data.py \
    --scenario_name city2foggy \
    --images_folder ../datasets/CityScapesFoggy/yolov5_format/images \
    --image_suffix jpg

# Step 2: Train augmentation module
python train.py \
    --scenario_name city2foggy \
    --content_dir data/city2foggy \
    --style_dir data/meanfoggy \
    --vgg pre_trained/vgg16_ori.pth \
    --save_dir models/city2foggy \
    --n_threads=8 \
    --device 0
```

Supported scenarios: `voc2clipart`, `voc2wc`, `city2foggy`, `KC`, `city`, `bdd100k`

### Stage 2: SF-YOLO Adaptation

Run source-free domain adaptation using the pre-trained source model:

```bash
# Download source model weights to ./source_weights/
# Available at: https://drive.proton.me/urls/5WFVDJBDAC#EPs8OZmXtbWq

# Run SF-YOLO adaptation
python train_sf-yolo.py \
    --epochs 60 \
    --batch-size 16 \
    --data foggy_cityscapes.yaml \
    --weights ./source_weights/yolov5l_cityscapes.pt \
    --decoder_path TargetAugment_train/models/city2foggy/decoder_iter_160000.pth \
    --encoder_path TargetAugment_train/pre_trained/vgg16_ori.pth \
    --fc1 TargetAugment_train/models/city2foggy/fc1_iter_160000.pth \
    --fc2 TargetAugment_train/models/city2foggy/fc2_iter_160000.pth \
    --style_add_alpha 0.4 \
    --style_path ./TargetAugment_train/data/meanfoggy/meanfoggy.jpg \
    --SSM_alpha 0.5 \
    --device 0
```

### Standard YOLO Training (Source Model)

To train a source model from scratch:

```bash
python train_source.py \
    --data cityscapes.yaml \
    --weights yolov5l.pt \
    --epochs 300 \
    --batch-size 16 \
    --device 0
```

## Validation and Evaluation

```bash
# Validate a trained model
python val.py \
    --weights runs/train/exp/weights/best_student.pt \
    --data foggy_cityscapes.yaml \
    --img 640

# Benchmark different export formats
python benchmarks.py \
    --weights runs/train/exp/weights/best_student.pt \
    --img 640
```

## Model Export

Export trained models to various deployment formats:

```bash
python export.py \
    --weights runs/train/exp/weights/best_student.pt \
    --include torchscript onnx openvino engine coreml tflite
```

Supported formats: PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TensorFlow (SavedModel, GraphDef, Lite, Edge TPU), TensorFlow.js, PaddlePaddle

## Code Style Guidelines

Based on `setup.cfg`:

- **Line Length**: 120 characters maximum
- **Linting**: flake8 with specific ignores for YOLO-style code
  - E731: Allow lambda expressions
  - F405/F403: Allow star imports
  - E402: Allow imports not at top of file
- **Import Style**: Follow existing patterns with `from module import *` for utils
- **Formatting**: yapf with PEP8 base, 2 spaces before comments

### Development Conventions

1. **File Headers**: Include YOLOv5 attribution comment:
   ```python
   # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
   """Module description"""
   ```

2. **Imports**: Group as:
   - Standard library imports
   - Third-party imports (torch, numpy, etc.)
   - Local module imports

3. **Device Handling**: Use `select_device()` from `utils.torch_utils` for automatic GPU/CPU selection

4. **Distributed Training**: Support DDP via `LOCAL_RANK`, `RANK`, `WORLD_SIZE` environment variables

## Testing Instructions

The project uses pytest with doctest support:

```bash
# Run tests
pytest

# Run with doctest
pytest --doctest-modules

# Specific test durations
pytest --durations=25
```

## Key Module Descriptions

### train_sf-yolo.py
The main SF-YOLO adaptation script. Key features:
- **Dual Model Architecture**: Student (trainable) and Teacher (EMA of student)
- **Style Augmentation**: Uses TargetAugment module to generate styled images
- **Pseudo-Labeling**: Teacher generates pseudo-labels for target domain
- **SSM (Style-Supervised Module)**: Controlled style mixing with `SSM_alpha`

Key arguments:
- `--teacher_alpha`: EMA decay rate for teacher model
- `--style_add_alpha`: Style transfer intensity
- `--SSM_alpha`: Style-supervised mixing coefficient
- `--conf_thres`, `--iou_thres`: Pseudo-label filtering thresholds

### TargetAugment Module
Neural style transfer based on VGG16:
- Encoder: VGG16 features up to relu4_1
- Decoder: Learned upsampling network
- FC layers: Learnable AdaIN parameters (fc1, fc2)
- Adaptive Instance Normalization for style-content mixing

### models/yolo.py
Core YOLOv5 architecture:
- `Detect`: Detection head for object detection
- `Segment`: Segmentation head for instance segmentation
- `DetectionModel`: Full detection model with backbone + head

## Security Considerations

1. **Model Weights**: Always verify downloaded weights checksums when available
2. **Dataset Paths**: Use absolute paths in production; relative paths are for development
3. **Export Formats**: TensorRT and Edge TPU exports require specific hardware
4. **Multi-GPU**: Use `torch.distributed.run` instead of DataParallel for better performance

## Hardware Requirements

- **Training**: NVIDIA GPU with CUDA support (tested with CUDA 11.8/13.0)
- **Minimum VRAM**: 8GB for batch size 16 with YOLOv5l
- **CPU Threads**: Limited to 16 threads to avoid excessive CPU usage on shared servers

## Acknowledgments

This implementation builds upon:
- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [AdaIN](https://github.com/naoto0804/pytorch-AdaIN) for neural style transfer
- [LODS](https://github.com/Flashkong/Source-Free-Object-Detection-by-Learning-to-Overlook-Domain-Style) for source-free object detection ideas
