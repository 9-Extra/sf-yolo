---
name: tam-image-generator
description: Generate TAM (Target Augmentation Module) style-augmented images for SF-YOLO domain adaptation visualization. Use when the user wants to (1) visualize TAM style transfer effects, (2) generate comparison images between original and style-augmented samples, (3) debug TAM model outputs, or (4) create dataset samples for SF-YOLO training visualization. Requires pre-trained TAM weights (decoder, encoder, fc1, fc2) and style image.
---

# TAM Image Generator

Generate style-augmented images using SF-YOLO's Target Augmentation Module (TAM) for visualization and debugging.

## Overview

TAM is a neural style transfer module based on AdaIN that generates style-augmented target domain images. This skill helps visualize the effect of TAM on input images.

## Prerequisites

Required files (paths relative to SF-YOLO project root):

```
TargetAugment_train/pre_trained/vgg16_ori.pth      # VGG16 encoder
TargetAugment_train/models/city2foggy/decoder_iter_160000.pth  # Decoder
TargetAugment_train/models/city2foggy/fc1_iter_160000.pth      # FC1
TargetAugment_train/models/city2foggy/fc2_iter_160000.pth      # FC2
TargetAugment_train/data/meanfoggy/meanfoggy.jpg   # Style image
```

## Quick Start

### Basic Usage

Generate 5 samples with default settings:

```bash
python scripts/generate_tam_samples.py \
    --num_samples 5 \
    --output_dir tam_output
```

### Custom Parameters

```bash
python scripts/generate_tam_samples.py \
    --num_samples 10 \
    --style_add_alpha 0.4 \
    --source_images datasets/CityScapesFoggy/yolov5_format/images/val \
    --output_dir my_samples \
    --device cuda
```

## Output Files

The script generates three types of files for each sample:

| File Pattern | Description |
|-------------|-------------|
| `XX_comparison_*.png` | Side-by-side comparison (original vs augmented) |
| `XX_original.png` | Original input image |
| `XX_tam_augmented.png` | Style-augmented output |
| `00_reference_style_image.jpg` | The style image used |

## Key Parameters

- `--style_add_alpha`: Style transfer intensity (0.0-1.0, default: 0.4)
  - 0.0 = no change
  - 1.0 = full style transfer
  - Recommended: 0.3-0.5 for SF-YOLO
  
- `--source_images`: Input image directory (default: val set)

- `--num_samples`: Number of samples to generate

- `--device`: cuda (recommended) or cpu

## Common Scenarios

### 1. Check TAM Installation

Verify all required weights are present:

```bash
ls TargetAugment_train/models/city2foggy/*.pth
ls TargetAugment_train/pre_trained/vgg16_ori.pth
ls TargetAugment_train/data/meanfoggy/meanfoggy.jpg
```

### 2. Debug TAM Output Issues

If output is gray/blank, check:
- Alpha value not too high (try 0.4)
- Style image is valid
- Weights loaded correctly

### 3. Generate Training Visualization

Create samples for paper/presentation:

```bash
python scripts/generate_tam_samples.py \
    --num_samples 20 \
    --style_add_alpha 0.4 \
    --output_dir paper_figures/tam_samples
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Use `--device cpu` or reduce batch size |
| "No module named torch" | Activate venv: `source .venv/bin/activate` |
| Output is gray | Reduce alpha value (try 0.2-0.4) |
| Missing weights | Download from SF-YOLO release or train Stage 1 |

## Related Documentation

- SF-YOLO paper: https://arxiv.org/abs/2409.16538
- AdaIN: https://github.com/naoto0804/pytorch-AdaIN

## Notes

- TAM requires GPU for reasonable speed (CUDA)
- Default alpha=0.4 matches SF-YOLO training setting
- Generated images are 640x640 (YOLO input size)
