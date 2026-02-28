# TAM (Target Augmentation Module) Overview

## What is TAM?

TAM is a **neural style transfer module** based on Adaptive Instance Normalization (AdaIN) that generates style-augmented target domain images for SF-YOLO training.

## Architecture

```
Input Image (Foggy)
       ↓
┌─────────────────┐
│  VGG16 Encoder  │ → Extract content features
│  (up to relu4_1)│
└─────────────────┘
       ↓
┌─────────────────┐
│  AdaIN          │ → Blend content + style
│  (fc1, fc2)     │
└─────────────────┘
       ↓
┌─────────────────┐
│  Decoder        │ → Reconstruct image
│  (learned)      │
└─────────────────┘
       ↓
Style-Augmented Output
```

## Components

### 1. Encoder (VGG16)
- Pre-trained on ImageNet
- Extracts content features up to `relu4_1`
- Parameters frozen during SF-YOLO training

### 2. Decoder
- Mirror structure of encoder
- Learned during Stage 1 training
- Reconstructs image from AdaIN features

### 3. FC Layers (fc1, fc2)
- Predict AdaIN parameters
- Learned during Stage 1 training
- Enable adaptive style transfer

## AdaIN Formula

```
AdaIN(x, y) = σ(y) * (x - μ(x)) / σ(x) + μ(y)

Where:
- x = content features
- y = style features
- μ = mean
- σ = standard deviation
```

## Stage 1 Training

TAM is trained separately before SF-YOLO adaptation:

```bash
cd TargetAugment_train

# Extract target data
python extract_data.py \
    --scenario_name city2foggy \
    --images_folder ../datasets/CityScapesFoggy/yolov5_format/images \
    --image_suffix png

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

Training takes ~160k iterations (several hours to a day).

## Stage 2 Usage

In SF-YOLO training, TAM generates augmented images:

```python
# From train_sf-yolo.py
imgs_style = get_style_images(imgs_255, opt, adain) / 255
```

Parameters:
- `style_add_alpha`: Controls style intensity (0.0-1.0)
  - 0.4 recommended for SF-YOLO
  - Higher = more style transfer
  - Lower = preserve more original content

## Scenarios

SF-YOLO supports multiple domain adaptation scenarios:

| Scenario | Source | Target | Style Image |
|----------|--------|--------|-------------|
| city2foggy | Cityscapes | Foggy Cityscapes | meanfoggy.jpg |
| voc2clipart | PASCAL VOC | Clipart | meanclipart.jpg |
| voc2wc | PASCAL VOC | Watercolor | meanwc.jpg |
| KC | - | - | - |
| city | - | - | - |
| bdd100k | - | BDD100K | - |

## Common Issues

### Output is gray/blank
- **Cause**: Alpha too high (1.0) with mean style image
- **Fix**: Use alpha=0.3-0.5

### CUDA out of memory
- **Cause**: Large batch or image size
- **Fix**: Use CPU mode or reduce image size

### Style not applied
- **Cause**: Wrong weights loaded
- **Fix**: Verify decoder, fc1, fc2 paths

## References

- Paper: https://arxiv.org/abs/2409.16538
- AdaIN: https://arxiv.org/abs/1703.06868
- Code: https://github.com/naoto0804/pytorch-AdaIN
