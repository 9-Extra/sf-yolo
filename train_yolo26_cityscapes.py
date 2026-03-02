#!/usr/bin/env python3
"""
Train YOLOv26 on Cityscapes dataset using Ultralytics library
"""

from ultralytics import YOLO
import torch
import os

def main():
    torch.set_float32_matmul_precision('medium')
    # Configuration
    DATA_YAML = "datasets/cityscape-yolo/cityscapes.yaml"
    MODEL_NAME = "yolo26l.pt"
    # models: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt

    EPOCHS = 100
    IMGSZ = 960
    BATCH = 16
    DEVICE = 0
    WORKERS = 8
    PATIENCE = 20
    PROJECT = "cityscapes"

    print("=" * 60)
    print("🚀 YOLOv26 Training on Cityscapes Dataset")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}, Image Size: {IMGSZ}, Batch: {BATCH}")
    print(f"Device: cuda:{DEVICE}")
    print("=" * 60)

    # Load model
    print(f"\n📦 Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # Train
    print(f"\n🔥 Starting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,
        project=PROJECT,
        
        amp=True,
        cache="memory",
        exist_ok=False,
        pretrained=True,
        optimizer="MuSGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        verbose=True,
        compile=True
    )

    # Validate on test set
    print("\n🧪 Running validation...")
    metrics = model.val(data=DATA_YAML, split='test')
    print(f"\nFinal metrics:")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")


if __name__ == '__main__':
    main()