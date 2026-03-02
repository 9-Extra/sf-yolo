#!/usr/bin/env python3
"""
Example script for training SF-YOLO on YOLOv26

This example demonstrates how to use the SFYOLOTrainer for domain adaptation
from Cityscapes to Foggy Cityscapes.
"""

import os
import sys

# Set environment variable for uv
os.environ.setdefault("UV_PROJECT_ENVIRONMENT", "/home/featurize/venv")

from ultralytics import YOLO
from train_sf_yolo_ultralytics import SFYOLOTrainer


def main():
    """Run SF-YOLO training example."""
    
    # Configuration
    # =============
    
    # Model and data paths
    WEIGHTS = "runs/detect/cityscapes/train/weights/best.pt"  # Pretrained Cityscapes YOLOv26 model
    DATA_YAML = "datasets/cityscape/cityscape-foggy-yolo/cityscapes.yaml"
    
    # TAM model paths (trained in Step 1)
    DECODER_PATH = "TargetAugment_train/models/city2foggy/decoder_iter_160000.pth"
    ENCODER_PATH = "TargetAugment_train/pre_trained/vgg16_ori.pth"
    FC1_PATH = "TargetAugment_train/models/city2foggy/fc1_iter_160000.pth"
    FC2_PATH = "TargetAugment_train/models/city2foggy/fc2_iter_160000.pth"
    STYLE_PATH = "TargetAugment_train/data/meanfoggy/meanfoggy.jpg"
    
    # Training parameters
    EPOCHS = 60
    IMGSZ = 960
    BATCH = 16
    DEVICE = 0
    WORKERS = 8
    
    # SF-YOLO specific parameters
    TEACHER_ALPHA = 0.999
    CONF_THRES = 0.4
    IOU_THRES = 0.3
    MAX_DET = 20
    SSM_ALPHA = 0.5
    STYLE_ADD_ALPHA = 0.4
    
    # Output settings
    PROJECT = "runs/sf-yolo"
    NAME = "city2foggy_yolo26"
    
    print("=" * 70)
    print("SF-YOLO Training for YOLOv26")
    print("=" * 70)
    print(f"Model: {WEIGHTS}")
    print(f"Data: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH}, Image Size: {IMGSZ}")
    print(f"Device: cuda:{DEVICE}")
    print("=" * 70)
    
    # Prepare overrides
    overrides = {
        # SF-YOLO specific
        'teacher_alpha': TEACHER_ALPHA,
        'conf_thres': CONF_THRES,
        'iou_thres': IOU_THRES,
        'max_det': MAX_DET,
        'SSM_alpha': SSM_ALPHA,
        'decoder_path': DECODER_PATH,
        'encoder_path': ENCODER_PATH,
        'fc1': FC1_PATH,
        'fc2': FC2_PATH,
        'style_path': STYLE_PATH,
        'style_add_alpha': STYLE_ADD_ALPHA,
        'save_style_samples': False,
        
        # Standard training args
        'model': WEIGHTS,
        'data': DATA_YAML,
        'epochs': EPOCHS,
        'imgsz': IMGSZ,
        'batch': BATCH,
        'device': DEVICE,
        'workers': WORKERS,
        'project': PROJECT,
        'name': NAME,
        'exist_ok': False,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'patience': 20,
        'seed': 0,
        'cos_lr': False,
        'amp': True,
        'freeze': [0],
    }
    
    # Create trainer and start training
    trainer = SFYOLOTrainer(overrides=overrides)
    trainer.train()
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
