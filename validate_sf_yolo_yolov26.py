#!/usr/bin/env python3
"""
验证 SF-YOLO 训练后的 YOLOv26 模型性能
"""

from ultralytics import YOLO
import torch

def main():
    # 模型路径
    MODEL_PATH = "runs/detect/sf-yolo/city2foggy_yolo26/weights/best.pt"
    
    # 数据配置
    SOURCE_DATA = "datasets/cityscape_yolo/cityscapes.yaml"  # 源域: Cityscapes
    TARGET_DATA = "datasets/cityscape_foggy_yolo/cityscapes.yaml"  # 目标域: Foggy Cityscapes
    
    # 验证参数
    IMGSZ = 960
    BATCH = 16
    DEVICE = 0
    
    print("=" * 70)
    print("🚀 SF-YOLO YOLOv26 模型性能验证")
    print("=" * 70)
    print(f"模型路径: {MODEL_PATH}")
    print(f"图像尺寸: {IMGSZ}")
    print(f"批次大小: {BATCH}")
    print(f"设备: cuda:{DEVICE}")
    print("=" * 70)
    
    # 加载模型
    print(f"\n📦 加载模型...")
    model = YOLO(MODEL_PATH)
    print(f"✅ 模型加载成功")
    
    results = {}
    
    # 在目标域 (Foggy Cityscapes) 上验证
    print("\n" + "=" * 70)
    print("🧪 在目标域 (Foggy Cityscapes) 上验证")
    print("=" * 70)
    metrics_target = model.val(
        data=TARGET_DATA,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        split='val',
        verbose=False
    )
    results['target'] = {
        'mAP@50': metrics_target.box.map50,
        'mAP@50-95': metrics_target.box.map,
        'Precision': metrics_target.box.mp,
        'Recall': metrics_target.box.mr,
    }
    
    print(f"  mAP@50:     {metrics_target.box.map50:.4f}")
    print(f"  mAP@50-95:  {metrics_target.box.map:.4f}")
    print(f"  Precision:  {metrics_target.box.mp:.4f}")
    print(f"  Recall:     {metrics_target.box.mr:.4f}")
    
    # 在源域 (Cityscapes) 上验证
    print("\n" + "=" * 70)
    print("🧪 在源域 (Cityscapes) 上验证")
    print("=" * 70)
    metrics_source = model.val(
        data=SOURCE_DATA,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        split='val',
        verbose=False
    )
    results['source'] = {
        'mAP@50': metrics_source.box.map50,
        'mAP@50-95': metrics_source.box.map,
        'Precision': metrics_source.box.mp,
        'Recall': metrics_source.box.mr,
    }
    
    print(f"  mAP@50:     {metrics_source.box.map50:.4f}")
    print(f"  mAP@50-95:  {metrics_source.box.map:.4f}")
    print(f"  Precision:  {metrics_source.box.mp:.4f}")
    print(f"  Recall:     {metrics_source.box.mr:.4f}")
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("📊 性能汇总")
    print("=" * 70)
    print(f"{'Metric':<15} {'Source (Cityscapes)':>20} {'Target (Foggy)':>20}")
    print("-" * 70)
    for metric in ['mAP@50', 'mAP@50-95', 'Precision', 'Recall']:
        print(f"{metric:<15} {results['source'][metric]:>20.4f} {results['target'][metric]:>20.4f}")
    
    print("\n" + "=" * 70)
    print("✅ 验证完成!")
    print("=" * 70)

if __name__ == '__main__':
    main()
