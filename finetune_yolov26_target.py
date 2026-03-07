#!/usr/bin/env python3
"""
直接使用目标域数据集微调 YOLOv26
作为 SF-YOLO 的对比实验（下界 baseline）

此脚本不使用域适应技术，直接用目标域数据微调预训练模型
"""

from ultralytics import YOLO
import torch
import argparse
from pathlib import Path


def main():
    torch.multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description='Finetune YOLOv26 on target domain (Foggy Cityscapes)')
    parser.add_argument('--model', type=str, default='source_weights/yolov26l_cityscapes.pt',
                        help='Pretrained model path (source domain trained)')
    parser.add_argument('--data', type=str, default='datasets/cityscape_foggy_yolo/cityscapes.yaml',
                        help='Target domain dataset config')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=960, help='Image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--lr0', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor')
    parser.add_argument('--device', type=str, default='0', help='Device to use')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--project', type=str, default='finetune_yolov26', help='Project directory')
    parser.add_argument('--name', type=str, default='', help='Experiment name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--freeze', type=int, default=0, help='Freeze first N layers')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every N epochs')
    args = parser.parse_args()

    print("=" * 80)
    print("🎯 YOLOv26 目标域直接微调")
    print("=" * 80)
    print(f"模型: {args.model}")
    print(f"目标域数据: {args.data}")
    print(f"训练轮数: {args.epochs}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"批次大小: {args.batch}")
    print(f"学习率: {args.lr0} -> {args.lr0 * args.lrf}")
    print(f"设备: cuda:{args.device}")
    print("=" * 80)

    # 检查模型文件
    model_path = Path(args.model)
    assert model_path.exists(), f"模型{model_path}不存在"
    
    # 加载模型
    print(f"\n📦 加载模型: {args.model}")
    model = YOLO(args.model)
    print(f"✅ 模型加载成功")
    print(f"   类别数: {model.model.nc}")
    print(f"   模型类型: YOLOv26")

    # 冻结层（可选）
    if args.freeze > 0:
        print(f"\n🧊 冻结前 {args.freeze} 层...")
        # YOLOv26 使用不同的层命名方式
        freeze_layers = list(model.model.model[:args.freeze])
        for layer in freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"✅ 已冻结 {len(freeze_layers)} 层")

    # 开始训练
    print(f"\n🚀 开始训练...")
    print(f"   注意: 这是直接在目标域上微调的 baseline（无域适应）")
    print(f"   用于与 SF-YOLO 进行对比\n")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        save_period=args.save_period,
        # 数据增强设置（与 SF-YOLO 保持一致）
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
        copy_paste=0.0,
        # 优化器设置
        amp=True,
        cache="memory",
        optimizer="MuSGD",
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # 其他设置
        box=7.5,
        cls=0.5,
        dfl=1.5,
        compile=True,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
    )

    # 训练完成
    print("\n" + "=" * 80)
    print("✅ 训练完成!")
    print("=" * 80)
    
    # 打印最佳结果
    if hasattr(results, 'results_dict'):
        print("\n📊 最佳验证结果:")
        for key, value in results.results_dict.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
    
    # 验证最终模型
    print("\n🧪 在目标域验证最终模型...")
    metrics = model.val(data=args.data, split='val')
    print(f"   mAP@50: {metrics.box.map50:.4f}")
    print(f"   mAP@50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    # 同时验证源域性能（查看遗忘程度）
    source_data = "datasets/cityscape_yolo/cityscapes.yaml"
    if Path(source_data).exists():
        print("\n🧪 在源域验证（检查灾难性遗忘）...")
        metrics_source = model.val(data=source_data, split='val')
        print(f"   源域 mAP@50: {metrics_source.box.map50:.4f}")
        print(f"   源域 mAP@50-95: {metrics_source.box.map:.4f}")
    
    print("\n" + "=" * 80)
    print(f"📁 结果保存在: {Path(args.project) / args.name}")
    print("=" * 80)


if __name__ == '__main__':
    main()
