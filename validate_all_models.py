"""
验证所有模型在源域和目标域上的性能
包括：YOLOv5、YOLOv26、Fine-tuned、SF-YOLO
"""
from ultralytics import YOLO
import torch

# 模型路径
models = {
    "Fine-tuned YOLOv26": "runs/detect/finetune_yolov26/train/weights/best.pt",
    "Source YOLOv26l": "source_weights/yolov26l_cityscapes.pt",
    "Source YOLOv5l": "source_weights/yolov5l_cityscapes.pt",
    "SF-YOLO Student": "runs/detect/sf-yolo/exp/weights/best.pt",
    "SF-YOLO Teacher": "runs/detect/sf-yolo/exp/weights/best_teacher.pt",
}

# 数据集配置
datasets = {
    "Source (Cityscapes)": "datasets/cityscape_yolo/cityscapes.yaml",
    "Target (Foggy Cityscapes)": "datasets/cityscape_foggy_yolo/cityscapes.yaml",
}

# 验证参数
imgsz = 960
batch = 16

print("=" * 90)
print("模型性能对比 - 包含YOLOv5、YOLOv26、Fine-tuned和SF-YOLO")
print("=" * 90)

results_summary = {}

for model_name, model_path in models.items():
    print(f"\n{'='*90}")
    print(f"正在验证模型: {model_name}")
    print(f"模型路径: {model_path}")
    print(f"{'='*90}")
    
    # 加载模型
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        continue
    
    results_summary[model_name] = {}
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\n--- 在 {dataset_name} 上验证 ---")
        
        try:
            # 运行验证
            metrics = model.val(
                data=dataset_path,
                imgsz=imgsz,
                batch=batch,
                verbose=False,
                save_json=False,
                plots=False,
            )
            
            # 提取指标
            map50 = metrics.box.map50
            map50_95 = metrics.box.map
            
            results_summary[model_name][dataset_name] = {
                "mAP@50": map50,
                "mAP@50-95": map50_95,
            }
            
            print(f"  mAP@50:     {map50:.4f}")
            print(f"  mAP@50-95:  {map50_95:.4f}")
            
        except Exception as e:
            print(f"  验证失败: {e}")
            results_summary[model_name][dataset_name] = {
                "mAP@50": 0.0,
                "mAP@50-95": 0.0,
            }

# 打印总结表格
print("\n" + "=" * 90)
print("性能对比总结")
print("=" * 90)

# 表头
print(f"\n{'模型':<30} {'源域 mAP@50':<15} {'目标域 mAP@50':<15} {'域差距':<15} {'类型':<15}")
print("-" * 90)

model_types = {
    "Fine-tuned YOLOv26": "Fine-tuned",
    "Source YOLOv26l": "Source",
    "Source YOLOv5l": "Source",
    "SF-YOLO Student": "SF-YOLO",
    "SF-YOLO Teacher": "SF-YOLO",
}

for model_name in models.keys():
    if model_name in results_summary:
        source_map50 = results_summary[model_name].get("Source (Cityscapes)", {}).get("mAP@50", 0.0)
        target_map50 = results_summary[model_name].get("Target (Foggy Cityscapes)", {}).get("mAP@50", 0.0)
        gap = source_map50 - target_map50
        model_type = model_types.get(model_name, "Unknown")
        
        print(f"{model_name:<30} {source_map50:<15.4f} {target_map50:<15.4f} {gap:<15.4f} {model_type:<15}")

print("\n" + "=" * 90)
print("详细指标")
print("=" * 90)

for model_name, datasets_results in results_summary.items():
    print(f"\n【{model_name}】")
    for dataset_name, metrics in datasets_results.items():
        print(f"  {dataset_name}:")
        print(f"    mAP@50:     {metrics.get('mAP@50', 0.0):.4f}")
        print(f"    mAP@50-95:  {metrics.get('mAP@50-95', 0.0):.4f}")

# 按类型分组对比
print("\n" + "=" * 90)
print("按类型分组对比")
print("=" * 90)

print("\n【Source 模型对比】")
print(f"{'模型':<25} {'源域':<12} {'目标域':<12} {'域差距':<12}")
print("-" * 65)
for model_name in ["Source YOLOv5l", "Source YOLOv26l"]:
    if model_name in results_summary:
        source_map50 = results_summary[model_name].get("Source (Cityscapes)", {}).get("mAP@50", 0.0)
        target_map50 = results_summary[model_name].get("Target (Foggy Cityscapes)", {}).get("mAP@50", 0.0)
        gap = source_map50 - target_map50
        print(f"{model_name:<25} {source_map50:<12.4f} {target_map50:<12.4f} {gap:<12.4f}")

print("\n【SF-YOLO 模型对比】")
print(f"{'模型':<25} {'源域':<12} {'目标域':<12} {'域差距':<12}")
print("-" * 65)
for model_name in ["SF-YOLO Teacher", "SF-YOLO Student"]:
    if model_name in results_summary:
        source_map50 = results_summary[model_name].get("Source (Cityscapes)", {}).get("mAP@50", 0.0)
        target_map50 = results_summary[model_name].get("Target (Foggy Cityscapes)", {}).get("mAP@50", 0.0)
        gap = source_map50 - target_map50
        print(f"{model_name:<25} {source_map50:<12.4f} {target_map50:<12.4f} {gap:<12.4f}")

print("\n【所有模型目标域性能排序】")
target_results = []
for model_name in models.keys():
    if model_name in results_summary:
        target_map50 = results_summary[model_name].get("Target (Foggy Cityscapes)", {}).get("mAP@50", 0.0)
        target_results.append((model_name, target_map50))

target_results.sort(key=lambda x: x[1], reverse=True)
print(f"{'排名':<6} {'模型':<30} {'目标域 mAP@50':<15}")
print("-" * 55)
for i, (model_name, map50) in enumerate(target_results, 1):
    print(f"{i:<6} {model_name:<30} {map50:<15.4f}")

print("\n验证完成！")
