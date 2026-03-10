# AGENTS.md

本项目实现了sf-yolo域适应方法，可以将在源域（比如非雾天环境）训练的目标检测模型微调至适用于目标域（比如雾天环境）。与常规微调的不同之处在于域适应方法只需要目标域的图像，而不需要其标柱。

## 环境配置

- 使用 `uv run` 来运行 Python 脚本，使用`uv add`来安装依赖，这会自动使用项目配置的虚拟环境，**不要**直接使用系统 Python 或 pip 运行脚本或安装包
- 如果项目根目录下存在ENVIRONMENT.md，请遵循其中的说明
- 无视readme.md中关于环境配置的内容

## 数据集

1. CityScape数据集 
已预先转化为Yolo所需格式。源域数据位于（正常天气）"datasets/cityscape_yolo/cityscapes.yaml"，以及目标域数据（雾天）位于"datasets/cityscape_yolo_foggy/cityscapes.yaml"。如果数据集出现问题，请暂停任务向用户询问。

## 目标检测模型

目前使用了两种目标检测模型，因为实现框架不同，导致其训练和验证的脚本和行为存在差异，不能混用
1. Yolov26
使用来自ultralytics库的实现，训练和验证直接使用ultralytics库功能，可参照train_yolo26_cityscapes.py和train_yolo26_cityscapes.py，其sf-yolo训练则通过自定义训练器实现于train_sf-yolo-ultralytics.py。在源域上预训练的权重为source_weights/yolov26l_cityscapes.pt。
2. Yolov5
基本在本项目内实现，少量依赖ultralytics库，但模型与ultralytics内的Yolov5存在一定差别，故必须使用本项目内脚本进行加载，训练和验证。其训练脚本为train_source.py，验证脚本为val.py，sf-yolo训练则使用train_sf-yolo.py。在源域上预训练的权重位于source_weights/yolov5l_cityscapes.pt。
