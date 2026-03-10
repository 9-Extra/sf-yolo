# coding=utf-8
"""
TargetAugment 模块

用于目标域数据增强的模块，包含基于风格迁移的图像增强功能。
主要组件包括：
- enhance_base: 风格迁移基类，提供核心的风格迁移功能
- enhance_vgg16: 基于 VGG16 网络的风格迁移实现
- enhance_style: 风格迁移的简单接口封装

该模块用于 SF-YOLO 的领域自适应训练，通过风格迁移技术将源域图像转换为目标域风格。
"""
