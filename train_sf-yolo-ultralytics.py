#!/usr/bin/env python3
"""
SF-YOLO 训练脚本（基于 YOLOv26 / Ultralytics）

本脚本实现了 SF-YOLO（Source-Free YOLO）无源域自适应方法，
用于 YOLOv26 模型的域自适应训练。使用 Ultralytics 的自定义训练器 API。

参考文档: https://docs.ultralytics.com/zh/guides/custom-trainer/
"""

import argparse
import math
import os
import sys
import time
from copy import deepcopy, copy
from pathlib import Path
import warnings
import cv2

import numpy as np
import torch
import torchvision

# 添加项目根目录到 Python 路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK, colorstr, ops, DEFAULT_CFG, YAML
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import unwrap_model, autocast
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo import detect

# SF-YOLO 相关模块导入
from TargetAugment.enhance_style import get_style_images
from TargetAugment.enhance_vgg16 import enhance_vgg16


class WeightEMA:
    """权重指数移动平均（EMA）优化器
    
    使用无偏置的 Mean Teacher EMA 更新策略，
    教师模型参数通过学生模型参数的指数移动平均来更新。
    
    参考: github.com/kinredon/umt
    """
    
    def __init__(self, teacher_params, student_params, alpha=0.999):
        """初始化 EMA 优化器
        
        Args:
            teacher_params: 教师模型参数迭代器
            student_params: 学生模型参数迭代器
            alpha: EMA 衰减系数，越接近 1 表示历史权重占比越高
        """
        self.teacher_params = list(teacher_params)
        self.student_params = list(student_params)
        self.alpha = alpha
        
    def step(self):
        """执行 EMA 更新，将学生模型的参数更新到教师模型"""
        # 注意：EMA 更新通过 data 属性直接操作，不需要 requires_grad
        for teacher_param, student_param in zip(self.teacher_params, self.student_params):
            # EMA 公式: θ_teacher = α * θ_teacher + (1-α) * θ_student
            teacher_param.data.mul_(self.alpha).add_(student_param.data, alpha=1 - self.alpha)


class SFYOLOTrainer(DetectionTrainer):
    """SF-YOLO 自定义训练器（无源域自适应）
    
    本训练器实现了 SF-YOLO 的核心功能：
    1. 双模型架构（学生模型 + 教师模型，使用 EMA 机制）
    2. 目标域增强模块（TAM）进行风格迁移
    3. 教师模型生成伪标签
    4. 学生模型在风格化图像上训练，使用伪标签监督
    5. 可选的 SSM（稳定学生动量）机制
    6. 双域验证（同时验证目标域和源域）
    
    Attributes:
        model_teacher: 教师模型，用于生成伪标签
        optimizer_teacher: 教师模型的 EMA 优化器
        adain: 目标域增强模块（TAM）
        teacher_alpha: 教师模型 EMA 衰减率
        conf_thres: 伪标签置信度阈值
        iou_thres: NMS 的 IoU 阈值
        max_det: 每张图像最大检测数量
        SSM_alpha: SSM 动量系数（0 表示禁用）
        style_add_alpha: 风格迁移强度
        source_data: 源域数据集配置路径
        val_source: 是否在源域上验证
        test_loader_source: 源域验证数据加载器
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 SF-YOLO 训练器
        
        Args:
            cfg: 基础配置对象
            overrides: 配置覆盖参数
            _callbacks: 回调函数
        """
        # 在父类初始化前提取 SF-YOLO 特有参数
        self._extract_sf_yolo_args(overrides or {})
        
        super().__init__(cfg, overrides, _callbacks)
        
        # 初始化 TAM 模块（后续在 setup 中完成）
        self.adain = None
        
    def _extract_sf_yolo_args(self, overrides):
        """从 overrides 中提取 SF-YOLO 特有参数"""
        # 教师模型 EMA 参数
        self.teacher_alpha = overrides.pop('teacher_alpha', 0.999)
        self.conf_thres = overrides.pop('conf_thres', 0.4)
        self.iou_thres = overrides.pop('iou_thres', 0.3)
        self.max_det = overrides.pop('max_det', 20)
        self.SSM_alpha = overrides.pop('SSM_alpha', 0.0)
        
        # 源域验证数据集配置（用于监控域迁移性能）
        self.source_data = overrides.pop('source_data', None)
        self.val_source = overrides.pop('val_source', True)
        
        # TAM 模块参数
        self.decoder_path = overrides.pop('decoder_path', None)
        self.encoder_path = overrides.pop('encoder_path', None)
        self.fc1_path = overrides.pop('fc1', None)
        self.fc2_path = overrides.pop('fc2', None)
        self.style_path = overrides.pop('style_path', '')
        self.style_add_alpha = overrides.pop('style_add_alpha', 1.0)
        self.save_style_samples = overrides.pop('save_style_samples', False)
        self.random_style = self.style_path == ''
        
        # 存储图像路径供 TAM 使用
        self.imgs_paths = []
        
        # Test mode: 伪标签调试相关
        self.debug_mode = overrides.pop('debug_mode', False)
        self.debug_interval = overrides.pop('debug_interval', 50)
        self.debug_counter = 0
        self.debug_save_dir = Path(overrides.pop('debug_save_dir', './check/teacher_output'))
        if self.debug_mode:
            self.debug_save_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"{colorstr('SF-YOLO:')} Test mode 已启用，伪标签将保存到 {self.debug_save_dir}")
        
        
    def _setup_train(self):
        """设置训练环境，初始化双模型架构"""
        # 先调用父类的训练设置
        super()._setup_train()
        
        # 初始化目标域增强模块
        self._init_tam()
        
        # 创建教师模型（深拷贝学生模型）
        LOGGER.info(f"{colorstr('SF-YOLO:')} 正在创建教师模型...")
        self.model_teacher = deepcopy(self.model).eval()
        
        # 冻结教师模型参数（不参与梯度更新，只通过 EMA 更新）
        for param in self.model_teacher.parameters():
            param.requires_grad = False
            
        # 创建教师模型的 EMA 优化器
        # FIX: 使用所有参数，不要过滤 requires_grad（因为教师模型参数都是 False）
        teacher_params = list(self.model_teacher.parameters())
        student_params = list(self.model.parameters())
        self.optimizer_teacher = WeightEMA(teacher_params, student_params, alpha=self.teacher_alpha)
        
        LOGGER.info(f"{colorstr('SF-YOLO:')} 教师模型初始化完成，EMA alpha={self.teacher_alpha}")
        LOGGER.info(f"{colorstr('SF-YOLO:')} SSM_alpha={self.SSM_alpha}, conf_thres={self.conf_thres}, iou_thres={self.iou_thres}")
        
        # 初始化源域验证器（如果提供了源域数据配置）
        self._setup_source_validator()
        
    def _setup_source_validator(self):
        """设置源域验证器用于监控域迁移性能
        
        注意：验证器会在首次验证时延迟初始化，以确保使用正确的数据加载器
        """
        self.validator_source = None
        self.test_loader_source = None
        if self.source_data and self.val_source:
            LOGGER.info(f"{colorstr('SF-YOLO:')} 源域验证将在首次验证时初始化，数据: {self.source_data}")
        elif not self.source_data:
            LOGGER.info(f"{colorstr('SF-YOLO:')} 未提供源域数据配置，跳过源域验证")
        
    def _init_tam(self):
        """初始化目标域增强模块（TAM）"""
        if self.decoder_path and self.encoder_path and self.fc1_path and self.fc2_path:
            LOGGER.info(f"{colorstr('SF-YOLO:')} 正在初始化目标域增强模块（TAM）...")
            
            # 创建 TAM 参数对象
            class TAMArgs:
                pass
            
            tam_args = TAMArgs()
            tam_args.decoder_path = self.decoder_path
            tam_args.encoder_path = self.encoder_path
            tam_args.fc1 = self.fc1_path
            tam_args.fc2 = self.fc2_path
            tam_args.style_path = self.style_path
            tam_args.style_add_alpha = self.style_add_alpha
            tam_args.random_style = self.random_style
            tam_args.cuda = self.device.type != 'cpu'
            tam_args.imgsz = self.args.imgsz
            tam_args.log_dir = str(self.save_dir / 'enhance_style_samples')
            tam_args.save_style_samples = self.save_style_samples
            tam_args.imgs_paths = []
            
            self.tam_args = tam_args
            self.adain = enhance_vgg16(tam_args)
            LOGGER.info(f"{colorstr('SF-YOLO:')} TAM 初始化完成，风格迁移强度 style_add_alpha={self.style_add_alpha}")
        else:
            LOGGER.warning(f"{colorstr('SF-YOLO:')} 未提供 TAM 权重路径，风格增强功能已禁用")
    
    def validate(self):
        """同时在目标域和源域验证集上运行验证
        
        重写父类方法以：
        1. 在目标域验证集上验证（原有功能）
        2. 在源域验证集上验证（新增功能，用于监控域迁移性能）
        
        Returns:
            (tuple): 包含目标域指标和 fitness 的元组
        """
        from torch import distributed as dist
        
        # 同步 EMA 缓冲区（多GPU时）
        if self.ema and self.world_size > 1:
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        
        # ========== 目标域验证 ==========
        metrics_target = self.validator(self)
        fitness_target = None
        if metrics_target is not None:
            fitness_target = metrics_target.pop("fitness", -self.loss.detach().cpu().numpy())
            if not self.best_fitness or self.best_fitness < fitness_target:
                self.best_fitness = fitness_target
        
        # ========== 源域验证 ==========
        metrics_source = None
        fitness_source = None
        if self.source_data and self.val_source:
            try:
                # 保存原始配置
                original_data = self.args.data
                original_test_loader = self.test_loader
                
                # 切换到源域数据配置
                self.args.data = self.source_data
                
                # 创建源域验证数据加载器（如果尚未创建）
                if self.test_loader_source is None:
                    # 解析源域 YAML 配置文件
                    source_data_dict = YAML.load(self.source_data)
                    
                    # 检查 YAML 加载结果
                    if source_data_dict is None:
                        raise ValueError(f"无法加载源域 YAML 文件: {self.source_data}")
                    
                    # 获取源域验证集路径
                    if isinstance(source_data_dict.get('val'), list):
                        val_path = source_data_dict['val'][0]
                    else:
                        val_path = source_data_dict['val']
                    
                    # 如果路径是相对路径，基于 YAML 文件位置解析
                    path_base = source_data_dict.get('path', '')
                    if path_base:
                        val_path = str(Path(path_base) / val_path)
                    else:
                        val_path = str(Path(self.source_data).parent / val_path)
                    
                    LOGGER.info(f"{colorstr('SF-YOLO:')} 加载源域验证数据: {val_path}")
                    
                    # 构建源域验证数据集
                    dataset_source = YOLODataset(
                        img_path=val_path,
                        imgsz=self.args.imgsz,
                        batch_size=self.args.batch,
                        augment=False,
                        hyp=self.args,
                        rect=self.args.rect,
                        cache=self.args.cache or None,
                        single_cls=self.args.single_cls or False,
                        stride=self.model.stride.max().item() if hasattr(self.model, 'stride') else 32,
                        pad=0.5,
                        task=getattr(self, 'task', 'detect'),
                    )
                    
                    # 构建源域验证数据加载器
                    self.test_loader_source = build_dataloader(
                        dataset_source, 
                        self.args.batch, 
                        self.args.workers, 
                        shuffle=False, 
                        rank=RANK
                    )
                    LOGGER.info(f"{colorstr('SF-YOLO:')} 源域验证数据加载器创建完成，共 {len(dataset_source)} 张图像")
                
                # 切换到源域验证数据加载器
                self.test_loader = self.test_loader_source
                
                # 创建新的验证器（使用源域数据加载器）
                validator_source = detect.DetectionValidator(
                    self.test_loader_source, 
                    save_dir=self.save_dir, 
                    args=copy(self.args), 
                    _callbacks=self.callbacks
                )
                # 设置 loss_names（如果目标域验证器已设置）
                if hasattr(self.validator, 'loss_names') and self.validator.loss_names is not None:
                    validator_source.loss_names = self.validator.loss_names
                else:
                    validator_source.loss_names = ["box_loss", "cls_loss", "dfl_loss"]
                
                # 运行源域验证
                metrics_source = validator_source(self)
                
                # 恢复原始配置
                self.args.data = original_data
                self.test_loader = original_test_loader
                
                if metrics_source is not None:
                    fitness_source = metrics_source.pop("fitness", -self.loss.detach().cpu().numpy())
                    
            except Exception as e:
                LOGGER.warning(f"{colorstr('SF-YOLO:')} 源域验证失败: {e}")
                import traceback
                LOGGER.debug(traceback.format_exc())
                metrics_source = None
        
        # 存储源域指标供后续使用（如保存指标到CSV）
        self.metrics_source = metrics_source
        self.fitness_source = fitness_source
        
        # 记录验证结果
        if RANK in {-1, 0}:
            if fitness_target is not None:
                log_msg = f"{colorstr('SF-YOLO:')} 目标域验证 fitness={fitness_target:.4f}"
                if fitness_source is not None:
                    gap = fitness_target - fitness_source
                    log_msg += f", 源域 fitness={fitness_source:.4f}, 域差距={gap:+.4f}"
                LOGGER.info(log_msg)
        
        return metrics_target, fitness_target
    
    def save_metrics(self, metrics):
        """保存训练指标到 CSV 文件（包含源域和目标域）
        
        重写父类方法以同时保存源域和目标域的验证指标
        """
        # 合并目标域和源域指标
        all_metrics = dict(metrics)
        
        # 添加源域指标（如果有）
        if hasattr(self, 'metrics_source') and self.metrics_source is not None:
            for key, value in self.metrics_source.items():
                all_metrics[f"source_{key}"] = value
            if hasattr(self, 'fitness_source') and self.fitness_source is not None:
                all_metrics["source_fitness"] = self.fitness_source
        
        # 添加域差距指标
        if hasattr(self, 'fitness_source') and self.fitness_source is not None:
            target_fitness = all_metrics.get('fitness', 0)
            all_metrics["domain_gap"] = target_fitness - self.fitness_source
        
        keys, vals = list(all_metrics.keys()), list(all_metrics.values())
        n = len(all_metrics) + 2  # number of cols (epoch, time, metrics...)
        t = time.time() - self.train_time_start
        
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入表头（如果是新文件）
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        
        # 写入数据
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",") + "\n")
    
    def save_model(self):
        """保存学生模型和教师模型的训练检查点
        
        重写父类方法以同时保存：
        1. 学生模型 (last.pt, best.pt)
        2. 教师模型 (last_teacher.pt, best_teacher.pt)
        """
        import io
        from ultralytics.utils.torch_utils import unwrap_model
        from ultralytics.utils import __version__
        from datetime import datetime
        
        # 确保权重目录存在
        self.wdir.mkdir(parents=True, exist_ok=True)
        
        # ========== 保存学生模型 (与父类逻辑相同) ==========
        buffer_student = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(unwrap_model(self.ema.ema)).half(),
                "updates": self.ema.updates,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "date": datetime.now().isoformat(),
                "version": __version__,
            },
            buffer_student,
        )
        serialized_student = buffer_student.getvalue()
        
        # 保存学生模型检查点
        self.last.write_bytes(serialized_student)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_student)
            LOGGER.info(f"{colorstr('SF-YOLO:')} 保存最佳学生模型到 {self.best}")
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_student)
        
        # ========== 保存教师模型 ==========
        if hasattr(self, 'model_teacher') and self.model_teacher is not None:
            buffer_teacher = io.BytesIO()
            
            # 获取教师模型的状态
            teacher_model = unwrap_model(self.model_teacher)
            
            # 在验证集上评估教师模型以获取其性能
            # 暂时切换到教师模型进行验证
            original_model = self.model
            self.model = self.model_teacher
            try:
                teacher_metrics, teacher_fitness = self.validate()
            except Exception as e:
                LOGGER.warning(f"{colorstr('SF-YOLO:')} 教师模型验证失败: {e}")
                teacher_metrics = {}
                teacher_fitness = 0.0
            finally:
                self.model = original_model
            
            # 更新教师模型的最佳性能
            if not hasattr(self, 'best_fitness_teacher'):
                self.best_fitness_teacher = 0.0
            if teacher_fitness is not None and teacher_fitness > self.best_fitness_teacher:
                self.best_fitness_teacher = teacher_fitness
                is_best_teacher = True
            else:
                is_best_teacher = False
            
            torch.save(
                {
                    "epoch": self.epoch,
                    "best_fitness": self.best_fitness_teacher,
                    "model": deepcopy(teacher_model).half(),
                    "ema": None,  # 教师模型不使用 EMA
                    "updates": self.ema.updates,
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "train_args": vars(self.args),
                    "train_metrics": {**(teacher_metrics or {}), **{"fitness": teacher_fitness or 0.0}},
                    "date": datetime.now().isoformat(),
                    "version": __version__,
                    "model_type": "teacher",  # 标记为教师模型
                },
                buffer_teacher,
            )
            serialized_teacher = buffer_teacher.getvalue()
            
            # 保存教师模型检查点
            last_teacher_path = self.wdir / "last_teacher.pt"
            best_teacher_path = self.wdir / "best_teacher.pt"
            
            last_teacher_path.write_bytes(serialized_teacher)
            if is_best_teacher:
                best_teacher_path.write_bytes(serialized_teacher)
                LOGGER.info(f"{colorstr('SF-YOLO:')} 保存最佳教师模型到 {best_teacher_path} "
                           f"(fitness={teacher_fitness:.4f})")
            
            if (self.save_period > 0) and (self.epoch % self.save_period == 0):
                (self.wdir / f"epoch{self.epoch}_teacher.pt").write_bytes(serialized_teacher)
    
    def _apply_style_augmentation(self, imgs):
        """使用 TAM 对图像进行风格增强
        
        Args:
            imgs: 图像张量，形状为 (B, C, H, W)，像素值范围 [0, 1]
            
        Returns:
            风格化后的图像张量
        """
        assert self.adain is not None
                    
        # 将像素值从 [0, 1] 转换到 [0, 255] 范围（TAM 需要）
        imgs_255 = imgs * 255.0
        
        # 应用风格迁移
        styled_imgs = get_style_images(imgs_255, self.tam_args, self.adain) / 255.0
        
        return styled_imgs
    
    def _generate_pseudo_labels(self, preds):
        """从教师模型预测中生成伪标签
        
        Args:
            preds: 教师模型的原始预测输出
            
        Returns:
            每张图像的伪标签列表
        """
        # 应用非极大值抑制（NMS）获取最终检测结果
        pred_nms = non_max_suppression(
            preds,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.max_det,
        )
        
        return pred_nms
    
    def _save_student_output(self, imgs_style, preds_student, pseudo_labels, batch_idx):
        """保存学生模型的输出可视化（与教师伪标签对比）
        
        使用与 Ultralytics Detect 头相同的解码逻辑：
        1. 训练时使用 one2many 头的输出
        2. 通过 dfl() 解码 Distribution Focal Loss
        3. 使用 decode_bboxes() 转换为 xyxy 坐标
        4. 应用 sigmoid 到 scores
        
        Args:
            imgs_style: 风格化图像张量 (B, C, H, W)，范围 [0, 1]
            preds_student: 学生模型的原始预测输出（训练模式下的字典格式）
            pseudo_labels: 教师模型生成的伪标签（用于对比）
            batch_idx: 当前batch索引
        """
        if not self.debug_mode:
            return
        
        student_save_dir = Path(self.debug_save_dir) / "student_output"
        student_save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with torch.no_grad():
                # 获取模型头部信息
                model_head = self.model.model[-1] if hasattr(self.model, 'model') else self.model.module.model[-1]
                
                # 处理 end2end 模型的输出格式
                # YOLOv26 训练时返回 {'one2many': {...}, 'one2one': {...}}
                # 我们使用 one2many 头（训练头）的输出来与学生模型的训练状态一致
                if isinstance(preds_student, dict):
                    if "one2many" in preds_student:
                        preds_student = preds_student["one2many"]
                
                # 使用 Detect 头的内部方法解码预测
                # 参考: ultralytics/nn/modules/head.py 中的 _inference() 和 _get_decode_boxes()
                
                # 1. 创建锚点 (anchors) 和步长 (strides)
                # 参考: ultralytics/nn/modules/head.py 中的 _get_decode_boxes()
                from ultralytics.utils.tal import make_anchors
                feats = preds_student["feats"]
                stride = model_head.stride
                anchors, strides = make_anchors(feats, stride, 0.5)
                # anchors: [num_anchors, 2], strides: [num_anchors, 1]
                # 需要转置 anchors 以匹配 boxes 的维度 [batch, 4, num_anchors]
                anchors = anchors.t().unsqueeze(0)  # [1, 2, num_anchors]
                strides = strides.view(1, 1, -1)  # [1, 1, num_anchors]
                
                # 2. 解码边界框
                # boxes: [batch, 4*reg_max, num_anchors]
                pred_distri = preds_student["boxes"]  # [batch, 4*reg_max, num_anchors]
                
                # 应用 DFL 解码 (Distribution Focal Loss)，当 reg_max > 1 时
                if model_head.reg_max > 1:
                    pred_bboxes = model_head.decode_bboxes(model_head.dfl(pred_distri), anchors)
                else:
                    # reg_max == 1 时，boxes 直接是 [batch, 4, num_anchors]，跳过 DFL
                    pred_bboxes = model_head.decode_bboxes(pred_distri, anchors)
                
                # 乘以 strides 得到最终坐标
                pred_bboxes = pred_bboxes * strides  # [batch, num_anchors, 4]
                # pred_bboxes: [batch, num_anchors, 4] in xyxy format
                
                # 3. 对 scores 应用 sigmoid
                # scores: [batch, nc, num_anchors]
                pred_scores = preds_student["scores"].sigmoid()  # [batch, nc, num_anchors]
                
                # 4. 合并预测结果 [batch, 4+nc, num_anchors]
                # 注意：NMS 期望的输入格式是 [batch, 4+nc, num_anchors]
                # pred_bboxes: [batch, 4, num_anchors]
                # pred_scores: [batch, nc, num_anchors]
                preds_combined = torch.cat([pred_bboxes, pred_scores], dim=1)  # [batch, 4+nc, num_anchors]
                
                # 5. 应用 NMS
                student_preds = non_max_suppression(
                    preds_combined,
                    conf_thres=self.conf_thres,
                    iou_thres=self.iou_thres,
                    multi_label=True,
                    agnostic=self.args.single_cls,
                    max_det=self.max_det,
                )
        except Exception as e:
            LOGGER.warning(f"解码学生模型输出失败: {e}")
            import traceback
            LOGGER.warning(traceback.format_exc())
            return
        
        # 只保存前2张图像
        num_save = min(2, imgs_style.shape[0])
        
        for i in range(num_save):
            # 转换图像为numpy格式
            img_style = imgs_style[i].cpu().permute(1, 2, 0).numpy() * 255
            img_style = img_style.astype(np.uint8).copy()
            img_style = cv2.cvtColor(img_style, cv2.COLOR_RGB2BGR)
            
            h, w = img_style.shape[:2]
            
            # 创建对比图像：左列=教师伪标签，右列=学生预测
            img_teacher = img_style.copy()
            img_student = img_style.copy()
            
            # 绘制教师伪标签（红色）
            teacher_labels = pseudo_labels[i]
            teacher_count = 0
            if teacher_labels is not None and len(teacher_labels) > 0:
                teacher_count = len(teacher_labels)
                for det in teacher_labels:
                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()[:6]
                    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                    color = (0, 0, 255)  # 红色 - 教师
                    cv2.rectangle(img_teacher, (x1, y1), (x2, y2), color, 2)
                    label_text = f"T:{int(cls)} {conf:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img_teacher, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
                    cv2.putText(img_teacher, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(img_teacher, "NO TEACHER LABELS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 绘制学生预测（蓝色）
            student_labels = student_preds[i]
            student_count = 0
            if student_labels is not None and len(student_labels) > 0:
                student_count = len(student_labels)
                for det in student_labels:
                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()[:6]
                    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                    color = (255, 0, 0)  # 蓝色 - 学生
                    cv2.rectangle(img_student, (x1, y1), (x2, y2), color, 2)
                    label_text = f"S:{int(cls)} {conf:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img_student, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
                    cv2.putText(img_student, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(img_student, "NO STUDENT PRED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # 拼接对比图像
            combined = np.hstack([img_style, img_teacher, img_student])
            
            # 添加标题行
            title_h = 40
            title_bar = np.zeros((title_h, combined.shape[1], 3), dtype=np.uint8)
            section_w = combined.shape[1] // 3
            titles = [
                f"Style Input",
                f"Teacher Pseudo Labels ({teacher_count} objs)",
                f"Student Prediction ({student_count} objs)"
            ]
            colors = [(255, 255, 255), (0, 0, 255), (255, 0, 0)]  # 白、红、蓝
            for idx, (title, color) in enumerate(zip(titles, colors)):
                cv2.putText(title_bar, title, (idx * section_w + 10, 28), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            combined = np.vstack([title_bar, combined])
            
            # 保存图像
            save_path = student_save_dir / f"compare_epoch{self.epoch}_batch{batch_idx}_img{i}.jpg"
            cv2.imwrite(str(save_path), combined)
            
            # 保存文本对比信息
            txt_path = student_save_dir / f"compare_epoch{self.epoch}_batch{batch_idx}_img{i}.txt"
            with open(txt_path, 'w') as f:
                f.write(f"Epoch: {self.epoch}, Batch: {batch_idx}, Image: {i}\n")
                f.write(f"conf_thres: {self.conf_thres}, iou_thres: {self.iou_thres}\n\n")
                f.write(f"Teacher Pseudo Labels ({teacher_count}):\n")
                if teacher_labels is not None and len(teacher_labels) > 0:
                    for det in teacher_labels:
                        x1, y1, x2, y2, conf, cls = det.cpu().numpy()[:6]
                        f.write(f"  cls:{int(cls)} conf:{conf:.3f} box:[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]\n")
                else:
                    f.write("  None\n")
                
                f.write(f"\nStudent Predictions ({student_count}):\n")
                if student_labels is not None and len(student_labels) > 0:
                    for det in student_labels:
                        x1, y1, x2, y2, conf, cls = det.cpu().numpy()[:6]
                        f.write(f"  cls:{int(cls)} conf:{conf:.3f} box:[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]\n")
                else:
                    f.write("  None\n")
        
        LOGGER.info(f"{colorstr('SF-YOLO:')} 已保存学生模型对比图像到 {student_save_dir}")

    def _save_teacher_output(self, imgs, imgs_style, pseudo_labels, batch_idx, prefix="debug"):
        """保存调试可视化图像（原始图像、风格化图像、伪标签）
        
        Args:
            imgs: 原始图像张量 (B, C, H, W)，范围 [0, 1]
            imgs_style: 风格化图像张量 (B, C, H, W)，范围 [0, 1]
            pseudo_labels: 伪标签列表
            batch_idx: 当前batch索引
            prefix: 文件名前缀
        """
        if not self.debug_mode:
            return
        
        # 只保存前2张图像
        num_save = min(2, imgs.shape[0])
        
        for i in range(num_save):
            # 转换图像为numpy格式 (H, W, C)，范围 [0, 255]
            img_orig = imgs[i].cpu().permute(1, 2, 0).numpy() * 255
            img_orig = img_orig.astype(np.uint8).copy()
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
            
            img_style = imgs_style[i].cpu().permute(1, 2, 0).numpy() * 255
            img_style = img_style.astype(np.uint8).copy()
            img_style = cv2.cvtColor(img_style, cv2.COLOR_RGB2BGR)
            
            h, w = img_orig.shape[:2]
            
            # 绘制伪标签到原始图像
            img_with_labels = img_orig.copy()
            labels = pseudo_labels[i]
            
            if labels is not None and len(labels) > 0:
                for det in labels:
                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()[:6]
                    # 确保坐标在图像范围内
                    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                    
                    # 绘制边界框
                    color = (0, 255, 0)  # 绿色
                    thickness = 2
                    cv2.rectangle(img_with_labels, (x1, y1), (x2, y2), color, thickness)
                    
                    # 绘制标签文本
                    label_text = f"cls:{int(cls)} conf:{conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    text_color = (0, 255, 0)
                    text_thickness = 1
                    
                    # 获取文本尺寸
                    (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
                    
                    # 绘制文本背景
                    cv2.rectangle(img_with_labels, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
                    # 绘制文本
                    cv2.putText(img_with_labels, label_text, (x1, y1 - 2), font, font_scale, (0, 0, 0), text_thickness)
            else:
                # 没有检测到目标，在图像上标注
                cv2.putText(img_with_labels, "NO DETECTIONS", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 拼接原始图像、风格化图像和带伪标签的图像
            combined = np.hstack([img_orig, img_style, img_with_labels])
            
            # 添加标题行
            title_h = 30
            title_bar = np.zeros((title_h, combined.shape[1], 3), dtype=np.uint8)
            titles = ["Original", "Style Augmented", f"Pseudo Labels ({len(labels) if labels is not None else 0} objs)"]
            section_w = combined.shape[1] // 3
            for idx, title in enumerate(titles):
                cv2.putText(title_bar, title, (idx * section_w + 10, 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            combined = np.vstack([title_bar, combined])
            
            # 保存图像
            teacher_save_dir = Path(self.debug_save_dir) / "teacher_output"
            teacher_save_dir.mkdir(parents=True, exist_ok=True)
            save_path = teacher_save_dir / f"{prefix}_epoch{self.epoch}_batch{batch_idx}_img{i}.jpg"
            cv2.imwrite(str(save_path), combined)
            
            # 同时保存伪标签的文本信息
            txt_path = teacher_save_dir / f"{prefix}_epoch{self.epoch}_batch{batch_idx}_img{i}.txt"
            with open(txt_path, 'w') as f:
                f.write(f"Epoch: {self.epoch}, Batch: {batch_idx}, Image: {i}\n")
                f.write(f"conf_thres: {self.conf_thres}, iou_thres: {self.iou_thres}\n")
                f.write(f"Num detections: {len(labels) if labels is not None else 0}\n\n")
                if labels is not None and len(labels) > 0:
                    f.write("Format: [x1, y1, x2, y2, conf, cls]\n")
                    for det in labels:
                        x1, y1, x2, y2, conf, cls = det.cpu().numpy()[:6]
                        f.write(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {conf:.4f} {int(cls)}\n")
        
        LOGGER.info(f"{colorstr('SF-YOLO:')} 已保存调试图像到 {teacher_save_dir}")
    
    def _convert_pseudo_labels_to_targets(self, pseudo_labels, batch_size, img_height, img_width):
        """将伪标签转换为损失计算所需的格式
        
        Args:
            pseudo_labels: 每张图像的伪标签列表
            batch_size: 批次大小
            img_height: 图像高度
            img_width: 图像宽度
            
        Returns:
            包含类别、边界框和批次索引的元组 (cls_tensor, bboxes_tensor, batch_idx_tensor)
        """
        device = self.device
        cls_list = []
        bboxes_list = []
        batch_idx_list = []
        
        for image_id in range(batch_size):
            if pseudo_labels[image_id].shape[0] > 0:
                # pseudo_labels 格式: [x1, y1, x2, y2, conf, cls, ...]
                # 提取类别和边界框
                cls_data = pseudo_labels[image_id][:, 5:6]  # (N, 1)
                boxes_xyxy = pseudo_labels[image_id][:, :4]  # (N, 4)
                
                # 将 xyxy 格式转换为 xywh 格式（归一化）
                boxes_xywh = ops.xyxy2xywhn(boxes_xyxy, w=img_width, h=img_height)
                
                cls_list.append(cls_data)
                bboxes_list.append(boxes_xywh)
                batch_idx_list.append(torch.full((cls_data.shape[0],), image_id, device=device, dtype=torch.float32))
        
        if cls_list:
            cls_tensor = torch.cat(cls_list, dim=0)
            bboxes_tensor = torch.cat(bboxes_list, dim=0)
            batch_idx_tensor = torch.cat(batch_idx_list, dim=0).unsqueeze(1)
            return cls_tensor, bboxes_tensor, batch_idx_tensor
        else:
            # 如果没有检测到目标，返回虚拟标签（避免训练崩溃）
            cls_tensor = torch.tensor([[0]], device=device, dtype=torch.float32)
            bboxes_tensor = torch.tensor([[0.5, 0.7, 0.3, 0.3]], device=device, dtype=torch.float32)
            batch_idx_tensor = torch.tensor([[0]], device=device, dtype=torch.float32)
            return cls_tensor, bboxes_tensor, batch_idx_tensor
    
    def _ssm_update(self):
        """应用稳定学生动量（SSM）更新
        
        在每个 epoch 开始时，将教师模型的权重按一定比例转移到学生模型
        """
        if self.SSM_alpha > 0:
            student_state_dict = self.model.state_dict()
            teacher_state_dict = self.model_teacher.state_dict()
            
            # 执行 SSM 更新（只更新浮点类型参数）
            update_count = 0
            for name, param in student_state_dict.items():
                if name in teacher_state_dict:
                    # 跳过整数类型参数（如 batch norm 的 num_batches_tracked）
                    if param.dtype in [torch.long, torch.int]:
                        continue
                    param.data.copy_(
                        (1.0 - self.SSM_alpha) * param.data + 
                        self.SSM_alpha * teacher_state_dict[name].data
                    )
                    update_count += 1
            
            LOGGER.info(f"{colorstr('SF-YOLO:')} 已在第 {self.epoch} 轮应用 SSM 更新，alpha={self.SSM_alpha}，更新了 {update_count} 个参数")
    
    def _do_train(self):
        """SF-YOLO 自定义训练循环
        
        重写父类训练循环以实现：
        1. 风格增强
        2. 教师模型生成伪标签
        3. 学生模型在风格化图像上训练
        4. 教师模型 EMA 更新
        """
        if self.world_size > 1:
            self._setup_ddp()
            
        self._setup_train()
        
        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        
        LOGGER.info(
            f"图像尺寸 {self.args.imgsz} 训练, {self.args.imgsz} 验证\n"
            f"使用 {self.train_loader.num_workers * (self.world_size or 1)} 个数据加载工作进程\n"
            f"日志保存到 {colorstr('bold', self.save_dir)}\n"
            f"开始 SF-YOLO 训练，共 {self.epochs} 轮..."
        )
        
        # 关闭 mosaic 增强（如果指定）
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
            
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        self._oom_retries = 0
        
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            
            # 每轮开始时应用 SSM 更新（第一轮除外）
            if self.SSM_alpha > 0 and epoch > self.start_epoch:
                self._ssm_update()
                
            self._model_train()
            
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
                
            pbar = enumerate(self.train_loader)
            
            # 更新数据加载器属性
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
                
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                from ultralytics.utils import TQDM
                pbar = TQDM(enumerate(self.train_loader), total=nb)
                
            self.tloss = None
            
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                
                # 预热阶段学习率调整
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                # SF-YOLO 训练步骤
                with autocast(self.amp):
                    # 预处理批次数据
                    imgs = self.preprocess_batch(batch)["img"]  # SF-YOLO 只使用图像，不使用真实标签
                    batch = dict(img=imgs)
                    batch_size, _, img_height, img_width = imgs.shape
                    
                    # 应用风格增强
                    imgs_style = self._apply_style_augmentation(imgs)
                    
                    # 教师模型在前向传播（用于生成伪标签）
                    self.model_teacher.eval()
                    with torch.no_grad():
                        preds_teacher = self.model_teacher(imgs)
                        if isinstance(preds_teacher, (list, tuple)):
                            preds_teacher = preds_teacher[0]
                    
                    # 生成伪标签
                    pseudo_labels = self._generate_pseudo_labels(preds_teacher)
            
                    # 将伪标签转换为训练目标格式
                    pseudo_cls, pseudo_bboxes, pseudo_batch_idx = self._convert_pseudo_labels_to_targets(
                        pseudo_labels, batch_size, img_height, img_width
                    )
                    
                    # 更新批次数据，使用伪标签
                    batch["cls"] = pseudo_cls
                    batch["bboxes"] = pseudo_bboxes
                    batch["batch_idx"] = pseudo_batch_idx
                    
                    # 学生模型在风格化图像上前向传播
                    preds_student = self.model(imgs_style)
                    
                    # Test mode: 保存伪标签和学生模型输出可视化
                    if self.debug_mode and self.debug_counter % self.debug_interval == 0:
                        self._save_teacher_output(imgs, imgs_style, pseudo_labels, i, prefix="pseudo")
                        # 同时保存学生模型的输出（与教师伪标签对比）
                        self._save_student_output(imgs_style, preds_student, pseudo_labels, i)
                    self.debug_counter += 1
                    
                    # 使用伪标签计算损失
                    if self.args.compile:
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds_student)
                    else:
                        loss, self.loss_items = self.model(batch, preds_student)
                        
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= self.world_size
                        
                    self.tloss = (
                        self.loss_items if self.tloss is None 
                        else (self.tloss * i + self.loss_items) / (i + 1)
                    )
                
                # 反向传播
                self.scaler.scale(self.loss).backward()
                    
                # 优化步骤
                if ni - last_opt_step >= self.accumulate:
                    last_opt_step = ni
                    # 优化学生模型
                    self.optimizer_step()
                    
                    # 使用 EMA 更新教师模型
                    self.model_teacher.train()
                    self.model_teacher.zero_grad()
                    self.optimizer_teacher.step()
                    
                    # 定时停止检查
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            from torch import distributed as dist
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break
                    
                # 日志记录
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                
                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break
            else:
                # 循环正常完成（未触发 break）
                self._oom_retries = 0
            
            if self._oom_retries and not self.stop:
                continue
            
            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()
            
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            
            self.scheduler.step()
            
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
            
            # 验证
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()
            
            # NaN 恢复
            if self._handle_nan_recovery(epoch):
                continue
            
            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
                
                # 保存模型
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")
            
            # 学习率调度
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)
            
            # 早停
            if RANK != -1:
                from torch import distributed as dist
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            
            if self.stop:
                break
            epoch += 1
        
        # 训练完成
        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n训练完成，共 {epoch - self.start_epoch + 1} 轮，耗时 {seconds / 3600:.3f} 小时。")
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        from ultralytics.utils.torch_utils import unset_deterministic
        unset_deterministic()
        self.run_callbacks("teardown")


def parse_args():
    """解析 SF-YOLO 训练的命令行参数"""
    parser = argparse.ArgumentParser(description="SF-YOLO 训练脚本（YOLOv26）")
    
    # 模型和数据参数
    parser.add_argument("--weights", type=str, default="yolo26n.pt", help="初始权重路径")
    parser.add_argument("--data", type=str, required=True, help="目标域数据集 YAML 文件路径")
    parser.add_argument("--source_data", type=str, default=None, help="源域数据集 YAML 文件路径（用于监控域迁移性能）")
    parser.add_argument("--val_source", action="store_true", default=True, help="在源域验证集上进行验证（监控域迁移性能）")
    parser.add_argument("--epochs", type=int, default=60, help="训练总轮数")
    parser.add_argument("--imgsz", type=int, default=960, help="图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--device", type=str, default="0", help="计算设备（cuda 或 cpu）")
    parser.add_argument("--workers", type=int, default=8, help="数据加载工作进程数")
    parser.add_argument("--project", type=str, default="sf-yolo", help="项目目录")
    parser.add_argument("--name", type=str, default="exp", help="实验名称")
    parser.add_argument("--exist-ok", action="store_true", help="覆盖已存在的实验目录")
    
    # SF-YOLO 特有参数
    parser.add_argument("--teacher_alpha", type=float, default=0.999, help="教师模型 EMA alpha")
    parser.add_argument("--conf_thres", type=float, default=0.4, help="伪标签置信度阈值")
    parser.add_argument("--iou_thres", type=float, default=0.3, help="NMS IoU 阈值")
    parser.add_argument("--max_det", type=int, default=20, help="每张图像最大检测数")
    parser.add_argument("--SSM_alpha", type=float, default=0.0, help="SSM 动量系数（0 表示禁用）")
    
    # TAM 参数
    parser.add_argument("--decoder_path", type=str, required=True, help="解码器权重路径")
    parser.add_argument("--encoder_path", type=str, required=True, help="编码器（VGG）权重路径")
    parser.add_argument("--fc1", type=str, required=True, help="FC1 权重路径")
    parser.add_argument("--fc2", type=str, required=True, help="FC2 权重路径")
    parser.add_argument("--style_path", type=str, default="", help="风格图像路径（空表示使用随机风格）")
    parser.add_argument("--style_add_alpha", type=float, default=1.0, help="风格迁移强度")
    parser.add_argument("--save_style_samples", action="store_true", help="保存风格样本图像")
    
    # 训练参数
    parser.add_argument("--cache", type=str, default="disk", help="数据集缓存模式: 'ram'/'disk'/False, 默认 'disk'")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率系数")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD 动量")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="权重衰减")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="预热轮数")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--cos_lr", action="store_true", help="使用余弦学习率调度")
    parser.add_argument("--amp", action="store_true", default=True, help="使用自动混合精度")
    parser.add_argument("--freeze", type=int, nargs="+", default=[0], help="冻结层索引")
    
    # Test mode 参数
    parser.add_argument("--debug_mode", action="store_true", help="启用调试模式，保存伪标签可视化")
    parser.add_argument("--debug_interval", type=int, default=50, help="每隔N个batch保存一次调试图像")
    parser.add_argument("--debug_save_dir", type=str, default="./check", help="调试图像保存目录")

    return parser.parse_args()


def main():
    
    torch.multiprocessing.set_start_method('spawn')
    
    """SF-YOLO 训练主函数"""
    args = parse_args()
    
    # 验证 TAM 权重路径
    if args.decoder_path and not os.path.exists(args.decoder_path):
        LOGGER.error(f"解码器路径不存在: {args.decoder_path}")
        sys.exit(1)
    if args.encoder_path and not os.path.exists(args.encoder_path):
        LOGGER.error(f"编码器路径不存在: {args.encoder_path}")
        sys.exit(1)
    if args.fc1 and not os.path.exists(args.fc1):
        LOGGER.error(f"FC1 路径不存在: {args.fc1}")
        sys.exit(1)
    if args.fc2 and not os.path.exists(args.fc2):
        LOGGER.error(f"FC2 路径不存在: {args.fc2}")
        sys.exit(1)
    if args.style_path and not os.path.exists(args.style_path):
        LOGGER.warning(f"风格图像路径不存在: {args.style_path}，将使用随机风格")
        args.style_path = ""
    
    # 验证源域数据路径（如果提供）
    if args.source_data and not os.path.exists(args.source_data):
        LOGGER.warning(f"源域数据路径不存在: {args.source_data}，将跳过源域验证")
        args.source_data = None
        args.val_source = False
    
    # 准备训练器参数
    overrides = {
        # SF-YOLO 特有参数
        'teacher_alpha': args.teacher_alpha,
        'conf_thres': args.conf_thres,
        'iou_thres': args.iou_thres,
        'max_det': args.max_det,
        'SSM_alpha': args.SSM_alpha,
        'decoder_path': args.decoder_path,
        'encoder_path': args.encoder_path,
        'fc1': args.fc1,
        'fc2': args.fc2,
        'style_path': args.style_path,
        'style_add_alpha': args.style_add_alpha,
        'save_style_samples': args.save_style_samples,
        
        # 源域验证参数
        'source_data': args.source_data,
        'val_source': args.val_source,
        
        # Test mode 参数
        'debug_mode': args.debug_mode,
        'debug_interval': args.debug_interval,
        'debug_save_dir': args.debug_save_dir,
        
        # 标准训练参数
        'model': args.weights,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'seed': args.seed,
        'cos_lr': args.cos_lr,
        'amp': args.amp,
        'freeze': args.freeze,
        'cache': args.cache
    }
    
    # 创建训练器并开始训练
    trainer = SFYOLOTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
