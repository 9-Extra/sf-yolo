#!/usr/bin/env python3
"""
SF-YOLO Training Script for YOLOv26 (Ultralytics)

This script implements the Source-Free Domain Adaptation method SF-YOLO 
for YOLOv26 models using Ultralytics' custom trainer API.

Reference: https://docs.ultralytics.com/zh/guides/custom-trainer/
"""

import argparse
import math
import os
import random
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.train import DetectionModel
from ultralytics.utils import LOGGER, RANK, colorstr, ops, DEFAULT_CFG
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model

# SF-YOLO specific imports
from TargetAugment.enhance_style import get_style_images
from TargetAugment.enhance_vgg16 import enhance_vgg16


class WeightEMA:
    """Unbiased Mean Teacher EMA optimizer.
    
    Reference: github.com/kinredon/umt
    """
    
    def __init__(self, teacher_params, student_params, alpha=0.999):
        self.teacher_params = list(teacher_params)
        self.student_params = list(student_params)
        self.alpha = alpha
        
    def step(self):
        """Apply EMA update from student to teacher."""
        for teacher_param, student_param in zip(self.teacher_params, self.student_params):
            if teacher_param.requires_grad:
                teacher_param.data.mul_(self.alpha).add_(student_param.data, alpha=1 - self.alpha)


class SFYOLOTrainer(DetectionTrainer):
    """Custom trainer for SF-YOLO (Source-Free Domain Adaptation).
    
    This trainer implements:
    1. Dual model architecture (Student + Teacher with EMA)
    2. Target Augmentation Module (TAM) for style transfer
    3. Pseudo-label generation by Teacher model
    4. Student training on stylized images with pseudo-labels
    5. Optional SSM (Stable Student Momentum)
    
    Attributes:
        model_teacher: Teacher model for pseudo-label generation
        optimizer_teacher: EMA optimizer for teacher model
        adain: Target Augmentation Module (TAM)
        teacher_alpha: EMA decay rate for teacher
        conf_thres: Confidence threshold for pseudo-labels
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections per image
        SSM_alpha: SSM momentum (0 to disable)
        style_add_alpha: Style transfer intensity
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize SF-YOLO trainer.
        
        Args:
            cfg: Configuration
            overrides: Configuration overrides
            _callbacks: Callbacks
        """
        # Extract SF-YOLO specific args before parent init
        self._extract_sf_yolo_args(overrides or {})
        
        super().__init__(cfg, overrides, _callbacks)
        
        # Initialize TAM module
        self.adain = None
        
    def _extract_sf_yolo_args(self, overrides):
        """Extract SF-YOLO specific arguments from overrides."""
        # Teacher EMA parameters
        self.teacher_alpha = overrides.pop('teacher_alpha', 0.999)
        self.conf_thres = overrides.pop('conf_thres', 0.4)
        self.iou_thres = overrides.pop('iou_thres', 0.3)
        self.max_det = overrides.pop('max_det', 20)
        self.SSM_alpha = overrides.pop('SSM_alpha', 0.0)
        
        # TAM parameters
        self.decoder_path = overrides.pop('decoder_path', None)
        self.encoder_path = overrides.pop('encoder_path', None)
        self.fc1_path = overrides.pop('fc1', None)
        self.fc2_path = overrides.pop('fc2', None)
        self.style_path = overrides.pop('style_path', '')
        self.style_add_alpha = overrides.pop('style_add_alpha', 1.0)
        self.save_style_samples = overrides.pop('save_style_samples', False)
        self.random_style = self.style_path == ''
        
        # Test mode
        self.test_batches = overrides.pop('test_batches', 0)
        
        # Store paths for TAM
        self.imgs_paths = []
        
    def plot_training_samples(self, batch, ni):
        """Override to skip plotting when using pseudo labels.
        
        The default plot_training_samples expects ground truth labels,
        but SF-YOLO uses pseudo labels which may have different format.
        """
        # Skip plotting to avoid format mismatch errors
        pass
        
    def _setup_train(self):
        """Setup training with dual model architecture."""
        # Call parent setup first
        super()._setup_train()
        
        # Initialize Target Augmentation Module
        self._init_tam()
        
        # Create teacher model
        LOGGER.info(f"{colorstr('SF-YOLO:')} Creating teacher model...")
        self.model_teacher = deepcopy(self.model).eval()
        
        # Freeze teacher model parameters
        for param in self.model_teacher.parameters():
            param.requires_grad = False
            
        # Create teacher EMA optimizer
        teacher_params = [p for p in self.model_teacher.parameters() if p.requires_grad]
        student_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer_teacher = WeightEMA(teacher_params, student_params, alpha=self.teacher_alpha)
        
        LOGGER.info(f"{colorstr('SF-YOLO:')} Teacher model initialized with alpha={self.teacher_alpha}")
        LOGGER.info(f"{colorstr('SF-YOLO:')} SSM_alpha={self.SSM_alpha}, conf_thres={self.conf_thres}, iou_thres={self.iou_thres}")
        
    def _init_tam(self):
        """Initialize Target Augmentation Module."""
        if self.decoder_path and self.encoder_path and self.fc1_path and self.fc2_path:
            LOGGER.info(f"{colorstr('SF-YOLO:')} Initializing Target Augmentation Module...")
            
            # Create args object for TAM
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
            LOGGER.info(f"{colorstr('SF-YOLO:')} TAM initialized with style_add_alpha={self.style_add_alpha}")
        else:
            LOGGER.warning(f"{colorstr('SF-YOLO:')} TAM weights not provided, style augmentation disabled")
            
    def _apply_style_augmentation(self, imgs):
        """Apply style augmentation to images using TAM.
        
        Args:
            imgs: Tensor of images (B, C, H, W) in range [0, 1]
            
        Returns:
            Stylized images tensor
        """
        if self.adain is None:
            return imgs
            
        # Convert to 0-255 range for TAM
        imgs_255 = imgs * 255.0
        
        # Apply style transfer
        styled_imgs = get_style_images(imgs_255, self.tam_args, self.adain) / 255.0
        
        return styled_imgs
    
    def _generate_pseudo_labels(self, preds):
        """Generate pseudo labels from teacher predictions.
        
        Args:
            preds: Raw predictions from teacher model
            
        Returns:
            List of pseudo label tensors per image
        """
        # Apply NMS to get final detections
        pred_nms = non_max_suppression(
            preds,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.max_det,
        )
        
        return pred_nms
    
    def _convert_pseudo_labels_to_targets(self, pseudo_labels, batch_size, img_height, img_width):
        """Convert pseudo labels to target format for loss computation.
        
        Args:
            pseudo_labels: List of pseudo label tensors per image
            batch_size: Batch size
            img_height: Image height
            img_width: Image width
            
        Returns:
            Tuple of (cls_tensor, bboxes_tensor, batch_idx_tensor)
        """
        device = self.device
        cls_list = []
        bboxes_list = []
        batch_idx_list = []
        
        for image_id in range(batch_size):
            if pseudo_labels[image_id].shape[0] > 0:
                # pseudo_labels format: [x1, y1, x2, y2, conf, cls, ...]
                # Extract cls and boxes
                cls_data = pseudo_labels[image_id][:, 5:6]  # (N, 1)
                boxes_xyxy = pseudo_labels[image_id][:, :4]  # (N, 4)
                
                # Convert xyxy to xywh (normalized)
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
            # Return dummy label if no detections
            cls_tensor = torch.tensor([[0]], device=device, dtype=torch.float32)
            bboxes_tensor = torch.tensor([[0.5, 0.7, 0.3, 0.3]], device=device, dtype=torch.float32)
            batch_idx_tensor = torch.tensor([[0]], device=device, dtype=torch.float32)
            return cls_tensor, bboxes_tensor, batch_idx_tensor
    
    def _ssm_update(self):
        """Apply Stable Student Momentum update.
        
        Transfers teacher weights to student at the start of epoch.
        """
        if self.SSM_alpha > 0:
            student_state_dict = self.model.state_dict()
            teacher_state_dict = self.model_teacher.state_dict()
            
            for name, param in student_state_dict.items():
                if name in teacher_state_dict:
                    param.data.copy_(
                        (1.0 - self.SSM_alpha) * param.data + 
                        self.SSM_alpha * teacher_state_dict[name].data
                    )
            
            LOGGER.info(f"{colorstr('SF-YOLO:')} SSM update applied at epoch {self.epoch}, alpha={self.SSM_alpha}")
    
    def _do_train(self):
        """Custom training loop for SF-YOLO.
        
        Overrides parent _do_train to implement:
        1. Style augmentation
        2. Teacher pseudo-label generation
        3. Student training on stylized images
        4. Teacher EMA update
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
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting SF-YOLO training for {self.epochs} epochs..."
        )
        
        # Close mosaic if specified
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
            
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        self._oom_retries = 0
        
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            
            # SSM update at start of epoch (except first)
            if self.SSM_alpha > 0 and epoch > self.start_epoch:
                self._ssm_update()
            
            # Step scheduler
            with torch.no_grad():
                self.scheduler.step()
                
            self._model_train()
            
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
                
            pbar = enumerate(self.train_loader)
            
            # Update dataloader attributes
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
                
                # Warmup
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
                
                # SF-YOLO Training Step
                try:
                    from ultralytics.utils.torch_utils import autocast
                    
                    with autocast(self.amp):
                        # Preprocess batch
                        batch = self.preprocess_batch(batch)
                        imgs = batch["img"]  # Original images
                        batch_size, _, img_height, img_width = imgs.shape
                        
                        # Apply style augmentation
                        imgs_style = self._apply_style_augmentation(imgs)
                        
                        # Teacher forward on original images (for pseudo labels)
                        self.model_teacher.eval()
                        with torch.no_grad():
                            preds_teacher = self.model_teacher(imgs)
                            if isinstance(preds_teacher, (list, tuple)):
                                preds_teacher = preds_teacher[0]
                        
                        # Generate pseudo labels
                        pseudo_labels = self._generate_pseudo_labels(preds_teacher)
                        
                        # Convert pseudo labels to target format
                        pseudo_cls, pseudo_bboxes, pseudo_batch_idx = self._convert_pseudo_labels_to_targets(
                            pseudo_labels, batch_size, img_height, img_width
                        )
                        
                        # Update batch with pseudo targets
                        batch["cls"] = pseudo_cls
                        batch["bboxes"] = pseudo_bboxes
                        batch["batch_idx"] = pseudo_batch_idx
                        
                        # Student forward on stylized images
                        preds_student = self.model(imgs_style)
                        
                        # Compute loss with pseudo labels
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
                    
                    # Backward
                    self.scaler.scale(self.loss).backward()
                    
                except torch.cuda.OutOfMemoryError:
                    if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                        raise
                    self._oom_retries += 1
                    old_batch = self.batch_size
                    self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                    LOGGER.warning(
                        f"CUDA out of memory with batch={old_batch}. "
                        f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                    )
                    self._clear_memory()
                    self._build_train_pipeline()
                    self.scheduler.last_epoch = self.start_epoch - 1
                    nb = len(self.train_loader)
                    nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
                    last_opt_step = -1
                    self.optimizer.zero_grad()
                    break
                
                # Optimize
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    
                    # Update teacher model with EMA
                    self.model_teacher.train()
                    self.model_teacher.zero_grad()
                    self.optimizer_teacher.step()
                    
                    last_opt_step = ni
                    
                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            from torch import distributed as dist
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break
                
                # Log
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
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                
                # Test mode: stop after N batches
                if self.test_batches > 0 and i >= self.test_batches - 1:
                    LOGGER.info(f"{colorstr('SF-YOLO:')} Test mode - stopping after {self.test_batches} batches")
                    break
                
                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break
            else:
                # Loop completed without break
                self._oom_retries = 0
            
            if self._oom_retries and not self.stop:
                continue
            
            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()
            
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
            
            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()
            
            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue
            
            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
                
                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")
            
            # Scheduler
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
            
            # Early Stopping
            if RANK != -1:
                from torch import distributed as dist
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            
            if self.stop:
                break
            epoch += 1
        
        # Training complete
        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
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
    """Parse command line arguments for SF-YOLO training."""
    parser = argparse.ArgumentParser(description="SF-YOLO Training for YOLOv26")
    
    # Model and data arguments
    parser.add_argument("--weights", type=str, default="yolo26n.pt", help="Initial weights path")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=60, help="Total training epochs")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (cuda or cpu)")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--project", type=str, default="runs/sf-yolo", help="Project directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing experiment")
    
    # SF-YOLO specific arguments
    parser.add_argument("--teacher_alpha", type=float, default=0.999, help="Teacher EMA alpha")
    parser.add_argument("--conf_thres", type=float, default=0.4, help="Confidence threshold for pseudo labels")
    parser.add_argument("--iou_thres", type=float, default=0.3, help="IoU threshold for NMS")
    parser.add_argument("--max_det", type=int, default=20, help="Maximum detections per image")
    parser.add_argument("--SSM_alpha", type=float, default=0.0, help="SSM momentum (0 to disable)")
    
    # TAM arguments
    parser.add_argument("--decoder_path", type=str, required=True, help="Decoder path")
    parser.add_argument("--encoder_path", type=str, required=True, help="Encoder (VGG) path")
    parser.add_argument("--fc1", type=str, required=True, help="FC1 path")
    parser.add_argument("--fc2", type=str, required=True, help="FC2 path")
    parser.add_argument("--style_path", type=str, default="", help="Style image path (empty for random)")
    parser.add_argument("--style_add_alpha", type=float, default=1.0, help="Style transfer intensity")
    parser.add_argument("--save_style_samples", action="store_true", help="Save style sample images")
    
    # Training arguments
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cos_lr", action="store_true", help="Use cosine LR scheduler")
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP")
    parser.add_argument("--freeze", type=int, nargs="+", default=[0], help="Freeze layers")
    parser.add_argument("--test-batches", type=int, default=0, help="Test mode: stop after N batches (0 = disable)")
    
    return parser.parse_args()


def main():
    """Main function for SF-YOLO training."""
    args = parse_args()
    
    # Set environment variable for uv
    os.environ.setdefault("UV_PROJECT_ENVIRONMENT", "/home/featurize/venv")
    
    # Validate TAM paths
    if args.decoder_path and not os.path.exists(args.decoder_path):
        LOGGER.error(f"Decoder path not found: {args.decoder_path}")
        sys.exit(1)
    if args.encoder_path and not os.path.exists(args.encoder_path):
        LOGGER.error(f"Encoder path not found: {args.encoder_path}")
        sys.exit(1)
    if args.fc1 and not os.path.exists(args.fc1):
        LOGGER.error(f"FC1 path not found: {args.fc1}")
        sys.exit(1)
    if args.fc2 and not os.path.exists(args.fc2):
        LOGGER.error(f"FC2 path not found: {args.fc2}")
        sys.exit(1)
    if args.style_path and not os.path.exists(args.style_path):
        LOGGER.warning(f"Style path not found: {args.style_path}, using random style")
        args.style_path = ""
    
    # Fix project path - Ultralytics uses runs/{task}/{project}/{name} format
    # Remove 'runs/' prefix if present to avoid double nesting
    project = args.project
    if project.startswith('runs/'):
        project = project[5:]  # Remove 'runs/' prefix
    
    # Prepare overrides for trainer
    overrides = {
        # SF-YOLO specific
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
        
        # Standard training args
        'model': args.weights,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'project': project,
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
        'test_batches': args.test_batches,  # Will be extracted in _extract_sf_yolo_args
    }
    
    # Create trainer and start training
    trainer = SFYOLOTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
