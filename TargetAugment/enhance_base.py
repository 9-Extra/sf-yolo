# coding=utf-8
"""
风格迁移基类模块

提供基于自适应实例归一化(AdaIN)的风格迁移功能，支持：
- 多层级编码器-解码器结构
- 风格图像的随机选择或固定选择
- torch.compile 编译加速
- 协方差匹配(CORAL)等高级风格迁移技术

主要类:
    TAMWrapper: TAM 编译包装器，用于 torch.compile 整体编译
    enhance_base: 风格迁移基类，提供完整的风格迁移流程
"""

from PIL import Image
import numpy as np
# from model.utils.config import cfg
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import os
import random
from torchvision import transforms as transforms


class TAMWrapper(nn.Module):
    """TAM 编译包装器 - 用于 torch.compile
    
    将 TAM 的所有子模块（编码器、解码器、全连接层）封装为一个整体 nn.Module，
    以便使用 torch.compile 进行整体编译加速，提高推理速度。
    
    属性:
        encoders: 编码器模块列表 (nn.ModuleList)
        decoders: 解码器模块列表 (nn.ModuleList)
        fc1: 风格均值预测全连接层
        fc2: 风格标准差预测全连接层
        num: 编码器/解码器层数
    
    示例:
        >>> wrapper = TAMWrapper(encoders, decoders, fc1, fc2)
        >>> compiled = torch.compile(wrapper)
        >>> output = compiled(content, style, flag=0, alpha=1.0)
    """
    
    def __init__(self, encoders, decoders, fc1, fc2):
        """初始化 TAMWrapper
        
        Args:
            encoders: 编码器模块列表
            decoders: 解码器模块列表
            fc1: 风格均值预测全连接层
            fc2: 风格标准差预测全连接层
        """
        super().__init__()
        # 将模块列表转换为 ModuleList 以便正确注册参数
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.fc1 = fc1
        self.fc2 = fc2
        self.num = len(encoders)
    
    def forward(self, content: torch.Tensor, style, flag=0, alpha=1.0):
        """TAM 风格迁移的前向传播
        
        执行完整的多层级编码-解码流程，包括：
        1. 多层级编码：对内容图像和风格图像进行分层编码
        2. 自适应实例归一化：将内容特征与风格特征进行融合
        3. 多层级解码：将融合后的特征解码回图像空间
        
        Args:
            content: 内容图像 [N, C, H, W]，需要进行风格迁移的图像
            style: 风格图像 [1, C, H, W]，提供目标风格的图像
            flag: 编码器起始层索引，用于控制从哪一层开始编码
            alpha: 风格迁移强度，范围 [0, 1]，0表示保持原图，1表示完全迁移
            
        Returns:
            torch.Tensor: 风格化后的图像 [N, C, H, W]
        """
        # 编码阶段：从指定层开始，逐层提取特征
        for i in range(flag, self.num):
            content = self.encoders[i](content).float()
            style = self.encoders[i](style).float()
        
        # 自适应实例归一化：将风格特征注入内容特征
        feat = self._adaptive_instance_normalization(content, style)
        feat = feat * alpha + content * (1 - alpha)  # 混合原始内容和风格化特征
        
        # 解码阶段：将融合后的特征解码为图像
        for i in range(self.num - flag):
            feat = self.decoders[i](feat)
        
        return feat
    
    def _adaptive_instance_normalization(self, content_feat: torch.Tensor, style_feat: torch.Tensor):
        """自适应实例归一化 (Adaptive Instance Normalization)
        
        将内容特征的统计特性（均值和标准差）替换为风格特征的统计特性，
        同时通过全连接层学习更复杂的风格迁移映射。
        
        Args:
            content_feat: 内容特征 [N, C, H, W]
            style_feat: 风格特征 [1, C, H, W]
            
        Returns:
            torch.Tensor: 融合后的特征 [N, C, H, W]
        """
        size = content_feat.size()
        batch_size = size[0]  # 批次大小 N
        
        # 计算风格和内容特征的均值和标准差
        style_mean, style_std = self._calc_mean_std(style_feat)  # [1, C, 1, 1]
        content_mean, content_std = self._calc_mean_std(content_feat)  # [N, C, 1, 1]
        
        # 对内容特征进行归一化
        normalized_feat = (content_feat - content_mean) / content_std  # [N, C, H, W]
        
        # 将风格统计信息和内容统计信息拼接，用于全连接层预测
        mixed_style_mean = torch.cat(
            (style_mean.expand(batch_size, -1, -1, -1), content_mean), 1
        ).squeeze((2, 3))  # [N, 2C]
        mixed_style_std = torch.cat(
            (style_std.expand(batch_size, -1, -1, -1), content_std), 1
        ).squeeze((2, 3))  # [N, 2C]
        
        # 通过全连接层预测新的风格统计信息
        new_style_mean = self.fc1(mixed_style_mean)[:, :, None, None]  # [N, C, 1, 1]
        new_style_std = self.fc2(mixed_style_std)[:, :, None, None]  # [N, C, 1, 1]
        
        # 使用预测的风格统计信息对归一化后的特征进行缩放和平移
        return normalized_feat * new_style_std + new_style_mean
    
    def _calc_mean_std(self, feat, eps=1e-5):
        """计算特征的均值和标准差
        
        Args:
            feat: 输入特征 [N, C, H, W]
            eps: 防止除零的小数值
            
        Returns:
            tuple: (均值 [N, C, 1, 1], 标准差 [N, C, 1, 1])
        """
        N, C = feat.size()[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


class enhance_base:
    """风格迁移基类
    
    提供完整的风格迁移流程，包括：
    - 风格图像的加载和预处理
    - 多层级编码器-解码器管理
    - 批量风格迁移
    - torch.compile 编译加速支持
    - 训练过程可视化
    
    属性:
        pixel_means: 图像像素均值，用于预处理
        target_size: 目标图像尺寸
        encoders: 编码器模块列表
        decoders: 解码器模块列表
        fc1/fc2: 风格预测全连接层
        alpha: 风格迁移强度
        compiled_wrapper: 编译后的 TAMWrapper（如果启用编译）
        step: 当前训练步数，用于可视化
    
    示例:
        >>> args = parse_args()
        >>> enhancer = enhance_base(args, encoders, decoders, fcs)
        >>> styled_image = enhancer.add_style(content_image, flag=0)
    """
    
    compiled_wrapper: TAMWrapper | None
    
    def add_style(self, content: torch.Tensor, flag):
        """对输入图像添加风格
        
        这是主要的风格迁移接口，支持批量处理和编译加速。
        如果启用了 compiled_wrapper，会使用编译后的整体模型进行推理；
        否则会对每张图像单独进行风格迁移。
        
        Args:
            content: 内容图像 [N, C, H, W]
            flag: 编码器起始层索引，0表示从第一层开始
            
        Returns:
            torch.Tensor: 风格化后的图像 [N, C, H, W]
        """
        if flag == 0:
            # 只在第一层时增加步数计数
            self.step += 1
        assert (len(content.shape) == 4)
        
        # 选择风格图像：随机或固定
        if self.args.random_style:
            style = self.load_style_img(
                self.args, content, wh=(content.size(3), content.size(2))
            )
        else:
            # 检查预加载的风格特征是否与当前内容尺寸匹配
            if self.style_feats[flag][0].size() == content[0].size():
                style = self.style_feats[flag][0]
            else:
                style = self.load_style_img(
                    self.args, content, wh=(content.size(3), content.size(2))
                )
        
        # 执行风格迁移（无梯度）
        with torch.no_grad():
            if self.compiled_wrapper is not None:
                # 使用编译后的整体模型进行批量推理
                output = self.compiled_wrapper(content, style.unsqueeze(0), flag, self.alpha)
            else:
                # 逐张图像进行风格迁移
                output = []
                for i in range(0, content.shape[0]):
                    output.append(self.style_transfer(content[i], style, flag, self.alpha))
                output = torch.stack(output)
        
        # 第一层输出需要进行后处理：添加像素均值、裁剪到有效范围
        if flag == 0:
            output = (
                output.permute(0, 2, 3, 1) + 
                torch.from_numpy(self.pixel_means).float().cuda()
            ).clamp(0, 255)
            output = output.permute(0, 3, 1, 2).contiguous()
        
        return output.detach()

    def __init__(self, args, encoders, decoders, fcs):
        """初始化风格迁移基类
        
        Args:
            args: 配置参数，包含各种路径和超参数
            encoders: 编码器模块列表
            decoders: 解码器模块列表
            fcs: 全连接层列表 [fc1, fc2]
        """
        assert len(encoders) == len(decoders)
        
        # 图像预处理参数（ImageNet 预训练模型的像素均值）
        self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        self.target_size = args.imgsz
        self.encoders = encoders
        self.num = len(self.encoders)
        self.decoders = decoders
        self.fc1 = fcs[0]
        self.fc2 = fcs[1]
        self.alpha = args.style_add_alpha
        self.args = args
        self.compiled_wrapper = None
        
        # 将模型移动到 GPU 并设置为评估模式
        if args.cuda:
            for encoder in self.encoders:
                encoder.cuda()
            for decoder in self.decoders:
                decoder.cuda()
            self.fc1.cuda()
            self.fc2.cuda()
        
        for encoder in self.encoders:
            encoder.eval()
        for decoder in self.decoders:
            decoder.eval()
        self.fc1.eval()
        self.fc2.eval()
        
        # 如果启用了编译选项，编译 TAM 模块
        if getattr(args, 'compile_tam', False) and hasattr(torch, 'compile'):
            self._compile_tam(args)
        
        # 加载固定风格图像（如果未启用随机风格）
        self.style_image = None
        if not self.args.random_style:
            self.style_image = self.load_and_process_style_img(args.style_path)
            self.style_feats = [self.style_image.unsqueeze(0)]

        # 创建可视化输出目录
        path = os.path.join(os.path.dirname(__file__), '..', self.args.log_dir, 'noise')
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        self.step = 0

    def _compile_tam(self, args):
        """编译 TAM 模块以加速推理
        
        参考 ultralytics.utils.torch_utils.attempt_compile 的实现，
        创建 TAMWrapper 包装器并使用 torch.compile 进行编译。
        
        Args:
            args: 配置参数，包含 compile_tam 编译模式
        """
        compile_mode = args.compile_tam if isinstance(args.compile_tam, str) else 'default'
        
        try:
            # 创建包装器并移动到设备
            device = torch.device('cuda' if args.cuda else 'cpu')
            wrapper = TAMWrapper(self.encoders, self.decoders, self.fc1, self.fc2)
            wrapper = wrapper.to(device)
            wrapper.eval()
            
            # 使用 inductor 后端编译包装器
            self.compiled_wrapper = torch.compile(wrapper, mode=compile_mode, backend='inductor')
            
            print(f"TAM compiled")
            
        except Exception as e:
            print(f"torch.compile failed, continuing uncompiled: {e}")
            self.compiled_wrapper = None

    def get_style_feats(self, args):
        """获取风格特征
        
        Args:
            args: 配置参数
            
        Returns:
            list: 包含风格特征的列表
        """
        if self.style_image is None:
            self.style_image = self.load_and_process_style_img(args.style_path)
        feats = [self.style_image.unsqueeze(0)]
        return feats

    def load_and_process_style_img(self, path):
        """加载并预处理风格图像
        
        处理流程：
        1. 打开图像并转换为 RGB
        2. 转换为 numpy 数组并调整通道顺序 (RGB -> BGR)
        3. 减去像素均值
        4. 按比例缩放到目标尺寸
        5. 转换为 torch.Tensor 并调整维度顺序
        
        Args:
            path: 风格图像路径
            
        Returns:
            torch.Tensor: 预处理后的风格图像 [C, H, W]
        """
        im = Image.open(path)
        im = im.convert('RGB')
        im = np.array(im)
        im = im[:, :, ::-1]  # RGB 转 BGR
        im = im.astype(np.float32, copy=False)
        im -= self.pixel_means  # 减去像素均值
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_scale = float(self.target_size) / float(im_size_min)
        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )
        im = torch.from_numpy(im).permute(2, 0, 1).contiguous()
        if self.args.cuda:
            im = im.cuda()
        return im
    
    def style_transfer(self, content: torch.Tensor, style, flag, alpha=1.0):
        """单张图像的风格迁移
        
        对单张图像执行完整的编码-解码风格迁移流程。
        如果 compiled_wrapper 可用，会使用编译后的版本。
        
        Args:
            content: 内容图像 [C, H, W]
            style: 风格图像 [C, H, W]
            flag: 编码器起始层索引
            alpha: 风格迁移强度 [0, 1]
            
        Returns:
            torch.Tensor: 风格化后的图像 [1, C, H, W]
        """
        assert (0.0 <= alpha <= 1.0)
        assert (len(content.size()) == 3)
        content = content.unsqueeze(0)  # 添加批次维度
        style = style.unsqueeze(0)
        size = content.size()
        
        # 使用编译后的包装器（如果可用）
        if self.compiled_wrapper is not None:
            with torch.no_grad():
                feat = self.compiled_wrapper(content, style, flag, alpha)
        else:
            # 原始实现：手动编码-解码
            with torch.no_grad():
                # 编码阶段
                for i in range(flag, self.num):
                    content = self.encoders[i](content).float()
                    style = self.encoders[i](style).float()
                # 自适应实例归一化
                feat = self.adaptive_instance_normalization(content, style, self.fc1, self.fc2)
                feat = feat * alpha + content * (1 - alpha)
                # 解码阶段
                for i in range(self.num - flag):
                    feat = self.decoders[i](feat)

        assert feat.size() == size, f"TAM 生成图像的大小{feat.shape}和需要的大小{size}不一致"
        return feat
            
    def load_style_img(self, args, content=None, wh=None):
        """加载风格图像
        
        根据配置选择风格图像来源：
        - 随机风格：从当前批次中随机选择或从图像列表中随机加载
        - 固定风格：使用预加载的风格图像
        
        Args:
            args: 配置参数
            content: 当前批次的内容图像 [N, C, H, W]，用于随机风格选择
            wh: 目标尺寸 (width, height)
            
        Returns:
            torch.Tensor: 风格图像 [C, H, W]
        """
        if args.random_style and content is not None and len(content.size()) == 4:
            # 从当前批次中随机选择一张作为风格图像（加速训练）
            i = random.randint(0, content.size(0) - 1)
            im = content[i]
            on_device = True
        else:
            on_device = False
            if args.random_style:
                # 从图像列表中随机加载风格图像
                i = random.randint(0, len(args.imgs_paths) - 1)
                path = args.imgs_paths[i]
                im = Image.open(path)
                im = im.convert('RGB')
                im = np.array(im)
                im = im[:, :, ::-1]
                im = im.astype(np.float32, copy=False)
                im -= self.pixel_means
                im_shape = im.shape
                im_size_min = np.min(im_shape[0:2])
                if wh is not None:
                    im = cv2.resize(im, wh, interpolation=cv2.INTER_LINEAR)
                else:
                    im_scale = float(self.target_size) / float(im_size_min)
                    im = cv2.resize(
                        im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR
                    )
                im = torch.from_numpy(im).permute(2, 0, 1).contiguous()
            else:
                # 使用固定风格图像
                im = self.style_image
        
        if args.cuda and not on_device:
            im = im.cuda()
        return im

    def coral(self, source, target):
        """CORAL (CORrelation Alignment) 风格迁移
        
        通过匹配协方差矩阵来进行风格迁移，是一种更高级的风格迁移技术。
        
        Args:
            source: 源风格特征 [C, H, W]
            target: 目标内容特征 [C, H, W]
            
        Returns:
            torch.Tensor: 风格迁移后的特征 [C, H, W]
        """
        # source: 源风格特征
        # target: 目标内容特征
        
        # 计算源特征的统计信息
        source_f, source_f_mean, source_f_std = self._calc_feat_flatten_mean_std(source)
        source_f_norm = (
            source_f - source_f_mean.expand_as(source_f)
        ) / source_f_std.expand_as(source_f)
        source_f_cov_eye = (
            torch.mm(source_f_norm, source_f_norm.t()) +
            torch.eye(source.size(0)).cuda()
        )

        # 计算目标特征的统计信息
        target_f, target_f_mean, target_f_std = self._calc_feat_flatten_mean_std(target)
        target_f_norm = (
            target_f - target_f_mean.expand_as(target_f)
        ) / target_f_std.expand_as(target_f)
        target_f_cov_eye = (
            torch.mm(target_f_norm, target_f_norm.t()) +
            torch.eye(source.size(0)).cuda()
        )

        # 通过协方差匹配进行风格迁移
        source_f_norm_transfer = torch.mm(
            self._mat_sqrt(target_f_cov_eye),
            torch.mm(torch.inverse(self._mat_sqrt(source_f_cov_eye)), source_f_norm)
        )

        source_f_transfer = (
            source_f_norm_transfer * target_f_std.expand_as(source_f_norm) +
            target_f_mean.expand_as(source_f_norm)
        )

        return source_f_transfer.view(source.size())

    def _mat_sqrt(self, x):
        """计算矩阵的平方根
        
        使用 SVD 分解计算矩阵的平方根。
        
        Args:
            x: 输入矩阵
            
        Returns:
            torch.Tensor: 矩阵的平方根
        """
        U, D, V = torch.svd(x)
        return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())

    def _calc_feat_flatten_mean_std(self, feat):
        """计算特征的展平均值和标准差
        
        将 3D 特征 [C, H, W] 展平为 [C, H*W]，然后计算每通道的均值和标准差。
        
        Args:
            feat: 输入特征 [C, H, W]
            
        Returns:
            tuple: (展平特征 [C, H*W], 均值 [C, 1], 标准差 [C, 1])
        """
        feat_flatten = feat.view(feat.size(0), -1)
        mean = feat_flatten.mean(dim=-1, keepdim=True)
        std = feat_flatten.std(dim=-1, keepdim=True)
        return feat_flatten, mean, std

    def adaptive_instance_normalization(self, content_feat, style_feat, fc1, fc2):
        """自适应实例归一化（单张图像版本）
        
        与 _adaptive_instance_normalization 功能相同，但处理单张图像。
        
        Args:
            content_feat: 内容特征 [1, C, H, W]
            style_feat: 风格特征 [1, C, H, W]
            fc1: 风格均值预测全连接层
            fc2: 风格标准差预测全连接层
            
        Returns:
            torch.Tensor: 融合后的特征 [1, C, H, W]
        """
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (
            content_feat - content_mean.expand(size)
        ) / content_std.expand(size)
        
        mixed_style_mean = torch.cat((style_mean, content_mean), 1).squeeze(2).squeeze(2)
        mixed_style_std = torch.cat((style_std, content_std), 1).squeeze(2).squeeze(2)

        new_style_mean = (fc1(mixed_style_mean)).unsqueeze(2).unsqueeze(2)
        new_style_std = (fc2(mixed_style_std)).unsqueeze(2).unsqueeze(2)
        final = normalized_feat * new_style_std.expand(size) + new_style_mean.expand(size)
        return final

    def calc_mean_std(self, feat, eps=1e-5):
        """计算特征的均值和标准差（4D 张量版本）
        
        Args:
            feat: 输入特征 [N, C, H, W]
            eps: 防止除零的小数值
            
        Returns:
            tuple: (均值 [N, C, 1, 1], 标准差 [N, C, 1, 1])
        """
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def show(self, feat, content=False, save=True):
        """可视化特征/图像
        
        将特征张量转换为图像并显示或保存。
        
        Args:
            feat: 特征张量 [N, C, H, W]
            content: 是否为内容图像，影响保存的文件名
            save: 是否保存图像，False 则直接显示
        """
        feat = torch.nan_to_num(feat)  # 处理 NaN 值
        for i in range(feat.size(0)):
            s = feat[i].transpose(0, 1).transpose(1, 2).cpu().numpy()
            s = np.clip(s, 0, 255).astype(np.uint8)
            
            if save:
                path = self.args.log_dir
                if not os.path.exists(path):
                    os.makedirs(path)
                if content:
                    matplotlib.image.imsave(
                        os.path.join(path, 'step' + str(self.step) + '_real' + str(i) + '.jpg'),
                        s
                    )
                else:
                    matplotlib.image.imsave(
                        os.path.join(path, 'step' + str(self.step) + '_' + str(i) + '.jpg'),
                        s
                    )
            else:
                plt.imshow(s)
                plt.show()
