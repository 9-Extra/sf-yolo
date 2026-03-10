# coding=utf-8
"""
Target Augment Model (TAM) - 自包含版本

基于 VGG16 的风格迁移模型，包含完整的编码器-解码器结构：
- VGG16 编码器：提取多层级图像特征
- 对称解码器：将特征还原为图像
- 全连接层：预测风格统计信息

模型结构:
    编码器: VGG16 features (去掉最后一层 maxpool)
        - Encoder 0: conv1_1 -> relu1_1
        - Encoder 1: conv1_2 -> relu1_2 -> maxpool1
        - Encoder 2: conv2_1 -> relu2_1 -> conv2_2 -> relu2_2 -> maxpool2
        - Encoder 3: conv3_1 -> relu3_1 -> ... -> relu3_3 -> maxpool3
        
    解码器: 对称的上采样结构
        - Decoder 0: conv -> relu -> upsample (512 -> 256)
        - Decoder 1: conv -> relu -> conv -> relu -> upsample (256 -> 128)
        - Decoder 2: conv -> relu -> conv -> relu -> upsample (128 -> 64)
        - Decoder 3: conv -> relu -> conv (64 -> 3)
        
    全连接层:
        - fc1: 1024 -> 512 -> 512 (均值预测)
        - fc2: 1024 -> 512 -> 512 (标准差预测)
"""

import torch
import torch.nn as nn
from torchvision import models


class TargetAugmentModel(nn.Module):
    """TAM 风格迁移模型
    
    自包含的单一神经网络，整合编码器、解码器和风格预测层，
    实现基于自适应实例归一化的多层级风格迁移。
    
    属性:
        encoders: VGG16 编码器模块列表 (nn.ModuleList，4层)
        decoders: 解码器模块列表 (nn.ModuleList，4层)
        fc1: 风格均值预测全连接层
        fc2: 风格标准差预测全连接层
        num: 编码器/解码器层数
    
    示例:
        >>> model = TargetAugmentModel()
        >>> model.load_weights('encoder.pth', 'decoder.pth', 'fc1.pth', 'fc2.pth')
        >>> output = model(content_images, style_image, flag=0, alpha=1.0)
    """
    
    encoders: nn.ModuleList
    decoders: nn.ModuleList
    fc1: nn.Sequential
    fc2: nn.Sequential
    num: int
    
    def __init__(self):
        """初始化 TAM 模型"""
        super().__init__()
        
        # 构建编码器 (VGG16 features，去掉最后一层 maxpool)
        vgg = self._build_encoder()
        
        # 构建解码器 (对称结构)
        decoder = self._build_decoder()
        
        # 构建全连接层
        fc1, fc2 = self._build_fcs()
        
        # 将网络分割为多个层级并注册为 ModuleList
        self.encoders = nn.ModuleList(self._split_encoder(vgg))
        self.decoders = nn.ModuleList(self._split_decoder(decoder))
        self.fc1 = fc1
        self.fc2 = fc2
        self.num = len(self.encoders)
    
    def _build_encoder(self):
        """构建 VGG16 编码器
        
        加载 torchvision 的 VGG16 模型，截取 features 部分（去掉最后几层），
        只保留前19层（到 relu3_3 + pool3），并设置 ceil_mode=True 以确保特征图尺寸正确。
        
        注意: 原始分割逻辑基于19层结构 (0-18):
        - Encoder 0: layers 0-1
        - Encoder 1: layers 2-6  
        - Encoder 2: layers 7-11
        - Encoder 3: layers 12-18
        
        Returns:
            nn.Sequential: VGG16 编码器网络 (19层)
        """
        vgg = models.vgg16()
        # 截取 features 部分，只保留前19层（索引0-18）
        # 这包括 conv1_1 到 pool3，不包含后续的 conv4/conv5 层
        vgg = nn.Sequential(*list(vgg.features._modules.values())[:19])
        # 设置 ceil_mode=True 以确保上采样后的尺寸匹配
        vgg[4].ceil_mode = True   # maxpool1
        vgg[9].ceil_mode = True   # maxpool2
        vgg[16].ceil_mode = True  # maxpool3
        return vgg
    
    def _build_decoder(self):
        """构建对称解码器
        
        创建与 VGG16 编码器对称的解码器网络，使用最近邻上采样和卷积层
        逐步将特征图从 512 通道还原为 3 通道图像。
        
        Returns:
            nn.Sequential: 解码器网络
        """
        decoder = nn.Sequential(
            # Block 0: 512 -> 256
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 1: 256 -> 256
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 2: 128 -> 128
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 3: 64 -> 3
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        return decoder
    
    def _build_fcs(self):
        """构建风格统计信息预测的全连接层
        
        创建两个并行的全连接网络，分别用于预测：
        - fc1: 融合后的风格均值
        - fc2: 融合后的风格标准差
        
        输入维度为 1024（512维风格 + 512维内容的拼接），
        输出维度为 512（通道数）。
        
        Returns:
            tuple: (fc1, fc2) 全连接层
        """
        fc1 = nn.Sequential(
            nn.Linear(1024, 512),  # 输入: 风格512 + 内容512
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)     # 输出: 新的均值
        )
        fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)     # 输出: 新的标准差
        )
        return fc1, fc2
    
    def _split_encoder(self, vgg):
        """将 VGG16 编码器分割为多层级模块
        
        按照特征层级分割为4个子模块：
        - Encoder 0: layers 0-1 (conv1_1 -> relu1_1)
        - Encoder 1: layers 2-6 (conv1_2 -> relu1_2 -> maxpool1)
        - Encoder 2: layers 7-11 (conv2_1 -> ... -> maxpool2)
        - Encoder 3: layers 12-18 (conv3_1 -> ... -> maxpool3)
        
        Args:
            vgg: VGG16 编码器网络
            
        Returns:
            list: 编码器模块列表
        """
        encoders = []
        enc_layers = list(vgg.children())
        encoders.append(nn.Sequential(*enc_layers[:2]))    # relu1_1
        encoders.append(nn.Sequential(*enc_layers[2:7]))   # relu1_2 + pool1
        encoders.append(nn.Sequential(*enc_layers[7:12]))  # relu2_2 + pool2
        encoders.append(nn.Sequential(*enc_layers[12:]))   # relu3_3 + pool3
        return encoders
    
    def _split_decoder(self, decoder):
        """将解码器分割为多层级模块
        
        按照层级分割为4个子模块：
        - Decoder 0: layers 0-6 (conv -> relu -> upsample)
        - Decoder 1: layers 7-11 (conv -> relu -> ... -> upsample)
        - Decoder 2: layers 12-16 (conv -> relu -> ... -> upsample)
        - Decoder 3: layers 17-21 (conv -> relu -> conv)
        
        Args:
            decoder: 解码器网络
            
        Returns:
            list: 解码器模块列表
        """
        dec_layers = list(decoder.children())
        decoders = []
        decoders.append(nn.Sequential(*dec_layers[:7]))    # 512->256 + upsample
        decoders.append(nn.Sequential(*dec_layers[7:12]))  # 256->256 + upsample
        decoders.append(nn.Sequential(*dec_layers[12:17])) # 256->128 + upsample
        decoders.append(nn.Sequential(*dec_layers[17:]))   # 128->64->3
        return decoders
    
    def load_weights(self, encoder_path: str, decoder_path: str, fc1_path: str, fc2_path: str):
        """加载预训练权重并冻结参数
        
        Args:
            encoder_path: VGG16 编码器权重路径
            decoder_path: 解码器权重路径
            fc1_path: 均值预测层权重路径
            fc2_path: 标准差预测层权重路径
        """
        # 加载编码器权重
        vgg_state = torch.load(encoder_path, map_location='cpu', weights_only=False)
        if 'model' in vgg_state:
            vgg_state = vgg_state['model']
        
        # 过滤权重：只保留前19层的权重 (0-18)
        filtered_state = {}
        for k, v in vgg_state.items():
            # 解析层索引
            parts = k.split('.')
            if len(parts) >= 1:
                try:
                    layer_idx = int(parts[0])
                    if layer_idx < 19:  # 只保留前19层
                        filtered_state[k] = v
                except ValueError:
                    continue
        
        # 直接加载到分割后的编码器
        # 先构建一个19层的参考编码器
        ref_vgg = models.vgg16()
        ref_vgg = nn.Sequential(*list(ref_vgg.features._modules.values())[:19])
        ref_vgg.load_state_dict(filtered_state)
        
        # 将权重复制到分割后的编码器
        # Encoder 0: layers 0-1 (Conv2d(3,64), ReLU)
        self.encoders[0][0].weight.data.copy_(ref_vgg[0].weight.data)
        self.encoders[0][0].bias.data.copy_(ref_vgg[0].bias.data)
        
        # Encoder 1: layers 2-6 (Conv2d(64,64), ReLU, MaxPool, Conv2d(64,128), ReLU)
        self.encoders[1][0].weight.data.copy_(ref_vgg[2].weight.data)
        self.encoders[1][0].bias.data.copy_(ref_vgg[2].bias.data)
        self.encoders[1][3].weight.data.copy_(ref_vgg[5].weight.data)
        self.encoders[1][3].bias.data.copy_(ref_vgg[5].bias.data)
        
        # Encoder 2: layers 7-11 (Conv2d(128,128), ReLU, MaxPool, Conv2d(128,256), ReLU)
        self.encoders[2][0].weight.data.copy_(ref_vgg[7].weight.data)
        self.encoders[2][0].bias.data.copy_(ref_vgg[7].bias.data)
        self.encoders[2][3].weight.data.copy_(ref_vgg[10].weight.data)
        self.encoders[2][3].bias.data.copy_(ref_vgg[10].bias.data)
        
        # Encoder 3: layers 12-18 (Conv2d(256,256), ReLU, Conv2d(256,256), ReLU, MaxPool, Conv2d(256,512), ReLU)
        self.encoders[3][0].weight.data.copy_(ref_vgg[12].weight.data)
        self.encoders[3][0].bias.data.copy_(ref_vgg[12].bias.data)
        self.encoders[3][2].weight.data.copy_(ref_vgg[14].weight.data)
        self.encoders[3][2].bias.data.copy_(ref_vgg[14].bias.data)
        self.encoders[3][5].weight.data.copy_(ref_vgg[17].weight.data)
        self.encoders[3][5].bias.data.copy_(ref_vgg[17].bias.data)
        
        # 加载解码器权重
        decoder_state = torch.load(decoder_path, map_location='cpu', weights_only=False)
        # 重建解码器用于加载权重
        full_decoder = self._build_decoder()
        full_decoder.load_state_dict(decoder_state)
        
        # 重新分割解码器并更新权重
        new_decoders = self._split_decoder(full_decoder)
        for i, dec in enumerate(new_decoders):
            self.decoders[i].load_state_dict(dec.state_dict())
        
        # 加载全连接层权重
        self.fc1.load_state_dict(torch.load(fc1_path, map_location='cpu', weights_only=False))
        self.fc2.load_state_dict(torch.load(fc2_path, map_location='cpu', weights_only=False))
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, content: torch.Tensor, style: torch.Tensor, flag: int = 0, alpha: float = 1.0):
        """TAM 风格迁移的前向传播
        
        执行完整的多层级编码-解码流程，包括：
        1. 多层级编码：对内容图像和风格图像进行分层编码
        2. 自适应实例归一化：将内容特征与风格特征进行融合
        3. 多层级解码：将融合后的特征解码回图像空间
        
        Args:
            content: 内容图像 [N, C, H, W]，需要进行风格迁移的图像
            style: 风格图像 [N, C, H, W]，提供目标风格的图像
            flag: 编码器起始层索引，用于控制从哪一层开始编码
            alpha: 风格迁移强度，范围 [0, 1]，0表示保持原图，1表示完全迁移
            
        Returns:
            torch.Tensor: 风格特征 [N, C, H, W]
        """
        # 编码阶段：从指定层开始，逐层提取特征
        for i in range(flag, self.num):
            content = self.encoders[i](content).float()
            style = self.encoders[i](style).float()
        
        # 自适应实例归一化：将风格特征注入内容特征
        feat = self._adaptive_instance_normalization(content, style)
        feat = feat * alpha + content * (1 - alpha)  # 混合原始内容和风格化特征
        
        # 解码阶段：将融合后的特征解码到图像空间
        for i in range(self.num - flag):
            feat = self.decoders[i](feat)
        
        return feat
    
    def _adaptive_instance_normalization(self, content_feat: torch.Tensor, style_feat: torch.Tensor):
        """自适应实例归一化 (Adaptive Instance Normalization)
        
        将内容特征的统计特性（均值和标准差）替换为风格特征的统计特性，
        同时通过全连接层学习更复杂的风格迁移映射。
        
        Args:
            content_feat: 内容特征 [N, C, H, W]
            style_feat: 风格特征 [N, C, H, W]
            
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
    
    @staticmethod
    def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
        """计算特征的均值和标准差
        
        Args:
            feat: 输入特征 [N, C, H, W]
            eps: 防止除零的小数值
            
        Returns:
            tuple: (均值 [N, C, 1, 1], 标准差 [N, C, 1, 1])
        """
        feat_std, feat_mean = torch.std_mean(feat, dim=[2, 3], unbiased=False, keepdim=True)        
        # 添加 epsilon 防止除零
        feat_std = feat_std + eps
        return feat_mean, feat_std
    
    # ------------------------------训练相关---------------------------------
    
    def calc_content_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        return torch.nn.functional.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        input_mean, input_std = self._calc_mean_std(input)
        target_mean, target_std = self._calc_mean_std(target)
        return torch.nn.functional.mse_loss(input_mean, target_mean) + torch.nn.functional.mse_loss(
            input_std, target_std
        )

    def calc_content_constraint_loss(self, input, target):
        assert input.size() == target.size()
        return torch.nn.functional.mse_loss(input, target)
    
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, "enc_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def decode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, "dec_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def loss(self, content, style, scenario_name, flag=0, alpha=1.0):
        assert 0 <= alpha <= 1
        if flag == 0:
            style_feats = self.encode_with_intermediate(style)
            content_feats = self.encode_with_intermediate(content)

            if scenario_name != "city2foggy":
                t = self._adaptive_instance_normalization(content_feats[-1], style_feats[-1])
                t = alpha * t + (1 - alpha) * content_feats[-1]

                g_ts = self.decode_with_intermediate(t)
                g_t_feats = self.encode_with_intermediate(g_ts[-1])

                loss_c = self.calc_content_loss(g_t_feats[-1], t.detach())
            else:
                loss_c = 0

            deco_ts = self.decode_with_intermediate(content_feats[-1])
            loss_const = self.calc_content_constraint_loss(
                content_feats[0], deco_ts[-2]
            )
            for i in range(1, 3):
                loss_const += self.calc_content_constraint_loss(
                    content_feats[i], deco_ts[-(i + 2)]
                )
            loss_const += self.calc_content_constraint_loss(content, deco_ts[-1])
            return loss_c, loss_const
        elif flag == 1:
            style_feats = self.encode_with_intermediate(style)
            content_feats = self.encode_with_intermediate(content)
            t = self._adaptive_instance_normalization(content_feats[-1], style_feats[-1])
            t = alpha * t + (1 - alpha) * content_feats[-1]

            g_ts = self.decode_with_intermediate(t)
            g_t_feats = self.encode_with_intermediate(g_ts[-1])

            if scenario_name == "city2foggy":
                loss_s_1 = self.calc_style_loss(g_t_feats[-1], style_feats[-1])
                loss_s_2 = self.calc_style_loss(g_t_feats[-1], content_feats[-1])
            else:
                loss_s_1 = self.calc_style_loss(g_t_feats[0], style_feats[0])
                for i in range(1, 4):
                    loss_s_1 += self.calc_style_loss(g_t_feats[i], style_feats[i])
                loss_s_2 = self.calc_style_loss(g_t_feats[0], content_feats[0])
                for i in range(1, 4):
                    loss_s_2 += self.calc_style_loss(g_t_feats[i], content_feats[i])

            return loss_s_1, loss_s_2
