# coding=utf-8
"""
基于 VGG16 的风格迁移实现模块

提供使用 VGG16 作为骨干网络的风格迁移功能，包括：
- VGG16 编码器：用于提取图像的多层级特征
- 对称解码器：将特征还原为图像
- 全连接层：预测风格统计信息

主要类:
    enhance_vgg16: 基于 VGG16 的风格迁移类，继承自 enhance_base

网络结构:
    编码器: VGG16 features (去掉最后一层 maxpool)
        - Encoder 1: conv1_1 -> relu1_1
        - Encoder 2: conv1_2 -> relu1_2 -> maxpool1
        - Encoder 3: conv2_1 -> relu2_1 -> conv2_2 -> relu2_2 -> maxpool2
        - Encoder 4: conv3_1 -> relu3_1 -> ... -> relu3_3 -> maxpool3
        
    解码器: 对称的上采样结构
        - Decoder 1: conv -> relu -> upsample (512 -> 256)
        - Decoder 2: conv -> relu -> conv -> relu -> upsample (256 -> 128)
        - Decoder 3: conv -> relu -> conv -> relu -> upsample (128 -> 64)
        - Decoder 4: conv -> relu -> conv (64 -> 3)
        
    全连接层: 预测风格统计信息
        - fc1: 1024 -> 512 -> 512 (均值预测)
        - fc2: 1024 -> 512 -> 512 (标准差预测)
"""

from torchvision import models
import torch.nn as nn
import torch
from TargetAugment.enhance_base import enhance_base


class enhance_vgg16(enhance_base):
    """基于 VGG16 的风格迁移类
    
    使用预训练的 VGG16 网络作为编码器，配合对称的解码器结构，
    实现基于自适应实例归一化的风格迁移。
    
    属性:
        encoders: VGG16 编码器模块列表（4层）
        decoders: 解码器模块列表（4层）
        fc1/fc2: 风格统计信息预测全连接层
    
    示例:
        >>> args = parse_args()
        >>> args.encoder_path = 'path/to/encoder.pth'
        >>> args.decoder_path = 'path/to/decoder.pth'
        >>> args.fc1 = 'path/to/fc1.pth'
        >>> args.fc2 = 'path/to/fc2.pth'
        >>> enhancer = enhance_vgg16(args)
        >>> styled = enhancer.add_style(content_images, flag=0)
    """
    
    def __init__(self, args):
        """初始化 enhance_vgg16
        
        Args:
            args: 配置参数，包含以下路径：
                - encoder_path: VGG16 编码器权重路径
                - decoder_path: 解码器权重路径
                - fc1: 均值预测层权重路径
                - fc2: 标准差预测层权重路径
        """
        # 创建解码器网络
        decoder = self.get_decoder()
        # 创建 VGG16 编码器
        vgg = self.get_vgg()
        # 创建全连接层
        fcs = self.get_fcs()
        
        # 加载预训练权重并冻结参数
        vgg, decoder, fcs = self.load_param(args, vgg, decoder, fcs)
        
        # 将网络分割为多个层级
        self.encoders, self.decoders = self.splits(vgg, decoder)
        
        # 调用父类初始化
        enhance_base.__init__(self, args, self.encoders, self.decoders, fcs)

    def splits(self, vgg, decoder):
        """将 VGG16 和解码器分割为多层级模块
        
        将网络按照特征层级分割为多个子模块，以便进行多层级风格迁移。
        
        Args:
            vgg: VGG16 编码器网络
            decoder: 解码器网络
            
        Returns:
            tuple: (encoders列表, decoders列表)
            
        编码器层级划分:
            - Encoder 0: layers 0-1 (conv1_1 -> relu1_1)
            - Encoder 1: layers 2-6 (conv1_2 -> relu1_2 -> maxpool1)
            - Encoder 2: layers 7-11 (conv2_1 -> ... -> maxpool2)
            - Encoder 3: layers 12-18 (conv3_1 -> ... -> maxpool3)
            
        解码器层级划分:
            - Decoder 0: layers 0-6 (conv -> relu -> upsample)
            - Decoder 1: layers 7-11 (conv -> relu -> ... -> upsample)
            - Decoder 2: layers 12-16 (conv -> relu -> ... -> upsample)
            - Decoder 3: layers 17-21 (conv -> relu -> conv)
        """
        encoders = []
        decoders = []
        
        # 分割 VGG16 编码器为4个层级
        encoders.append(nn.Sequential(*list(vgg._modules.values())[:2]))    # relu1_1
        encoders.append(nn.Sequential(*list(vgg._modules.values())[2:7]))   # relu1_2 + pool1
        encoders.append(nn.Sequential(*list(vgg._modules.values())[7:12]))  # relu2_2 + pool2
        encoders.append(nn.Sequential(*list(vgg._modules.values())[12:]))   # relu3_3 + pool3
        
        # 分割解码器为4个层级
        decoders.append(nn.Sequential(*list(decoder._modules.values())[:7]))    # 512->256 + upsample
        decoders.append(nn.Sequential(*list(decoder._modules.values())[7:12]))  # 256->256 + upsample
        decoders.append(nn.Sequential(*list(decoder._modules.values())[12:17])) # 256->128 + upsample
        decoders.append(nn.Sequential(*list(decoder._modules.values())[17:]))   # 128->64->3
        
        return encoders, decoders
    
    def get_fcs(self):
        """创建风格统计信息预测的全连接层
        
        创建两个并行的全连接网络，分别用于预测：
        - fc1: 融合后的风格均值
        - fc2: 融合后的风格标准差
        
        输入维度为 1024（512维风格 + 512维内容的拼接），
        输出维度为 512（通道数）。
        
        Returns:
            list: [fc1, fc2] 全连接层列表
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
        return [fc1, fc2]

    def get_vgg(self):
        """创建 VGG16 编码器
        
        加载 torchvision 的 VGG16 模型，截取 features 部分（去掉最后一层 maxpool），
        并设置 ceil_mode=True 以确保特征图尺寸正确。
        
        Returns:
            nn.Sequential: VGG16 编码器网络
            
        网络结构:
            - conv1_1 -> relu1_1
            - conv1_2 -> relu1_2 -> maxpool1 (ceil_mode=True)
            - conv2_1 -> relu2_1 -> conv2_2 -> relu2_2 -> maxpool2 (ceil_mode=True)
            - conv3_1 -> relu3_1 -> ... -> relu3_3 -> maxpool3 (ceil_mode=True)
        """
        vgg = models.vgg16()
        # 截取 features 部分，去掉最后一层 maxpool
        vgg = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        # 设置 ceil_mode=True 以确保上采样后的尺寸匹配
        vgg[4].ceil_mode = True   # maxpool1
        vgg[9].ceil_mode = True   # maxpool2
        vgg[16].ceil_mode = True  # maxpool3
        return vgg

    def get_decoder(self):
        """创建对称解码器
        
        创建与 VGG16 编码器对称的解码器网络，使用最近邻上采样和卷积层
        逐步将特征图从 512 通道还原为 3 通道图像。
        
        Returns:
            nn.Sequential: 解码器网络
            
        网络结构:
            - Block 1 (512->256): Conv 512->256 -> ReLU -> Upsample x2
            - Block 2 (256->256): Conv 256->256 -> ReLU -> Conv 256->256 -> ReLU -> Upsample x2
            - Block 3 (256->128): Conv 256->128 -> ReLU -> Upsample x2
            - Block 4 (128->64):  Conv 128->128 -> ReLU -> Conv 128->64 -> ReLU -> Upsample x2
            - Block 5 (64->3):    Conv 64->64 -> ReLU -> Conv 64->3 (输出层，无激活)
        """
        decoder = nn.Sequential(
            # Block 1: 512 -> 256
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 2: 256 -> 256
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 3: 128 -> 128
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 4: 64 -> 3
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        return decoder

    def load_param(self, args, vgg, decoder, fcs):
        """加载模型参数并冻结
        
        加载预训练的编码器、解码器和全连接层权重，
        并设置所有参数为不需要梯度（冻结）。
        
        Args:
            args: 配置参数，包含权重文件路径
            vgg: VGG16 编码器
            decoder: 解码器
            fcs: 全连接层列表 [fc1, fc2]
            
        Returns:
            tuple: (vgg, decoder, fcs) 加载权重后的模型
            
        加载的权重文件:
            - decoder_path: 解码器权重
            - encoder_path: 编码器权重（包含 'model' 键）
            - fc1: 均值预测层权重
            - fc2: 标准差预测层权重
        """
        # 冻结所有参数
        for param in vgg.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        for i in range(len(fcs)):
            for param in fcs[i].parameters():
                param.requires_grad = False
        
        # 加载预训练权重
        decoder.load_state_dict(torch.load(args.decoder_path))
        vgg.load_state_dict(torch.load(args.encoder_path)['model'])
        # 只保留前19层（去掉最后的 maxpool）
        vgg = nn.Sequential(*list(vgg.children())[:19])
        fcs[0].load_state_dict(torch.load(args.fc1))
        fcs[1].load_state_dict(torch.load(args.fc2))
        
        return vgg, decoder, fcs
