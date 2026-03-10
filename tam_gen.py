# coding=utf-8
"""
TAM (Target Augmentation Module) 图像生成脚本

用于加载预训练的 TAM 模型并对输入图像进行风格迁移，生成风格化图像。

用法:
    python tam_gen.py --input <输入图像或目录> --output <输出目录>
    
示例:
    python tam_gen.py --input ./datasets/cityscape_yolo/images/val --output ./check/tam_gen
    python tam_gen.py --input image.jpg --output ./check/tam_gen
"""

import argparse
import os
import sys
from pathlib import Path
import cv2

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from TargetAugment.enhance_vgg16 import enhance_vgg16
from TargetAugment.enhance_style import get_style_images



# ImageNet 像素均值 (BGR 顺序)
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='使用 TAM 模型生成风格迁移图像',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 输入输出参数
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入图像路径或包含图像的目录')
    parser.add_argument('--output', '-o', type=str, default='./check/tam_gen',
                        help='输出目录路径')
    
    # 模型权重路径（使用 readme.md 中的默认值）
    parser.add_argument('--encoder_path', type=str,
                        default='TargetAugment_train/pre_trained/vgg16_ori.pth',
                        help='VGG16 编码器权重路径')
    parser.add_argument('--decoder_path', type=str,
                        default='TargetAugment_train/models/city2foggy/decoder_iter_160000.pth',
                        help='解码器权重路径')
    parser.add_argument('--fc1', type=str,
                        default='TargetAugment_train/models/city2foggy/fc1_iter_160000.pth',
                        help='fc1 权重路径（均值预测）')
    parser.add_argument('--fc2', type=str,
                        default='TargetAugment_train/models/city2foggy/fc2_iter_160000.pth',
                        help='fc2 权重路径（标准差预测）')
    parser.add_argument('--style_path', type=str,
                        default='./TargetAugment_train/data/meanfoggy/meanfoggy.jpg',
                        help='风格图像路径')
    
    # 风格迁移参数
    parser.add_argument('--style_add_alpha', type=float, default=0.4,
                        help='风格迁移强度 (0.0-1.0)，值越大风格越明显')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='目标图像尺寸（短边）')
    parser.add_argument('--random_style', action='store_true',
                        help='使用随机风格（默认使用固定风格图像）')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='0',
                        help='使用的设备，如 "0" 表示 GPU 0，"cpu" 表示 CPU')
    parser.add_argument('--cuda', action='store_true', default=None,
                        help='使用 CUDA（默认自动检测）')
    
    # 其他参数
    parser.add_argument('--batch', '-b', type=int, default=8,
                        help='批处理大小（每次处理多少张图像）')
    parser.add_argument('--log_dir', type=str, default='./check/tam_gen_logs',
                        help='日志目录')
    parser.add_argument('--compile', action='store_true',
                        help='使用 torch.compile 编译 TAM 模型以加速推理')
    
    args = parser.parse_args()
    
    # 自动检测 CUDA
    if args.cuda is None:
        args.cuda = torch.cuda.is_available() and args.device != 'cpu'
    
    # 设置 compile_tam 属性（供 enhance_base 使用）
    args.compile_tam = args.compile
    
    # 为兼容性设置 style_add_alpha
    if not hasattr(args, 'style_add_alpha'):
        args.style_add_alpha = 0.4
    
    return args


def load_image(image_path, target_size=640):
    """
    加载并预处理图像
    
    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸（短边）
        
    Returns:
        tuple: (预处理后的图像张量 [C, H, W], 原始图像 PIL 对象, 缩放比例)
    """
    # 打开图像
    im = Image.open(image_path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    
    original_im = im.copy()
    im = np.array(im)
    
    # RGB -> BGR
    im = im[:, :, ::-1]
    im = im.astype(np.float32, copy=False)
    
    # 减去像素均值
    im -= PIXEL_MEANS
    
    # 计算缩放比例
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    
    # 缩放图像
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    
    # 转换为张量 [C, H, W]
    im_tensor = torch.from_numpy(im).permute(2, 0, 1).contiguous().float()
    
    return im_tensor, original_im, im_scale


def save_image(tensor, output_path, original_size=None):
    """
    保存风格化后的图像
    
    Args:
        tensor: 风格化后的图像张量 [C, H, W] 或 [B, C, H, W]
        output_path: 输出路径
        original_size: 原始尺寸 (width, height)，用于恢复原始大小
    """
    # 确保张量在 CPU 上
    tensor = tensor.cpu()
    
    # 处理可能的5维张量 [1, 1, C, H, W] -> [C, H, W]
    while tensor.dim() > 3:
        tensor = tensor.squeeze(0)
    
    # 处理4维张量 [1, C, H, W] -> [C, H, W]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 确保维度是 [C, H, W]
    if tensor.dim() != 3:
        raise ValueError(f"意外的张量维度: {tensor.dim()}, 期望 3 (C, H, W)")
    
    # 转换维度顺序 [C, H, W] -> [H, W, C]
    im = tensor.permute(1, 2, 0).numpy()
    
    # 加回像素均值
    im = im + PIXEL_MEANS
    
    # 裁剪到有效范围
    im = np.clip(im, 0, 255).astype(np.uint8)
    
    # BGR -> RGB
    im = im[:, :, ::-1]
    
    # 如果需要，恢复原始尺寸
    if original_size is not None:
        import cv2
        im = cv2.resize(im, original_size, interpolation=cv2.INTER_LINEAR)
    
    # 保存图像
    Image.fromarray(im).save(output_path)


def get_image_files(input_path):
    """
    获取输入路径中的所有图像文件
    
    Args:
        input_path: 输入路径（文件或目录）
        
    Returns:
        list: 图像文件路径列表
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() in valid_extensions:
            return [str(input_path)]
        else:
            raise ValueError(f"不支持的图像格式: {input_path.suffix}")
    
    elif input_path.is_dir():
        image_files = []
        for ext in valid_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        return sorted([str(f) for f in image_files])
    
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.cuda and args.device != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    model_files = {
        'encoder': args.encoder_path,
        'decoder': args.decoder_path,
        'fc1': args.fc1,
        'fc2': args.fc2,
        'style': args.style_path
    }
    
    for name, path in model_files.items():
        if not os.path.exists(path):
            print(f"警告: {name} 文件不存在: {path}")
            if name != 'style':
                return 1
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取输入图像列表
    try:
        image_files = get_image_files(args.input)
    except (ValueError, FileNotFoundError) as e:
        print(f"错误: {e}")
        return 1
    
    if not image_files:
        print(f"在 {args.input} 中未找到图像文件")
        return 1
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 加载 TAM 模型
    print("正在加载 TAM 模型...")
    try:
        adain = enhance_vgg16(args)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return 1
    
    print("模型加载完成，开始生成风格化图像...")
    
    # 分批处理图像
    batch_size = args.batch
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="批次处理"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        try:
            # 加载批次图像
            batch_tensors = []
            batch_original_sizes = []
            batch_names = []
            
            for image_path in batch_files:
                try:
                    im_tensor, original_im, scale = load_image(image_path, args.imgsz)
                    batch_tensors.append(im_tensor)
                    batch_original_sizes.append(original_im.size)  # (width, height)
                    batch_names.append(Path(image_path).stem)
                except Exception as e:
                    print(f"\n加载 {image_path} 时出错: {e}")
                    continue
            
            if not batch_tensors:
                continue
            
            # 将 batch 内的图像 padding 到相同尺寸
            max_h = max(t.shape[1] for t in batch_tensors)
            max_w = max(t.shape[2] for t in batch_tensors)
            
            padded_tensors = []
            for tensor in batch_tensors:
                c, h, w = tensor.shape
                if h < max_h or w < max_w:
                    padded = torch.zeros(c, max_h, max_w, dtype=tensor.dtype)
                    padded[:, :h, :w] = tensor
                    padded_tensors.append(padded)
                else:
                    padded_tensors.append(tensor)
            
            # 合并成 batch: [B, C, H, W]
            im_batch = torch.stack(padded_tensors, dim=0)
            
            # 移动到设备
            if args.cuda:
                im_batch = im_batch.cuda()
            
            # 应用风格迁移
            with torch.no_grad():
                styled_batch = get_style_images(im_batch, adain)
            
            # 分别保存每张图像
            for i, (name, original_size) in enumerate(zip(batch_names, batch_original_sizes)):
                output_path = output_dir / f"{name}_styled.jpg"
                # 裁剪回原始尺寸
                styled_tensor = styled_batch[i]
                save_image(styled_tensor, output_path, original_size)
            
        except Exception as e:
            print(f"\n处理批次 {batch_idx + 1}/{num_batches} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n完成！生成的图像保存在: {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
