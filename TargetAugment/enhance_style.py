# coding=utf-8
from TargetAugment.enhance_vgg16 import enhance_vgg16


def get_style_images(im_data, adain: enhance_vgg16):
    """对输入图像应用风格迁移
    
    使用提供的 enhance_vgg16 实例对输入图像进行风格迁移。
    这是一个高层封装函数，简化了风格迁移的调用流程。
    
    Args:
        im_data: 输入图像张量 [N, C, H, W]，通常是批次图像
        adain: enhance_vgg16 实例，预加载了风格迁移模型
        
    Returns:
        torch.Tensor: 风格化后的图像 [N, C, H, W]
        
    注意:
        - 输入图像需要已经过预处理（减去像素均值）
        - 输出图像需要添加像素均值才能正确显示
        - 该函数会调用 adain.add_style(im_data, 0) 进行风格迁移
        
    示例:
        >>> # 假设已经创建好 enhance_vgg16 实例
        >>> styled = get_style_images(batch_images, adain_model)
        >>> # 结果可以直接用于训练或保存
    """
    # 应用风格到图像
    # 这实际上就是直接返回 add_style 的结果
    styled_im_data = im_data * 0 + 1 * adain.add_style(im_data, 0)
    
    return styled_im_data
