#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate TAM style-augmented sample images for SF-YOLO.

This script generates style-augmented images using the Target Augmentation Module (TAM)
to visualize the effect of neural style transfer on foggy cityscape images.

Example:
    python generate_tam_samples.py --num_samples 5 --output_dir samples
"""

import argparse
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate TAM style-augmented images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 samples with default settings
  python generate_tam_samples.py --num_samples 5

  # Custom alpha and output directory
  python generate_tam_samples.py --style_add_alpha 0.3 --output_dir my_samples

  # Use CPU instead of CUDA
  python generate_tam_samples.py --device cpu --num_samples 3
        """
    )
    
    # Model weights
    parser.add_argument('--decoder_path', type=str,
                        default='TargetAugment_train/models/city2foggy/decoder_iter_160000.pth',
                        help='Path to decoder weights (default: city2foggy decoder)')
    parser.add_argument('--encoder_path', type=str,
                        default='TargetAugment_train/pre_trained/vgg16_ori.pth',
                        help='Path to VGG16 encoder weights')
    parser.add_argument('--fc1', type=str,
                        default='TargetAugment_train/models/city2foggy/fc1_iter_160000.pth',
                        help='Path to fc1 weights')
    parser.add_argument('--fc2', type=str,
                        default='TargetAugment_train/models/city2foggy/fc2_iter_160000.pth',
                        help='Path to fc2 weights')
    parser.add_argument('--style_path', type=str,
                        default='TargetAugment_train/data/meanfoggy/meanfoggy.jpg',
                        help='Path to style image')
    
    # Generation parameters
    parser.add_argument('--style_add_alpha', type=float, default=0.4,
                        help='Style transfer intensity 0.0-1.0 (default: 0.4)')
    parser.add_argument('--source_images', type=str,
                        default='datasets/CityScapesFoggy/yolov5_format/images/val',
                        help='Source images directory')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate (default: 5)')
    parser.add_argument('--output_dir', type=str, default='tam_samples',
                        help='Output directory (default: tam_samples)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')
    
    # TAM internal parameters
    parser.add_argument('--save_style_samples', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--random_style', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--log_dir', type=str, default='runs/style_samples',
                        help=argparse.SUPPRESS)
    
    opt = parser.parse_args()
    
    # Set internal flags
    opt.cuda = (opt.device == 'cuda')
    
    return opt


def check_resources(opt):
    """Check if all required resources exist."""
    required_files = {
        'decoder': opt.decoder_path,
        'encoder': opt.encoder_path,
        'fc1': opt.fc1,
        'fc2': opt.fc2,
        'style image': opt.style_path,
    }
    
    missing = []
    for name, path in required_files.items():
        if not Path(path).exists():
            missing.append(f"  - {name}: {path}")
    
    if missing:
        print("ERROR: Missing required files:")
        print('\n'.join(missing))
        print("\nPlease download the TAM weights or train Stage 1 first.")
        print("See: https://github.com/your-repo/sf-yolo#stage-1-train-target-augmentation-module")
        return False
    
    return True


def load_image(path, size=(640, 640)):
    """Load and resize image."""
    img = Image.open(path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    return np.array(img)


def create_comparison(original, styled, alpha, img_name):
    """Create side-by-side comparison image."""
    h, w = original.shape[:2]
    
    # Convert to PIL
    orig_pil = Image.fromarray(original)
    styled_pil = Image.fromarray(styled)
    
    # Create comparison image with labels
    comp_w = w * 2
    comp_h = h + 40
    comparison = Image.new('RGB', (comp_w, comp_h), (255, 255, 255))
    
    # Paste images
    comparison.paste(orig_pil, (0, 40))
    comparison.paste(styled_pil, (w, 40))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((w//2 - 80, 10), "Original (Foggy)", fill=(0, 0, 0), font=font)
    draw.text((w + w//2 - 140, 10), f"TAM Augmented (alpha={alpha})", fill=(0, 0, 0), font=font)
    
    return comparison, orig_pil, styled_pil


def main():
    """Main function."""
    opt = parse_opt()
    
    # Check resources
    if not check_resources(opt):
        return 1
    
    # Create output directory
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TAM Style Augmentation Sample Generator")
    print("=" * 60)
    print(f"Device: {opt.device}")
    print(f"Style alpha: {opt.style_add_alpha}")
    print(f"Output dir: {output_dir.absolute()}")
    print("=" * 60)
    
    # Import TAM modules (requires SF-YOLO project structure)
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from TargetAugment.enhance_vgg16 import enhance_vgg16
        from TargetAugment.enhance_style import get_style_images
    except ImportError as e:
        print(f"ERROR: Cannot import TAM modules: {e}")
        print("Make sure you're running this script from the SF-YOLO project.")
        return 1
    
    # Initialize TAM
    print("\n[1/3] Loading TAM (Target Augmentation Module)...")
    try:
        adain = enhance_vgg16(opt)
        print("TAM loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load TAM: {e}")
        return 1
    
    # Get source images
    print(f"\n[2/3] Scanning source images from: {opt.source_images}")
    image_dir = Path(opt.source_images)
    
    if not image_dir.exists():
        print(f"ERROR: Directory not found: {opt.source_images}")
        return 1
    
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {opt.source_images}")
        return 1
    
    print(f"Found {len(image_files)} images")
    
    # Randomly select samples
    random.seed(42)
    selected = random.sample(image_files, min(opt.num_samples, len(image_files)))
    print(f"Selected {len(selected)} samples")
    
    # Generate style-augmented images
    print(f"\n[3/3] Generating style-augmented images...")
    
    for idx, img_path in enumerate(selected, 1):
        print(f"  Processing {idx}/{len(selected)}: {img_path.name}")
        
        try:
            # Load original image
            original_img = load_image(img_path)
            
            # Prepare for TAM
            img_tensor = torch.from_numpy(original_img).float().permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(opt.device)
            
            # Apply style augmentation
            with torch.no_grad():
                styled_tensor = get_style_images(img_tensor, opt, adain)
            
            # Convert back to numpy
            styled_img = styled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            # Create comparison
            comparison, orig_pil, styled_pil = create_comparison(
                original_img, styled_img, opt.style_add_alpha, img_path.name
            )
            
            # Save all versions
            comparison.save(output_dir / f"{idx:02d}_comparison_{img_path.stem}.png")
            orig_pil.save(output_dir / f"{idx:02d}_original.png")
            styled_pil.save(output_dir / f"{idx:02d}_tam_augmented.png")
            
            print(f"    Saved: {idx:02d}_comparison_{img_path.stem}.png")
            
        except Exception as e:
            print(f"    ERROR processing {img_path.name}: {e}")
            continue
    
    # Save style image for reference
    try:
        style_img = Image.open(opt.style_path)
        style_img.save(output_dir / "00_reference_style_image.jpg")
    except Exception as e:
        print(f"Warning: Could not save style image: {e}")
    
    print(f"\nDone! Generated {len(selected)} sample pairs.")
    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
