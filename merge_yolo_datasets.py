#!/usr/bin/env python3
"""
合并多个 YOLO 格式的数据集。

特性：
1. 检查所有数据集的类别标签和编码一致性，不一致则拒绝合并
2. 优先使用硬链接（hard link）而非复制，节省磁盘空间
3. 自动处理文件名冲突
4. 生成合并后的 data.yaml

用法：
    python merge_yolo_datasets.py dataset1/ dataset2/ ... -o output_dir/
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import yaml


def load_yaml(yaml_path: Path) -> dict:
    """加载 YAML 文件，处理可能的编码问题"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(yaml_path, 'r', encoding='gbk') as f:
            return yaml.safe_load(f)


def get_dataset_classes(dataset_path: Path) -> Tuple[Dict[int, str], List[str], Path]:
    """
    获取数据集的类别信息
    
    返回：
        - idx_to_name: 索引到名称的映射
        - names_list: 名称列表（保持顺序）
        - yaml_path: data.yaml 的路径
    """
    # 查找 data.yaml
    yaml_candidates = [
        dataset_path / 'data.yaml',
        dataset_path / 'dataset.yaml',
        dataset_path / 'cityscapes.yaml',
    ]
    
    yaml_path = None
    for cand in yaml_candidates:
        if cand.exists():
            yaml_path = cand
            break
    
    if yaml_path is None:
        # 搜索任何 .yaml 文件
        yaml_files = list(dataset_path.glob('*.yaml'))
        if yaml_files:
            yaml_path = yaml_files[0]
    
    if yaml_path is None:
        raise ValueError(f"在 {dataset_path} 中未找到 data.yaml 或类似的 YAML 文件")
    
    data = load_yaml(yaml_path)
    
    if 'names' not in data:
        raise ValueError(f"{yaml_path} 中没有 'names' 字段")
    
    names = data['names']
    
    # 处理不同的 names 格式（字典或列表）
    if isinstance(names, dict):
        idx_to_name = {int(k): v for k, v in names.items()}
        names_list = [idx_to_name[i] for i in range(len(names))]
    elif isinstance(names, list):
        idx_to_name = {i: name for i, name in enumerate(names)}
        names_list = names
    else:
        raise ValueError(f"{yaml_path} 中的 'names' 格式不正确")
    
    return idx_to_name, names_list, yaml_path


def validate_datasets_consistency(dataset_paths: List[Path]) -> Tuple[List[str], Path]:
    """
    验证所有数据集的类别一致性
    
    返回：
        - common_names: 共同的类别名称列表
        - reference_yaml: 参考 YAML 文件路径（用于复制其他配置）
    
    如果类别不一致，则抛出异常
    """
    if len(dataset_paths) < 1:
        raise ValueError("至少需要提供一个数据集路径")
    
    print("=" * 80)
    print("🔍 检查数据集类别一致性")
    print("=" * 80)
    
    dataset_infos = []
    for i, path in enumerate(dataset_paths):
        print(f"\n[{i+1}/{len(dataset_paths)}] 检查: {path}")
        try:
            idx_to_name, names_list, yaml_path = get_dataset_classes(path)
            dataset_infos.append({
                'path': path,
                'idx_to_name': idx_to_name,
                'names_list': names_list,
                'yaml_path': yaml_path,
                'num_classes': len(names_list)
            })
            print(f"    ✅ 类别数: {len(names_list)}")
            print(f"    ✅ 类别: {names_list}")
        except Exception as e:
            print(f"    ❌ 错误: {e}")
            raise
    
    # 检查类别数量是否一致
    num_classes_set = {info['num_classes'] for info in dataset_infos}
    if len(num_classes_set) > 1:
        print("\n" + "=" * 80)
        print("❌ 类别数量不一致！")
        print("=" * 80)
        for info in dataset_infos:
            print(f"  {info['path']}: {info['num_classes']} 类")
        raise ValueError("所有数据集的类别数量必须相同")
    
    # 检查类别名称和顺序是否一致
    reference_names = dataset_infos[0]['names_list']
    inconsistencies = []
    
    for info in dataset_infos[1:]:
        current_names = info['names_list']
        if current_names != reference_names:
            # 找出差异
            for idx, (ref, cur) in enumerate(zip(reference_names, current_names)):
                if ref != cur:
                    inconsistencies.append({
                        'dataset': info['path'],
                        'index': idx,
                        'reference': ref,
                        'current': cur
                    })
    
    if inconsistencies:
        print("\n" + "=" * 80)
        print("❌ 类别标签或编码不一致！")
        print("=" * 80)
        print("\n参考数据集（第一个）:")
        print(f"  路径: {dataset_infos[0]['path']}")
        for idx, name in enumerate(reference_names):
            print(f"    [{idx}] {name}")
        
        print("\n发现不一致:")
        for inc in inconsistencies:
            print(f"  数据集: {inc['dataset']}")
            print(f"    索引 [{inc['index']}]:")
            print(f"      参考: {inc['reference']}")
            print(f"      当前: {inc['current']}")
        
        raise ValueError("所有数据集的类别名称和编码顺序必须完全一致")
    
    print("\n" + "=" * 80)
    print("✅ 所有数据集类别一致性检查通过！")
    print(f"✅ 类别数: {len(reference_names)}")
    print(f"✅ 类别列表: {reference_names}")
    print("=" * 80)
    
    return reference_names, dataset_infos[0]['yaml_path']


def create_hardlink(src: Path, dst: Path, prefer_hardlink: bool = True) -> bool:
    """
    创建硬链接，如果失败则复制
    
    返回：
        - 是否使用了硬链接（False 表示使用了复制）
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    if prefer_hardlink:
        try:
            # 尝试创建硬链接
            os.link(src, dst)
            return True
        except (OSError, FileExistsError):
            # 硬链接失败（可能是跨文件系统），回退到复制
            pass
    
    # 使用复制
    shutil.copy2(src, dst)
    return False


def merge_split(
    dataset_infos: List[dict],
    split: str,
    output_path: Path,
    prefer_hardlink: bool = True
) -> Tuple[int, int]:
    """
    合并一个特定的数据分割（train/val/test）
    
    返回：
        - (硬链接数量, 复制数量)
    """
    print(f"\n📁 处理 '{split}' 分割...")
    
    hardlink_count = 0
    copy_count = 0
    total_images = 0
    
    output_images_dir = output_path / 'images' / split
    output_labels_dir = output_path / 'labels' / split
    
    # 收集所有文件，处理可能的命名冲突
    used_basenames: Set[str] = set()
    
    for info in dataset_infos:
        dataset_path = info['path']
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        if not images_dir.exists():
            print(f"  ⚠️  {dataset_path} 中没有 '{split}' 分割，跳过")
            continue
        
        # 获取所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f'*{ext}'))
            image_files.extend(images_dir.glob(f'*{ext.upper()}'))
        
        print(f"  📂 {dataset_path.name}: 找到 {len(image_files)} 张图片")
        
        for img_path in image_files:
            # 找到对应的标签文件
            label_path = labels_dir / (img_path.stem + '.txt')
            
            if not label_path.exists():
                # 尝试其他命名方式
                alt_label_paths = [
                    labels_dir / (img_path.stem.lower() + '.txt'),
                    labels_dir / (img_path.stem.upper() + '.txt'),
                ]
                for alt_path in alt_label_paths:
                    if alt_path.exists():
                        label_path = alt_path
                        break
            
            # 处理文件名冲突
            basename = img_path.stem
            counter = 0
            original_basename = basename
            while basename in used_basenames:
                counter += 1
                basename = f"{original_basename}_dup{counter}"
            used_basenames.add(basename)
            
            # 确定输出文件名
            new_img_name = basename + img_path.suffix
            new_label_name = basename + '.txt'
            
            output_img_path = output_images_dir / new_img_name
            output_label_path = output_labels_dir / new_label_name
            
            # 创建硬链接或复制图片
            if create_hardlink(img_path, output_img_path, prefer_hardlink):
                hardlink_count += 1
            else:
                copy_count += 1
            
            # 创建硬链接或复制标签（如果存在）
            if label_path.exists():
                create_hardlink(label_path, output_label_path, prefer_hardlink)
            
            total_images += 1
    
    print(f"  ✅ '{split}' 完成: {total_images} 张图片 (硬链接: {hardlink_count}, 复制: {copy_count})")
    return hardlink_count, copy_count


def create_output_yaml(
    output_path: Path,
    names: List[str],
    reference_yaml: Path,
    dataset_infos: List[dict]
) -> None:
    """创建合并后的 data.yaml 文件"""
    
    # 加载参考 YAML 获取其他配置
    ref_data = load_yaml(reference_yaml)
    
    # 构建新的配置
    output_data = {
        'names': names,
        'nc': len(names),
        'path': str(output_path.absolute()),
    }
    
    # 检查各个分割是否存在
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = output_path / 'images' / split
        if split_dir.exists() and any(split_dir.iterdir()):
            output_data[split] = str((output_path / 'images' / split).absolute())
    
    # 复制其他可能的配置（如 roboflow 信息等）
    for key in ['roboflow', 'license', 'description', 'url']:
        if key in ref_data:
            output_data[key] = ref_data[key]
    
    # 添加合并信息
    output_data['merged_from'] = [str(info['path']) for info in dataset_infos]
    output_data['merge_date'] = str(Path(__file__).stat().st_mtime)
    
    # 保存 YAML
    output_yaml_path = output_path / 'data.yaml'
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, allow_unicode=True, sort_keys=False)
    
    print(f"\n📝 生成配置文件: {output_yaml_path}")


def calculate_space_savings(output_path: Path) -> None:
    """计算并显示节省的磁盘空间"""
    print("\n" + "=" * 80)
    print("💾 磁盘空间统计")
    print("=" * 80)
    
    total_size = 0
    linked_size = 0
    
    for root, dirs, files in os.walk(output_path):
        for filename in files:
            filepath = Path(root) / filename
            try:
                stat = os.stat(filepath)
                file_size = stat.st_size
                total_size += file_size
                
                # 检查硬链接数（nlink > 1 表示是硬链接）
                if stat.st_nlink > 1:
                    linked_size += file_size
            except (OSError, FileNotFoundError):
                pass
    
    total_mb = total_size / (1024 * 1024)
    linked_mb = linked_size / (1024 * 1024)
    saved_mb = linked_mb
    
    print(f"  总数据大小: {total_mb:.2f} MB")
    print(f"  硬链接数据: {linked_mb:.2f} MB")
    print(f"  节省空间:   {saved_mb:.2f} MB ({100*saved_mb/total_mb:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='合并多个 YOLO 格式的数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合并两个数据集
  python merge_yolo_datasets.py /path/to/dataset1 /path/to/dataset2 -o merged/
  
  # 合并多个数据集，强制复制（不使用硬链接）
  python merge_yolo_datasets.py ds1/ ds2/ ds3/ -o merged/ --no-hardlink
  
  # 验证类别一致性但不合并
  python merge_yolo_datasets.py ds1/ ds2/ --dry-run
        """
    )
    
    parser.add_argument(
        'datasets',
        nargs='+',
        type=Path,
        help='要合并的数据集目录路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--no-hardlink',
        action='store_true',
        help='禁用硬链接，使用复制（默认优先使用硬链接）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅验证类别一致性，不执行合并'
    )
    
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        choices=['train', 'val', 'test'],
        help='要合并的数据分割（默认: train val test）'
    )
    
    args = parser.parse_args()
    
    # 验证输入路径
    for dataset_path in args.datasets:
        if not dataset_path.exists():
            print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
            sys.exit(1)
        if not dataset_path.is_dir():
            print(f"❌ 错误: 不是目录: {dataset_path}")
            sys.exit(1)
    
    # 检查类别一致性
    try:
        common_names, reference_yaml = validate_datasets_consistency(args.datasets)
    except ValueError as e:
        print(f"\n❌ 验证失败: {e}")
        sys.exit(1)
    
    if args.dry_run:
        print("\n✅ 类别一致性检查通过（干运行模式，不执行合并）")
        sys.exit(0)
    
    # 检查输出目录
    if args.output.exists():
        if any(args.output.iterdir()):
            print(f"\n⚠️ 警告: 输出目录已存在且非空: {args.output}")
            response = input("是否继续? 这将可能覆盖现有文件 [y/N]: ")
            if response.lower() != 'y':
                print("已取消")
                sys.exit(0)
    else:
        args.output.mkdir(parents=True)
    
    # 重新获取数据集信息用于合并
    dataset_infos = []
    for path in args.datasets:
        idx_to_name, names_list, yaml_path = get_dataset_classes(path)
        dataset_infos.append({
            'path': path,
            'idx_to_name': idx_to_name,
            'names_list': names_list,
            'yaml_path': yaml_path
        })
    
    # 执行合并
    print("\n" + "=" * 80)
    print("🚀 开始合并数据集")
    print("=" * 80)
    print(f"输出目录: {args.output}")
    print(f"硬链接模式: {'禁用' if args.no_hardlink else '启用'}")
    print(f"处理分割: {', '.join(args.splits)}")
    
    prefer_hardlink = not args.no_hardlink
    total_hardlinks = 0
    total_copies = 0
    
    for split in args.splits:
        h, c = merge_split(dataset_infos, split, args.output, prefer_hardlink)
        total_hardlinks += h
        total_copies += c
    
    # 创建配置文件
    create_output_yaml(args.output, common_names, reference_yaml, dataset_infos)
    
    # 统计信息
    print("\n" + "=" * 80)
    print("📊 合并完成统计")
    print("=" * 80)
    print(f"  总文件数: {total_hardlinks + total_copies}")
    print(f"  硬链接:   {total_hardlinks}")
    print(f"  复制:     {total_copies}")
    
    if not args.no_hardlink:
        calculate_space_savings(args.output)
    
    print("\n✅ 数据集合并完成！")
    print(f"📁 输出目录: {args.output}")
    print(f"📝 配置文件: {args.output / 'data.yaml'}")


if __name__ == '__main__':
    main()
