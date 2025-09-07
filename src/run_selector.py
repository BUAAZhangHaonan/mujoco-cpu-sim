#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速运行电子元件3D模型筛选器

用法示例：
python3 run_selector.py --threshold 0.8 1.5 --aspect-ratio 2.5 --output assets_0825_0.001/mesh
python3 run_selector.py --help
"""

import argparse
import sys
from pathlib import Path
from src.tools.obj_selector import ComponentSelector


def main():
    parser = argparse.ArgumentParser(
        description='电子元件3D模型筛选器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s                                    # 使用默认参数运行
  %(prog)s --threshold 0.3 2.5                # 设置自定义尺寸阈值
  %(prog)s --aspect-ratio 3.0                 # 设置长宽高比例阈值
  %(prog)s --output assets_custom/mesh        # 设置自定义输出目录
  %(prog)s --threshold 0.8 1.5 --aspect-ratio 2.0 --output my_parts/
        '''
    )

    parser.add_argument(
        '--threshold',
        nargs=2,
        type=float,
        default=[0.8, 1.5],
        metavar=('MIN', 'MAX'),
        help='尺寸阈值比例 [最小倍数, 最大倍数] (默认: 0.8 1.5)'
    )

    parser.add_argument(
        '--aspect-ratio',
        type=float,
        default=2.5,
        metavar='RATIO',
        help='长宽高比例阈值，最长边不超过最短边的倍数 (默认: 2.5)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='assets_0825/mesh',
        metavar='DIR',
        help='输出目录路径 (默认: assets_0824/mesh)'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default='/home/fuyx/zhn/mujoco-cpu-sim/electronic_components_3d_models_coacd_0803',
        metavar='DIR',
        help='电子元件基础目录 (默认: electronic_components_3d_models_coacd_0803)'
    )

    parser.add_argument(
        '--reference-dir',
        type=str,
        default='/home/fuyx/zhn/mujoco-cpu-sim/electronic_components_3d_models_coacd_0803/reference',
        metavar='DIR',
        help='参考零件目录 (默认: electronic_components_3d_models_coacd_0803/reference)'
    )

    args = parser.parse_args()

    # 验证输入参数
    if args.threshold[0] >= args.threshold[1]:
        print("错误: 最小阈值必须小于最大阈值")
        sys.exit(1)

    if args.threshold[0] <= 0 or args.threshold[1] <= 0:
        print("错误: 尺寸阈值必须为正数")
        sys.exit(1)
        
    if args.aspect_ratio <= 1.0:
        print("错误: 长宽高比例阈值必须大于1.0")
        sys.exit(1)

    # 转换为绝对路径
    workspace = Path('/home/fuyx/zhn/mujoco-cpu-sim')
    base_dir = Path(args.base_dir) if Path(args.base_dir).is_absolute() else workspace / args.base_dir
    reference_dir = Path(args.reference_dir) if Path(args.reference_dir).is_absolute() else workspace / args.reference_dir
    target_dir = Path(args.output) if Path(args.output).is_absolute() else workspace / args.output

    # 检查目录是否存在
    if not base_dir.exists():
        print(f"错误: 基础目录不存在: {base_dir}")
        sys.exit(1)

    if not reference_dir.exists():
        print(f"错误: 参考目录不存在: {reference_dir}")
        sys.exit(1)

    # 显示配置信息
    print("=" * 60)
    print("电子元件3D模型筛选器")
    print("=" * 60)
    print(f"基础目录: {base_dir}")
    print(f"参考目录: {reference_dir}")
    print(f"输出目录: {target_dir}")
    print(f"尺寸阈值范围: {args.threshold[0]} ~ {args.threshold[1]} 倍")
    print(f"长宽高比例阈值: {args.aspect_ratio}")
    print("=" * 60)

    # 创建筛选器并执行
    try:
        selector = ComponentSelector(
            str(base_dir), 
            str(reference_dir), 
            str(target_dir), 
            args.threshold, 
            args.aspect_ratio
        )
        stats = selector.filter_components()
        selector.print_summary(stats)

        print("\n✓ 筛选完成!")

    except KeyboardInterrupt:
        print("\n⚠ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()