#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电子元件3D模型筛选器（迁移到 src/tools 包）
"""

import os
import shutil
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


class ComponentSelector:
    def __init__(self, base_dir: str, reference_dir: str, target_dir: str,
                 size_thresholds: List[float] = [0.5, 2.0], aspect_ratio_threshold: float = 2.5):
        self.base_dir = Path(base_dir)
        self.reference_dir = Path(reference_dir)
        self.target_dir = Path(target_dir)
        self.size_thresholds = size_thresholds
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.source_dirs = [
            self.base_dir / "electronic_components_3d_models_coacd_0803_01~273",
            self.base_dir / "electronic_components_3d_models_coacd_0803_237~474",
        ]

    def parse_obj_file(self, obj_path: str) -> Tuple[float, float, float]:
        if not os.path.exists(obj_path):
            return 0, 0, 0
        vertices = []
        try:
            with open(obj_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('v '):
                        parts = line.split()
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            vertices.append([x, y, z])
        except Exception as e:
            print(f"读取OBJ文件时出错 {obj_path}: {e}")
            return 0, 0, 0
        if not vertices:
            return 0, 0, 0
        vertices = np.array(vertices)
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        dimensions = max_coords - min_coords
        return abs(dimensions[0]), abs(dimensions[1]), abs(dimensions[2])

    def get_max_dimension(self, obj_path: str) -> float:
        w, h, d = self.parse_obj_file(obj_path)
        return max(w, h, d)

    def check_aspect_ratio(self, obj_path: str) -> Tuple[bool, float]:
        w, h, d = self.parse_obj_file(obj_path)
        if w <= 0 or h <= 0 or d <= 0:
            return False, 0
        dims = [w, h, d]
        mx, mn = max(dims), min(dims)
        if mn <= 0:
            return False, 0
        ar = mx / mn
        return ar <= self.aspect_ratio_threshold, ar

    def calculate_reference_average_size(self) -> float:
        reference_sizes = []
        print("正在计算参考零件的平均尺寸...")
        for ref_folder in self.reference_dir.iterdir():
            if ref_folder.is_dir():
                obj_file = ref_folder / f"{ref_folder.name}.obj"
                if obj_file.exists():
                    max_dim = self.get_max_dimension(str(obj_file))
                    if max_dim > 0:
                        reference_sizes.append(max_dim)
                        print(f"  {ref_folder.name}: 最长边 = {max_dim:.4f}")
                else:
                    print(f"  警告: 未找到 {obj_file}")
        if not reference_sizes:
            raise ValueError("未找到有效的参考零件OBJ文件")
        avg = float(np.mean(reference_sizes))
        print(f"\n参考零件平均最长边尺寸: {avg:.4f}")
        print(f"尺寸阈值范围: [{avg * self.size_thresholds[0]:.4f}, {avg * self.size_thresholds[1]:.4f}]")
        print(f"长宽高比例阈值: {self.aspect_ratio_threshold}")
        return avg

    def find_all_component_folders(self) -> List[Path]:
        component_folders = []
        for source_dir in self.source_dirs:
            if not source_dir.exists():
                print(f"警告: 源目录不存在 {source_dir}")
                continue
            print(f"扫描目录: {source_dir}")
            for root, dirs, files in os.walk(source_dir):
                root_path = Path(root)
                folder_name = root_path.name
                obj_file = root_path / f"{folder_name}.obj"
                if obj_file.exists():
                    component_folders.append(root_path)
        print(f"找到 {len(component_folders)} 个零件文件夹")
        return component_folders

    def copy_component_to_target(self, source_folder: Path, component_name: str) -> bool:
        target_folder = self.target_dir / component_name
        try:
            if target_folder.exists():
                shutil.rmtree(target_folder)
            shutil.copytree(source_folder, target_folder)
            return True
        except Exception as e:
            print(f"复制文件夹时出错 {source_folder} -> {target_folder}: {e}")
            return False

    def filter_components(self) -> Dict[str, int]:
        avg_reference_size = self.calculate_reference_average_size()
        min_size = avg_reference_size * self.size_thresholds[0]
        max_size = avg_reference_size * self.size_thresholds[1]
        component_folders = self.find_all_component_folders()
        stats = {
            'total_found': len(component_folders),
            'within_range': 0,
            'too_small': 0,
            'too_large': 0,
            'aspect_ratio_failed': 0,
            'invalid': 0,
            'copied_success': 0,
            'copied_failed': 0
        }
        print("\n开始筛选零件...")
        print(f"尺寸阈值范围: [{min_size:.4f}, {max_size:.4f}]")
        print(f"长宽高比例阈值: {self.aspect_ratio_threshold}")
        for i, folder in enumerate(component_folders):
            component_name = folder.name
            obj_file = folder / f"{component_name}.obj"
            if (i + 1) % 50 == 0:
                print(f"处理进度: {i+1}/{len(component_folders)}")
            if not obj_file.exists():
                stats['invalid'] += 1
                continue
            max_dim = self.get_max_dimension(str(obj_file))
            if max_dim <= 0:
                stats['invalid'] += 1
                continue
            ok_ar, ar = self.check_aspect_ratio(str(obj_file))
            if max_dim < min_size:
                stats['too_small'] += 1
            elif max_dim > max_size:
                stats['too_large'] += 1
            elif not ok_ar:
                stats['aspect_ratio_failed'] += 1
                print(f"  ✗ 长宽高比例不合格: {component_name} (比例: {ar:.2f}, 最长边: {max_dim:.4f})")
            else:
                stats['within_range'] += 1
                if self.copy_component_to_target(folder, component_name):
                    stats['copied_success'] += 1
                    print(f"  ✓ 复制成功: {component_name} (最长边: {max_dim:.4f}, 比例: {ar:.2f})")
                else:
                    stats['copied_failed'] += 1
                    print(f"  ✗ 复制失败: {component_name}")
        return stats

    def print_summary(self, stats: Dict[str, int]):
        print("\n" + "=" * 60)
        print("筛选结果摘要")
        print("=" * 60)
        print(f"总共找到零件: {stats['total_found']}")
        print(f"符合所有要求: {stats['within_range']}")
        print(f"尺寸太小: {stats['too_small']}")
        print(f"尺寸太大: {stats['too_large']}")
        print(f"长宽高比例不合格: {stats['aspect_ratio_failed']}")
        print(f"无效/无法解析: {stats['invalid']}")
        print(f"成功复制: {stats['copied_success']}")
        print(f"复制失败: {stats['copied_failed']}")
        print(f"\n目标目录: {self.target_dir}")
        print("=" * 60)
