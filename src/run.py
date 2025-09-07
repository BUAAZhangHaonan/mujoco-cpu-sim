"""
批量仿真脚本（src 入口）
通过子进程调用 `python -m src.main` 实现批量仿真。
"""

import argparse
import glob
import os
import subprocess
import sys
from typing import List
import math


class BatchSimulationRunner:
    """批量仿真运行器"""

    def __init__(self):
        # 以模块方式运行 src.main，确保基于 src 布局
        self.main_module = "src.main"

    def validate_weights(self, weights: List[float]) -> bool:
        total = sum(weights)
        return abs(total - 1.0) <= 1e-6

    def calculate_batch_distribution(self, total_batch: int, weights: List[float]) -> List[int]:
        if not self.validate_weights(weights):
            raise ValueError("权重列表不合法，和必须为1.0")
        batches = []
        allocated = 0
        for weight in weights[:-1]:
            batch_count = int(math.floor(total_batch * weight))
            batches.append(batch_count)
            allocated += batch_count
        batches.append(total_batch - allocated)
        return batches

    def find_replicate_xml_files(self, scenes_dir: str) -> List[str]:
        pattern = os.path.join(scenes_dir, "**", "*_replicate.xml")
        return sorted(glob.glob(pattern, recursive=True))

    def build_main_command(self, xml_file: str, replicate_count: int, batch: int, extra_args: List[str]) -> List[str]:
        # 用 -m 方式调用，避免直接路径依赖
        cmd = [sys.executable, "-m", self.main_module, "--model_path", xml_file, "--replicate-count", str(replicate_count), "--batch", str(batch)]
        cmd.extend(extra_args)
        return cmd

    def run_single_simulation(self, xml_file: str, replicate_count: int, batch: int, extra_args: List[str]) -> bool:
        cmd = self.build_main_command(xml_file, replicate_count, batch, extra_args)
        print("\n" + "=" * 80)
        print("运行仿真:")
        print(f"  文件: {xml_file}")
        print(f"  Replicate数量: {replicate_count}")
        print(f"  场景数: {batch}")
        print(f"  命令: {' '.join(cmd)}")
        print("=" * 80)
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ 仿真完成: {xml_file} (replicate={replicate_count}, batch={batch})")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ 仿真失败: {xml_file} (replicate={replicate_count}, batch={batch})，返回码 {e.returncode}")
            return False
        except Exception as e:
            print(f"✗ 仿真异常: {xml_file} (replicate={replicate_count}, batch={batch})，错误: {e}")
            return False

    def run_batch_simulations(self, scenes_dir: str, total_batch: int, replicate_counts: List[int], replicate_weights: List[float], extra_args: List[str]) -> None:
        if len(replicate_counts) != len(replicate_weights):
            raise ValueError("replicate_counts 和 replicate_weights 长度必须相同")
        if not self.validate_weights(replicate_weights):
            raise ValueError("权重列表不合法，和必须为 1.0")
        xml_files = self.find_replicate_xml_files(scenes_dir)
        if not xml_files:
            print(f"在 {scenes_dir} 中未找到任何 *_replicate.xml 文件")
            return
        print(f"找到 {len(xml_files)} 个XML文件:")
        for xml_file in xml_files:
            print(f"  - {xml_file}")
        batch_distribution = self.calculate_batch_distribution(total_batch, replicate_weights)
        print("\n场景数分配:")
        for count, weight, batch_count in zip(replicate_counts, replicate_weights, batch_distribution):
            print(f"  Replicate {count}: 权重 {weight:.3f} -> {batch_count} 场景")
        print(f"  总计: {sum(batch_distribution)} 场景")
        total_simulations = len(xml_files) * len(replicate_counts)
        completed = 0
        failed = 0
        for xml_file in xml_files:
            for replicate_count, batch_count in zip(replicate_counts, batch_distribution):
                if batch_count <= 0:
                    continue
                ok = self.run_single_simulation(xml_file, replicate_count, batch_count, extra_args)
                completed += int(ok)
                failed += int(not ok)
        print("\n" + "=" * 80)
        print("批量仿真完成")
        print(f"  总仿真数: {total_simulations}")
        print(f"  成功: {completed}")
        print(f"  失败: {failed}")
        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="批量MuJoCo仿真脚本")
    parser.add_argument("--scenes-dir", type=str, required=True, help="场景文件目录路径，将遍历其中所有*_replicate.xml文件")
    parser.add_argument("--batch", type=int, required=True, help="总场景数量，将根据权重分配给不同replicate数量")
    parser.add_argument("--replicate-counts", type=int, nargs="+", required=True, help="replicate数量列表，例如: 10 25 50 100 250")
    parser.add_argument("--replicate-weights", type=float, nargs="+", required=True, help="replicate权重列表，总和必须为1，例如: 0.1 0.2 0.4 0.2 0.1")
    # 透传给 main 的参数（与 ConfigManager 保持一致）
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--part_name", type=str, help="要随机化的零件名前缀")
    parser.add_argument("--steps", type=int, help="每个场景要仿真的步数")
    parser.add_argument("--workers", type=int, help="并行进程数量")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--xy-extent", type=float, nargs=2, help="XY放置范围：[min, max]")
    parser.add_argument("--z-low", type=float, help="第一层的z起始高度")
    parser.add_argument("--layer-gap", type=float, help="不同Z层之间的额外安全间隙")
    parser.add_argument("--min-clearance", type=float, help="不同零件之间的最小XY间隙")
    parser.add_argument("--jitter-frac", type=float, help="单元内XY抖动占间隙比例")
    parser.add_argument("--allow-3d-rot", action="store_true", help="允许三轴随机旋转")
    parser.add_argument("--no-allow-3d-rot", dest="allow_3d_rot", action="store_false", help="禁用三轴随机旋转")
    parser.add_argument("--yaw-only", action="store_true", help="仅绕Z轴旋转")
    parser.add_argument("--fit-mode", choices=["sphere", "aabb"], help="几何拟合模式")
    parser.add_argument("--viewer", action="store_true", help="是否显示可视化界面")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--npy-out-dir", type=str, help=".npy输出目录")
    parser.add_argument("--viz-out-dir", type=str, help="可视化输出目录")
    parser.add_argument("--save-modified-xml", action="store_true", help="保存修改后的XML文件到磁盘（默认仅在内存中修改）")
    return parser.parse_args()


def build_extra_args(args) -> List[str]:
    extra = []
    simple = [
        ("config", "--config"),
        ("part_name", "--part_name"),
        ("steps", "--steps"),
        ("workers", "--workers"),
        ("seed", "--seed"),
        ("z_low", "--z-low"),
        ("layer_gap", "--layer-gap"),
        ("min_clearance", "--min-clearance"),
        ("jitter_frac", "--jitter-frac"),
        ("fit_mode", "--fit-mode"),
        ("output_dir", "--output-dir"),
        ("npy_out_dir", "--npy-out-dir"),
        ("viz_out_dir", "--viz-out-dir"),
    ]
    for attr, flag in simple:
        val = getattr(args, attr, None)
        if val is not None:
            extra.extend([flag, str(val)])
    if args.xy_extent is not None:
        extra.extend(["--xy-extent"] + [str(x) for x in args.xy_extent])
    if args.allow_3d_rot is True:
        extra.append("--allow-3d-rot")
    elif args.allow_3d_rot is False:
        extra.append("--no-allow-3d-rot")
    if args.yaw_only:
        extra.append("--yaw-only")
    if args.viewer:
        extra.append("--viewer")
    if getattr(args, "save_modified_xml", False):
        extra.append("--save-modified-xml")
    return extra


def main():
    args = parse_args()
    if len(args.replicate_counts) != len(args.replicate_weights):
        print("错误: replicate-counts和replicate-weights的长度必须相同")
        sys.exit(1)
    try:
        extra_args = build_extra_args(args)
        runner = BatchSimulationRunner()
        runner.run_batch_simulations(args.scenes_dir, args.batch, args.replicate_counts, args.replicate_weights, extra_args)
    except Exception as e:
        print(f"批量仿真失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
