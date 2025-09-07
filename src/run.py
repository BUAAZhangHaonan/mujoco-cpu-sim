"""
批量仿真脚本
通过subprocess调用main.py实现批量仿真支持:
1. 遍历指定文件夹中的所有_replicate.xml文件
2. 设定replicate数量列表和对应权重
3. 根据权重分配每个replicate数量的仿真场景数
4. 透传其他参数给main.py

使用示例：
python3 0821/run.py --scenes-dir assets_0825/mjcf/scenes --batch 4 --workers 4 --replicate-counts 25 50 100 --replicate-weights 0.25 0.5 0.25 --allow-3d-rot --enable-stability-check
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
        self.main_script = "0821/main.py"

    def validate_weights(self, weights: List[float]) -> bool:
        """验证权重列表是否合法（总和为1）"""
        total = sum(weights)
        tolerance = 1e-6
        if abs(total - 1.0) > tolerance:
            print(f"错误: 权重总和为 {total:.6f}，应该为 1.0")
            return False
        return True

    def calculate_batch_distribution(
        self, total_batch: int, weights: List[float]
    ) -> List[int]:
        """根据权重计算每个replicate数量的场景数分配"""
        if not self.validate_weights(weights):
            raise ValueError("权重列表不合法")

        # 计算每个权重对应的场景数
        batches = []
        allocated = 0

        for i, weight in enumerate(weights[:-1]):  # 除了最后一个
            batch_count = int(math.floor(total_batch * weight))
            batches.append(batch_count)
            allocated += batch_count

        # 最后一个分配剩余的场景数，确保总数正确
        batches.append(total_batch - allocated)

        return batches

    def find_replicate_xml_files(self, scenes_dir: str) -> List[str]:
        """查找指定目录下的所有_replicate.xml文件"""
        pattern = os.path.join(scenes_dir, "**", "*_replicate.xml")
        files = glob.glob(pattern, recursive=True)
        return sorted(files)

    def build_main_command(
        self, xml_file: str, replicate_count: int, batch: int, extra_args: List[str]
    ) -> List[str]:
        """构建调用main.py的命令"""
        cmd = [
            sys.executable,  # python3
            self.main_script,
            "--model_path",
            xml_file,
            "--replicate-count",
            str(replicate_count),
            "--batch",
            str(batch),
        ]

        # 添加额外参数
        cmd.extend(extra_args)

        return cmd

    def run_single_simulation(
        self, xml_file: str, replicate_count: int, batch: int, extra_args: List[str]
    ) -> bool:
        """运行单个仿真"""
        cmd = self.build_main_command(xml_file, replicate_count, batch, extra_args)

        print(f"\n{'='*80}")
        print(f"运行仿真:")
        print(f"  文件: {xml_file}")
        print(f"  Replicate数量: {replicate_count}")
        print(f"  场景数: {batch}")
        print(f"  命令: {' '.join(cmd)}")
        print(f"{'='*80}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(
                f"✓ 仿真完成: {xml_file} (replicate={replicate_count}, batch={batch})"
            )
            return True
        except subprocess.CalledProcessError as e:
            print(
                f"✗ 仿真失败: {xml_file} (replicate={replicate_count}, batch={batch})"
            )
            print(f"  错误码: {e.returncode}")
            return False
        except Exception as e:
            print(
                f"✗ 仿真异常: {xml_file} (replicate={replicate_count}, batch={batch})"
            )
            print(f"  错误: {e}")
            return False

    def run_batch_simulations(
        self,
        scenes_dir: str,
        total_batch: int,
        replicate_counts: List[int],
        replicate_weights: List[float],
        extra_args: List[str],
    ) -> None:
        """运行批量仿真"""
        # 验证参数
        if len(replicate_counts) != len(replicate_weights):
            raise ValueError("replicate_counts和replicate_weights长度必须相同")

        if not self.validate_weights(replicate_weights):
            raise ValueError("权重列表不合法")

        # 查找XML文件
        xml_files = self.find_replicate_xml_files(scenes_dir)
        if not xml_files:
            print(f"在 {scenes_dir} 中未找到任何 *_replicate.xml 文件")
            return

        print(f"找到 {len(xml_files)} 个XML文件:")
        for xml_file in xml_files:
            print(f"  - {xml_file}")

        # 计算场景数分配
        batch_distribution = self.calculate_batch_distribution(
            total_batch, replicate_weights
        )

        print(f"\n场景数分配:")
        for i, (count, weight, batch_count) in enumerate(
            zip(replicate_counts, replicate_weights, batch_distribution)
        ):
            print(f"  Replicate {count}: 权重 {weight:.3f} -> {batch_count} 场景")
        print(f"  总计: {sum(batch_distribution)} 场景")

        # 执行批量仿真
        total_simulations = len(xml_files) * len(replicate_counts)
        completed = 0
        failed = 0

        for xml_file in xml_files:
            for replicate_count, batch_count in zip(
                replicate_counts, batch_distribution
            ):
                if batch_count > 0:  # 只运行分配了场景数的配置
                    success = self.run_single_simulation(
                        xml_file, replicate_count, batch_count, extra_args
                    )
                    if success:
                        completed += 1
                    else:
                        failed += 1

        # 打印总结
        print(f"\n{'='*80}")
        print(f"批量仿真完成")
        print(f"  总仿真数: {total_simulations}")
        print(f"  成功: {completed}")
        print(f"  失败: {failed}")
        print(f"{'='*80}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量MuJoCo仿真脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python3 0821/run.py --scenes-dir assets/mjcf/scenes --batch 1000 \\
    --replicate-counts 10 25 50 100 250 \\
    --replicate-weights 0.1 0.2 0.4 0.2 0.1

  # 传递额外参数给main.py
  python3 0821/run.py --scenes-dir assets/mjcf/scenes --batch 500 \\
    --replicate-counts 50 100 \\
    --replicate-weights 0.6 0.4 \\
    --steps 2000 --workers 8
        """,
    )

    # 批量仿真专用参数
    parser.add_argument(
        "--scenes-dir",
        type=str,
        required=True,
        help="场景文件目录路径，将遍历其中所有*_replicate.xml文件",
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        help="总场景数量，将根据权重分配给不同replicate数量",
    )
    parser.add_argument(
        "--replicate-counts",
        type=int,
        nargs="+",
        required=True,
        help="replicate数量列表，例如: 10 25 50 100 250",
    )
    parser.add_argument(
        "--replicate-weights",
        type=float,
        nargs="+",
        required=True,
        help="replicate权重列表，总和必须为1，例如: 0.1 0.2 0.4 0.2 0.1",
    )

    # 透传给main.py的参数
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--part_name", type=str, help="要随机化的零件名前缀")
    parser.add_argument("--steps", type=int, help="每个场景要仿真的步数")
    parser.add_argument("--workers", type=int, help="并行进程数量")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument(
        "--xy-extent", type=float, nargs=2, help="XY放置范围：[min, max]"
    )
    parser.add_argument("--z-low", type=float, help="第一层的z起始高度")
    parser.add_argument("--layer-gap", type=float, help="不同Z层之间的额外安全间隙")
    parser.add_argument("--min-clearance", type=float, help="不同零件之间的最小XY间隙")
    parser.add_argument("--jitter-frac", type=float, help="单元内XY抖动占间隙比例")
    parser.add_argument("--allow-3d-rot", action="store_true", help="允许三轴随机旋转")
    parser.add_argument(
        "--no-allow-3d-rot",
        dest="allow_3d_rot",
        action="store_false",
        help="禁用三轴随机旋转",
    )
    parser.add_argument("--yaw-only", action="store_true", help="仅绕Z轴旋转")
    parser.add_argument("--fit-mode", choices=["sphere", "aabb"], help="几何拟合模式")
    parser.add_argument("--viewer", action="store_true", help="是否显示可视化界面")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--npy-out-dir", type=str, help=".npy输出目录")
    parser.add_argument("--viz-out-dir", type=str, help="可视化输出目录")
    parser.add_argument("--save-modified-xml",
        action="store_true",
        help="保存修改后的XML文件到磁盘（默认仅在内存中修改）",
    )
    
    # 稳态检测相关参数
    parser.add_argument("--enable-stability-check", action="store_true", help="启用稳态检测")
    parser.add_argument("--disable-stability-check", action="store_true", help="禁用稳态检测")
    parser.add_argument("--position-threshold", type=float, help="位置变化阈值")
    parser.add_argument("--velocity-threshold", type=float, help="速度阈值")
    parser.add_argument("--stability-steps", type=int, help="连续稳定步数")

    return parser.parse_args()


def build_extra_args(args) -> List[str]:
    """构建要透传给main.py的额外参数"""
    extra_args = []

    # 简单参数
    simple_params = [
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
        ("position_threshold", "--position-threshold"),
        ("velocity_threshold", "--velocity-threshold"),
        ("stability_steps", "--stability-steps"),
    ]

    for attr, flag in simple_params:
        value = getattr(args, attr, None)
        if value is not None:
            extra_args.extend([flag, str(value)])

    # 列表参数
    if args.xy_extent is not None:
        extra_args.extend(["--xy-extent"] + [str(x) for x in args.xy_extent])

    # 布尔参数
    if args.allow_3d_rot is True:
        extra_args.append("--allow-3d-rot")
    elif args.allow_3d_rot is False:
        extra_args.append("--no-allow-3d-rot")

    if args.yaw_only:
        extra_args.append("--yaw-only")

    if args.viewer:
        extra_args.append("--viewer")

    if args.save_modified_xml:
        extra_args.append("--save-modified-xml")
        
    if args.enable_stability_check:
        extra_args.append("--enable-stability-check")
        
    if args.disable_stability_check:
        extra_args.append("--disable-stability-check")

    return extra_args


def main():
    """主函数"""
    args = parse_args()

    # 验证参数
    if len(args.replicate_counts) != len(args.replicate_weights):
        print("错误: replicate-counts和replicate-weights的长度必须相同")
        sys.exit(1)

    # 创建运行器并执行
    runner = BatchSimulationRunner()

    try:
        extra_args = build_extra_args(args)
        runner.run_batch_simulations(
            args.scenes_dir,
            args.batch,
            args.replicate_counts,
            args.replicate_weights,
            extra_args,
        )
    except Exception as e:
        print(f"批量仿真失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
