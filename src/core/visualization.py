import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import os
import tqdm
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from geometry import GeometryCalculator
from simulation import PoseExporter


class Visualizer:
    """可视化器"""

    def __init__(self, model):
        self.model = model
        self.geometry_calc = GeometryCalculator(model)

    def get_body_aabb(self, qpos: np.ndarray, body_info: List[Dict]) -> List[Tuple]:
        """获取所有目标body的AABB包围盒"""
        aabbs = []
        for info in body_info:
            addr = info["qpos_addr"]
            pos = qpos[addr : addr + 3]
            size_matrix, _, _ = self.geometry_calc.compute_body_bounds(info["id"])
            if size_matrix.ndim != 1 or size_matrix.shape[0] != 3:
                size_matrix = np.array([0.1, 0.1, 0.1])
            aabbs.append((pos, size_matrix))
        return aabbs

    @staticmethod
    def plot_aabb(
        ax,
        center: np.ndarray,
        size: np.ndarray,
        color: str = "g",
        alpha: float = 1.0,
        linewidth: float = 1.5,
    ):
        """在ax上绘制一个AABB线框"""
        dx, dy, dz = size / 2.0
        corners = np.array(
            [
                [center[0] - dx, center[1] - dy, center[2] - dz],
                [center[0] - dx, center[1] - dy, center[2] + dz],
                [center[0] - dx, center[1] + dy, center[2] - dz],
                [center[0] - dx, center[1] + dy, center[2] + dz],
                [center[0] + dx, center[1] - dy, center[2] - dz],
                [center[0] + dx, center[1] - dy, center[2] + dz],
                [center[0] + dx, center[1] + dy, center[2] - dz],
                [center[0] + dx, center[1] + dy, center[2] + dz],
            ]
        )

        edges = [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ]

        for e in edges:
            ax.plot(
                *zip(corners[e[0]], corners[e[1]]),
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

    @staticmethod
    def get_aabb_overlap(aabb1: Tuple, aabb2: Tuple) -> Optional[Tuple]:
        """计算两个AABB是否重叠"""
        c1, s1 = aabb1
        c2, s2 = aabb2
        min1 = c1 - s1 / 2
        max1 = c1 + s1 / 2
        min2 = c2 - s2 / 2
        max2 = c2 + s2 / 2
        overlap_min = np.maximum(min1, min2)
        overlap_max = np.minimum(max1, max2)
        if np.all(overlap_min < overlap_max):
            return overlap_min, overlap_max
        return None

    def visualize_scene(
        self, qpos: np.ndarray, body_info: List[Dict], save_path: str
    ) -> bool:
        """可视化场景"""
        try:
            xyz = []
            for info in body_info:
                addr = info["qpos_addr"]
                pos = qpos[addr : addr + 3]
                xyz.append(pos)
            xyz = np.array(xyz)

            aabbs = self.get_body_aabb(qpos, body_info)

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection="3d")

            # 绘制中心点
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="b", s=40, label="center")

            # 绘制AABB线框
            for center, size in aabbs:
                self.plot_aabb(ax, center, size, color="g", alpha=0.7, linewidth=1.2)

            # 检查重叠区域
            n = len(aabbs)
            overlap_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    overlap = self.get_aabb_overlap(aabbs[i], aabbs[j])
                    if overlap is not None:
                        overlap_count += 1
                        min_pt, max_pt = overlap
                        dx, dy, dz = max_pt - min_pt
                        ax.bar3d(
                            min_pt[0],
                            min_pt[1],
                            min_pt[2],
                            dx,
                            dy,
                            dz,
                            color="r",
                            alpha=0.4,
                            shade=True,
                        )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            if overlap_count > 0:
                ax.set_title(f"positions & AABB (发现 {overlap_count} 个重叠区域)")
            else:
                ax.set_title("positions & AABB (无重叠区域)")

            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return True
        except Exception as e:
            print(f"✗ 点图保存失败: {save_path}, 错误: {e}")
            if "fig" in locals():
                plt.close(fig)
            return False

    def save_render_image(
        self, qpos: np.ndarray, save_path: str, width: int = 1920, height: int = 1080
    ) -> bool:
        """渲染并保存图片"""
        try:
            data = mujoco.MjData(self.model)
            data.qpos[:] = qpos
            mujoco.mj_forward(self.model, data)
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            renderer.update_scene(data)
            frame = renderer.render()

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            media.write_image(save_path, frame)
            return True
        except Exception as e:
            print(f"✗ 渲染图保存失败: {save_path}, 错误: {e}")
            return False

    def save_random_scene_renders(
        self, results: List[Dict], out_root_dir: str, part_name: str, body_info: List[Dict]
    ):
        """
        每个进程随机选中的场景：保存初始化图、仿真终态图、以及从npy重建的终态图。
          out_root/<timestamp>/{initial_position, simulation_position, reconstructed_position}/<PID>.png
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.abspath(os.path.join(out_root_dir, timestamp))
        init_dir = os.path.join(root, "initial_position")
        final_dir = os.path.join(root, "simulation_position")
        reco_dir = os.path.join(root, "reconstructed_position")
        os.makedirs(init_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        os.makedirs(reco_dir, exist_ok=True)

        saved = 0
        for i, res in enumerate(tqdm.tqdm(results, desc="Saving scene renders")):
            pid = res.get("worker_pid", f"worker_{i}")
            init_qpos = res.get("viz_initial_qpos")
            final_qpos = res.get("viz_final_qpos")
            npy_path = res.get("viz_npy_path")

            if init_qpos is None or final_qpos is None:
                continue

            # 1) 初始化渲染
            self.save_render_image(init_qpos, os.path.join(init_dir, f"{pid}.png"))
            # 2) 仿真终态渲染（直接data终态）
            self.save_render_image(final_qpos, os.path.join(final_dir, f"{pid}.png"))

            # 3) 从np y重建后再渲染（验证导出/恢复一致性）
            if npy_path and os.path.isfile(npy_path):
                qpos_reco = PoseExporter.reconstruct_qpos_from_npy(self.model, body_info, npy_path)
                self.save_render_image(qpos_reco, os.path.join(reco_dir, f"{pid}.png"))

            saved += 1

        print(f"✓ 可视化保存完成：{saved} 个进程的随机场景快照已保存到 {root}")
