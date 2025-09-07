import numpy as np
import math
from typing import Tuple, List, Dict
from .utils import ModelAnalyzer


class GeometryCalculator:
    """几何计算相关功能"""

    def __init__(self, model):
        self.model = model

    def compute_body_bounds(self, body_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """计算 body 的 AABB 和包围球半径"""
        analyzer = ModelAnalyzer(self.model)
        geoms = analyzer.get_collision_geoms(body_id)

        if len(geoms) == 0:
            return np.array([0, 0, 0]), np.zeros(3), 0.0

        if (
            hasattr(self.model, "geom_aabb")
            and self.model.geom_aabb is not None
            and self.model.geom_aabb.size == self.model.ngeom * 6
        ):
            centers = []
            sizes = []

            for g in geoms:
                c = self.model.geom_aabb[g * 6 : g * 6 + 3]
                s = self.model.geom_aabb[g * 6 + 3 : g * 6 + 6]
                centers.append(c)
                sizes.append(s)

            if len(centers) == 0 or len(sizes) == 0:
                return np.array([0, 0, 0]), np.zeros(3), 0.0

            centers = np.stack(centers, axis=0)
            sizes = np.stack(sizes, axis=0)
            min_corner = (centers - sizes / 2.0).min(axis=0)
            max_corner = (centers + sizes / 2.0).max(axis=0)
            aabb_size = max_corner - min_corner
            sphere_r = 0.5 * np.linalg.norm(aabb_size)
            return aabb_size, np.zeros(3), sphere_r
        else:
            r = 0.0
            for g in geoms:
                r = max(r, self.model.geom_rbound[g])
            aabb_size = np.array([2 * r, 2 * r, 2 * r])
            return aabb_size, np.zeros(3), r

    def estimate_radius_and_height(self, body_id: int, fit_mode: str = "sphere") -> Tuple[float, float]:
        """
        粗略估计：XY占地半径、Z高度。
        - 默认：用 geom_rbound 作为包围球半径（对mesh稳定）
        - 可选AABB：若能可靠获得XY尺寸，用 0.5*sqrt(w^2+d^2) 作为占地半径；Z高度取AABB_z
        """
        # 收集属于该body的碰撞geoms
        analyzer = ModelAnalyzer(self.model)
        geoms = analyzer.get_collision_geoms(body_id)

        # —— rbound 路径（稳）——
        rbound = 0.0
        for g in geoms:
            rbound = max(rbound, float(self.model.geom_rbound[g]))
        if rbound <= 0.0:
            # 没有碰撞几何，给个极小值避免除零
            rbound = 1e-4

        if fit_mode != "aabb":
            return rbound, 2.0 * rbound  # 占地半径、等效高度（保守）

        # —— aabb 路径（更紧，但遇mesh/旋转就回退 rbound）——
        # 有些版本/编译选项下 model.geom_aabb 不可用或形状不一致，这里尽量容错
        try:
            if getattr(self.model, "geom_aabb", None) is None:
                return rbound, 2.0 * rbound
            xs, ys, zs = [], [], []
            for g in geoms:
                a = np.array(self.model.geom_aabb[g], dtype=np.float64).reshape(-1)
                if a.shape[0] != 6:
                    return rbound, 2.0 * rbound
                if np.all(a[:3] <= a[3:]):
                    w = a[3] - a[0]
                    d = a[4] - a[1]
                    h = a[5] - a[2]
                else:
                    w = a[3]
                    d = a[4]
                    h = a[5]
                xs.append(w)
                ys.append(d)
                zs.append(h)
            if xs and ys and zs:
                w = float(np.max(xs))
                d = float(np.max(ys))
                h = float(np.max(zs))
                r_xy = 0.5 * math.hypot(w, d)
                return max(r_xy, rbound), max(h, 2.0 * rbound)
            else:
                return rbound, 2.0 * rbound
        except Exception:
            return rbound, 2.0 * rbound
