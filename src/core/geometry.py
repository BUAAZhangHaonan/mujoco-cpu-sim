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
            # 常见布局：每个geom一行，存 min/max 或 center/size；我们只用 XY 平均尺寸的并集
            # 兼容两种可能： [xmin,ymin,zmin,xmax,ymax,zmax] 或 [cx,cy,cz,sx,sy,sz]
            xs, ys, zs = [], [], []
            for g in geoms:
                a = np.array(self.model.geom_aabb[g], dtype=np.float64).reshape(-1)
                if a.shape[0] != 6:
                    return rbound, 2.0 * rbound
                # 粗判：若 a[:3] < a[3:] 很多，则当作 min/max；否则 center/size
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
                r_xy = 0.5 * math.hypot(w, d)  # 圆包矩形
                # 仍要与 rbound 取 max 保守
                return max(r_xy, rbound), max(h, 2.0 * rbound)
            else:
                return rbound, 2.0 * rbound
        except Exception:
            return rbound, 2.0 * rbound


class HexPackedInitializer:
    """
    用六角密堆在XY平面放置等效"包围圆"，并按需要进行多层Z放置。
    - 仅yaw随机，pitch/roll=0，避免改变垂直占用高度。
    - 不调用 mj_forward，不做碰撞松弛。初始化状态天然"零重叠"。
    """

    def __init__(
        self,
        model,
        body_info: List[Dict],
        xy_min: float,
        xy_max: float,
        clearance: float = 1e-3,
        layer_gap: float = 3e-3,
        yaw_only: bool = False,
        allow_3d_rot: bool = True,
        jitter_frac: float = 0.35,
        fit_mode: str = "sphere",  # "sphere"（安全保守）或 "aabb"（尽量紧，mesh回退sphere）
    ):
        assert xy_max > xy_min, "xy-extent无效"
        self.model = model
        self.body_info = body_info
        self.xy_min = xy_min
        self.xy_max = xy_max
        self.clearance = max(0.0, float(clearance))
        self.layer_gap = max(0.0, float(layer_gap))
        self.yaw_only = yaw_only
        self.allow_3d_rot = allow_3d_rot
        self.jitter_frac = float(jitter_frac)
        self.fit_mode = fit_mode
        if self.allow_3d_rot:
            self.fit_mode = "sphere"  # 方向无关，稳

        # 估计每个 body 的占地与高度，最后取"全局上界"作为统一放置尺度，确保零重叠
        geo_calc = GeometryCalculator(model)
        r_xy_list, h_z_list = [], []
        for info in body_info:
            r, h = geo_calc.estimate_radius_and_height(info["id"], self.fit_mode)
            r_xy_list.append(r)
            h_z_list.append(h)
        self.R_xy = float(np.max(r_xy_list))  # 占地等效半径（安全）
        self.H_z = float(np.max(h_z_list))  # 等效高度（安全）

        # 单元间距（圆的间距 + 间隙）
        self.step = 2.0 * self.R_xy + self.clearance
        # 六角格参数
        self.dx = self.step
        self.dy = math.sqrt(3.0) * 0.5 * self.step
        # 内边距：把中心点至少放在R_xy+clearance/2 以内，防止越界
        self.margin = self.R_xy + 0.5 * self.clearance

        # 预计算当前XY边界能容纳的最大列/行
        self._precompute_grid()

    def _precompute_grid(self):
        W = self.xy_max - self.xy_min
        # 能放置的 y 行数
        usable_y = max(0.0, W - 2.0 * self.margin)
        self.n_rows = int(usable_y // self.dy) + 1 if usable_y >= 0 else 0
        # 每一行的 x 能放置的列数（考虑奇偶行偏移）
        usable_x = max(0.0, W - 2.0 * self.margin)
        self.n_cols_main = int(usable_x // self.dx) + 1 if usable_x >= 0 else 0
        # 奇数行偏移后能放置的列数
        usable_x_shift = max(0.0, W - 2.0 * self.margin - 0.5 * self.dx)
        self.n_cols_shift = (
            int(usable_x_shift // self.dx) + 1 if usable_x_shift >= 0 else 0
        )
        self.capacity_per_layer = 0
        for r in range(self.n_rows):
            if r % 2 == 0:
                self.capacity_per_layer += self.n_cols_main
            else:
                self.capacity_per_layer += self.n_cols_shift

    def _layer_z(self, layer_idx: int, z_low: float) -> float:
        return z_low + layer_idx * (self.H_z + self.layer_gap)

    def _iter_xy_centers_layer(self):
        """
        迭代一层的中心点（按六角堆），不含抖动。
        """
        if self.capacity_per_layer <= 0:
            return
        y0 = self.xy_min + self.margin
        for r in range(self.n_rows):
            y = y0 + r * self.dy
            if y > self.xy_max - self.margin:
                break
            # 行偏移
            if r % 2 == 0:
                x_start = self.xy_min + self.margin
                ncols = self.n_cols_main
            else:
                x_start = self.xy_min + self.margin + 0.5 * self.dx
                ncols = self.n_cols_shift
            for c in range(ncols):
                x = x_start + c * self.dx
                if x > self.xy_max - self.margin:
                    break
                yield x, y

    def _apply_jitter(self, x: float, y: float) -> Tuple[float, float]:
        """
        在当前中心点周围加入抖动：抖动半径 <= jitter_frac * clearance
        保证两圆心距离仍 >= 2R + clearance*(1 - 2*jitter_frac) > 2R
        """
        if self.jitter_frac <= 0.0 or self.clearance <= 0.0:
            return x, y
        rj = self.jitter_frac * self.clearance
        theta = np.random.uniform(0.0, 2.0 * math.pi)
        return x + rj * math.cos(theta), y + rj * math.sin(theta)

    def generate(self, batch_size: int, z_low: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成 batch_size 个场景的 (qpos_batch, qvel_batch)
        - 绝不调用 mj_forward
        """
        from utils import MuJoCoUtils
        
        n = len(self.body_info)
        if n == 0:
            return np.zeros((batch_size, self.model.nq)), np.zeros(
                (batch_size, self.model.nv)
            )

        # 计算需要多少层
        if self.capacity_per_layer <= 0:
            raise RuntimeError(
                f"XY范围过小（[{self.xy_min},{self.xy_max}]），无法放置任何物体。"
                f" 请增大范围或减少安全半径（当前R={self.R_xy:.6f}，clearance={self.clearance:.6f})"
            )
        n_layers = int(math.ceil(n / float(self.capacity_per_layer)))

        # 预先生成一层的所有中心点列表（不抖动）
        centers_one_layer = list(self._iter_xy_centers_layer())
        assert len(centers_one_layer) == self.capacity_per_layer

        qpos_batch = np.zeros((batch_size, self.model.nq), dtype=np.float64)
        qvel_batch = np.zeros((batch_size, self.model.nv), dtype=np.float64)

        # 每个场景：打散 body 顺序、打散中心点次序，按层填充，加入抖动、随机yaw
        for b in range(batch_size):
            qpos = self.model.qpos0.copy()

            # 随机打乱 body 顺序
            order = np.arange(n, dtype=np.int32)
            np.random.shuffle(order)

            # 随机打乱一层中的中心点顺序
            centers_perm = centers_one_layer.copy()
            np.random.shuffle(centers_perm)

            idx = 0
            for layer in range(n_layers):
                z = self._layer_z(layer, z_low)
                # 本层可放置数
                remain = n - idx
                take = min(remain, self.capacity_per_layer)
                # 若最后一层不足，仍从 centers_perm 头部取
                for k in range(take):
                    i_body = order[idx]
                    info = self.body_info[i_body]
                    addr = info["qpos_addr"]

                    x0, y0 = centers_perm[k]
                    x, y = self._apply_jitter(x0, y0)

                    # 写入 qpos: [x,y,z, quat(w,x,y,z)]
                    qpos[addr : addr + 3] = [x, y, z]
                    if self.allow_3d_rot:
                        qpos[addr + 3 : addr + 7] = MuJoCoUtils.uniform_quat_wxyz()
                    elif self.yaw_only:
                        qpos[addr + 3 : addr + 7] = MuJoCoUtils.yaw_to_quat(
                            np.random.uniform(0.0, 2.0 * math.pi)
                        )
                    else:
                        # 保底（不建议用完全随机但未声明 allow_3d_rot 的情况）
                        qpos[addr + 3 : addr + 7] = MuJoCoUtils.yaw_to_quat(
                            np.random.uniform(0.0, 2.0 * math.pi)
                        )

                    idx += 1
                    if idx >= n:
                        break

            qpos_batch[b] = qpos
            # 初速置零
            qvel_batch[b] = 0.0

        return qpos_batch, qvel_batch


class SpatialPacker:
    """空间打包算法（保留原功能）"""

    @staticmethod
    def dense_pack_xy(
        aabb_xy_sizes: np.ndarray,
        radii_xy: np.ndarray,
        n_obj: int,
        xy_min: float,
        xy_max: float,
        min_clearance: float,
        pack_ratio: float,
        max_trials_per_obj: int = 2000,
    ) -> np.ndarray:
        """使用网格哈希 + 贪心在 XY 平面放置对象"""
        order = np.argsort(-radii_xy)
        placed_xy = np.zeros((n_obj, 2), dtype=np.float64)
        cell = np.max(radii_xy) * 2.0 * pack_ratio
        if cell <= 0:
            cell = (xy_max - xy_min) / max(1, int(np.sqrt(n_obj)))
        nx = max(1, int((xy_max - xy_min) / cell))
        ny = max(1, int((xy_max - xy_min) / cell))
        grid = [[[] for _ in range(ny)] for _ in range(nx)]

        def cell_of(x, y):
            cx = int((x - xy_min) // cell)
            cy = int((y - xy_min) // cell)
            return np.clip(cx, 0, nx - 1), np.clip(cy, 0, ny - 1)

        def ok_to_place(i, x, y):
            ri = radii_xy[i] + min_clearance
            cx, cy = cell_of(x, y)
            for ix in range(max(0, cx - 1), min(nx, cx + 2)):
                for iy in range(max(0, cy - 1), min(ny, cy + 2)):
                    for j in grid[ix][iy]:
                        rj = radii_xy[j] + min_clearance
                        px, py = placed_xy[j]
                        if (x - px) ** 2 + (y - py) ** 2 < (ri + rj) ** 2:
                            half_size_i = aabb_xy_sizes[i] / 2.0 + min_clearance
                            half_size_j = aabb_xy_sizes[j] / 2.0 + min_clearance

                            if abs(x - px) < (half_size_i[0] + half_size_j[0]) and abs(
                                y - py
                            ) < (half_size_i[1] + half_size_j[1]):
                                return False
            return True

        for _, i in enumerate(order):
            ok = False
            for _ in range(max_trials_per_obj):
                x = np.random.uniform(xy_min, xy_max)
                y = np.random.uniform(xy_min, xy_max)
                if ok_to_place(i, x, y):
                    placed_xy[i] = [x, y]
                    cx, cy = cell_of(x, y)
                    grid[cx][cy].append(i)
                    ok = True
                    break
            if not ok:
                placed_xy[i] = [
                    np.random.uniform(xy_min, xy_max),
                    np.random.uniform(xy_min, xy_max),
                ]
        return placed_xy

    @staticmethod
    def poisson_disk_2d(num_points, r, xy_min, xy_max, k=30):
        """Bridson Poisson-disk 采样"""
        width = xy_max - xy_min
        cell_size = r / np.sqrt(2)
        grid_w = int(np.ceil(width / cell_size))
        grid_h = int(np.ceil(width / cell_size))
        grid = -np.ones((grid_w, grid_h), dtype=int)

        def grid_coord(p):
            return (int((p[0] - xy_min) / cell_size), int((p[1] - xy_min) / cell_size))

        def in_neighborhood(p):
            gx, gy = grid_coord(p)
            for i in range(max(gx - 2, 0), min(gx + 3, grid_w)):
                for j in range(max(gy - 2, 0), min(gy + 3, grid_h)):
                    idx = grid[i, j]
                    if idx != -1:
                        q = samples[idx]
                        if np.linalg.norm(p - q) < r:
                            return True
            return False

        samples = []
        active = []

        p = np.random.uniform(xy_min, xy_max, size=2)
        samples.append(p)
        active.append(0)
        gx, gy = grid_coord(p)
        grid[gx, gy] = 0

        while active and len(samples) < num_points:
            idx = np.random.choice(active)
            center = samples[idx]
            found = False
            for _ in range(k):
                theta = np.random.uniform(0, 2 * np.pi)
                rad = np.random.uniform(r, 2 * r)
                p = center + rad * np.array([np.cos(theta), np.sin(theta)])
                if p[0] < xy_min or p[0] > xy_max or p[1] < xy_min or p[1] > xy_max:
                    continue
                if not in_neighborhood(p):
                    samples.append(p)
                    active.append(len(samples) - 1)
                    gx, gy = grid_coord(p)
                    grid[gx, gy] = len(samples) - 1
                    found = True
                    break
            if not found:
                active.remove(idx)

        return np.array(samples[:num_points])
