import numpy as np
import mujoco
from tqdm import tqdm
from typing import List, Dict, Tuple
from .geometry import HexPackedInitializer, SpatialPacker
from .utils import MuJoCoUtils, ModelAnalyzer


class StateInitializer:
    """状态初始化器"""

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def generate_hex_packed_states(
        self,
        body_info: List[Dict],
        batch_size: int,
        xy_extent: Tuple[float, float] = (-0.05, 0.05),
        z_low: float = 0.05,
        min_clearance: float = 1e-3,
        layer_gap: float = 3e-3,
        yaw_only: bool = False,
        allow_3d_rot: bool = True,
        jitter_frac: float = 0.35,
        fit_mode: str = "sphere",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用六角密堆生成高密度非重叠初始状态"""
        print(f"正在为 {batch_size} 个场景生成六角密堆非重叠初始状态...")

        xy_min, xy_max = xy_extent
        
        # 创建六角密堆初始化器
        initializer = HexPackedInitializer(
            model=self.model,
            body_info=body_info,
            xy_min=xy_min,
            xy_max=xy_max,
            clearance=min_clearance,
            layer_gap=layer_gap,
            yaw_only=yaw_only,
            allow_3d_rot=allow_3d_rot,
            jitter_frac=jitter_frac,
            fit_mode=fit_mode,
        )

        # 生成初始状态
        qpos_batch, qvel_batch = initializer.generate(batch_size, z_low)
        print("六角密堆初始状态生成完毕。")
        
        return qpos_batch, qvel_batch

    def generate_dense_states_with_bounds(
        self,
        body_info: List[Dict],
        batch_size: int,
        xy_extent: Tuple[float, float] = (-0.2, 0.2),
        z_low: float = 0.05,
        z_high: float = 0.2,
        min_clearance: float = 1e-3,
        pack_ratio: float = 1.0,
        local_relax_iters: int = 15,
        local_relax_step: float = 2e-3,
        ignore_ground_contacts: bool = True,
        use_poisson: bool = False,
        poisson_k: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成高密度非重叠初始状态（保留原功能）"""
        print(f"正在为 {batch_size} 个场景生成高密度非重叠随机状态...")

        n = len(body_info)
        aabb_sizes = np.zeros((n, 3))
        sphere_rs = np.zeros(n)
        ec8_geom_ids_set = set()

        # 预先收集 ec8 碰撞 geom
        analyzer = ModelAnalyzer(self.model)

        for idx, info in enumerate(body_info):
            bid = info["id"]
            from geometry import GeometryCalculator
            
            geo_calc = GeometryCalculator(self.model)
            aabb, _, r = geo_calc.compute_body_bounds(bid)
            if aabb.ndim != 1 or aabb.shape[0] != 3:
                aabb = np.array([2 * r, 2 * r, 2 * r])

            aabb_sizes[idx] = aabb
            sphere_rs[idx] = r
            for g in analyzer.get_ec8_collision_geoms(bid):
                ec8_geom_ids_set.add(g)

        ec8_geom_ids = np.array(sorted(list(ec8_geom_ids_set)), dtype=np.int32)
        xy_min, xy_max = xy_extent
        qpos_batch = np.zeros((batch_size, self.model.nq))
        qvel_batch = np.zeros((batch_size, self.model.nv))

        packer = SpatialPacker()

        for b in tqdm(range(batch_size), desc="Generating dense non-colliding states"):
            # 1) XY 放置
            if use_poisson:
                r = np.max(sphere_rs) + min_clearance
                xy = packer.poisson_disk_2d(n, r, xy_min, xy_max, k=poisson_k)
                if xy.shape[0] < n:
                    xy = packer.dense_pack_xy(
                        aabb_sizes[:, :2],
                        sphere_rs,
                        n,
                        xy_min,
                        xy_max,
                        min_clearance,
                        pack_ratio,
                    )
            else:
                xy = packer.dense_pack_xy(
                    aabb_sizes[:, :2],
                    sphere_rs,
                    n,
                    xy_min,
                    xy_max,
                    min_clearance,
                    pack_ratio,
                )

            # 2) 随机 z & 姿态
            qpos = self.model.qpos0.copy()
            for j, info in enumerate(body_info):
                addr = info["qpos_addr"]
                z = np.random.uniform(z_low, z_high)
                qpos[addr : addr + 3] = np.array(
                    [xy[j, 0], xy[j, 1], z], dtype=np.float64
                )
                qpos[addr + 3 : addr + 7] = MuJoCoUtils.uniform_quat_wxyz()

            # 3) 前向一次，用于接触检测
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)

            # 4) 统计 ec8–ec8 接触
            def has_ec8_to_ec8_contact():
                for k in range(self.data.ncon):
                    c = self.data.contact[k]
                    g1, g2 = c.geom1, c.geom2
                    if ignore_ground_contacts:
                        if (g1 not in ec8_geom_ids) or (g2 not in ec8_geom_ids):
                            continue
                    if (g1 in ec8_geom_ids) and (g2 in ec8_geom_ids):
                        return True
                return False

            # 5) 局部微调
            it = 0
            while has_ec8_to_ec8_contact() and it < local_relax_iters:
                move_ids = np.random.choice(n, size=min(5, n), replace=False)
                for j in move_ids:
                    addr = body_info[j]["qpos_addr"]
                    delta = np.random.uniform(-1.0, 1.0, size=2)
                    if np.linalg.norm(delta) < 1e-6:
                        delta = np.array([1.0, 0.0])
                    delta = delta / np.linalg.norm(delta) * local_relax_step
                    qpos[addr] += delta[0]
                    qpos[addr + 1] += delta[1]
                    qpos[addr] = np.clip(qpos[addr], xy_min, xy_max)
                    qpos[addr + 1] = np.clip(qpos[addr + 1], xy_min, xy_max)

                self.data.qpos[:] = qpos
                mujoco.mj_forward(self.model, self.data)
                it += 1

            qpos_batch[b] = qpos
            qvel_batch[b] = 0.0

        return qpos_batch, qvel_batch
