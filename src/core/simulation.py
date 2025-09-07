import mujoco
import numpy as np
import multiprocessing as mp
import time
from queue import Empty, Queue
from tqdm import tqdm
from typing import List, Dict
import os


_unstable_flag = False


class Simulator:
    """仿真器"""

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def simulate_single_scene(
        self,
        initial_qpos: np.ndarray,
        initial_qvel: np.ndarray,
        steps: int,
        use_viewer: bool = False,
    ) -> bool:
        """仿真单个场景"""
        global _unstable_flag
        _unstable_flag = False

        self.data.qpos[:] = initial_qpos
        self.data.qvel[:] = initial_qvel
        mujoco.mj_forward(self.model, self.data)

        if use_viewer:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                step_count = 0
                while viewer.is_running() and step_count < steps:
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    step_count += 1
        else:
            for step in range(steps):
                if _unstable_flag:
                    print(f"警告: 仿真在步骤 {step+1} 变得不稳定，提前终止此场景。")
                    return False
                mujoco.mj_step(self.model, self.data)

        return True


class PoseExporter:
    """位姿导出器"""

    @staticmethod
    def export_final_poses_npy(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        part_name: str,
        out_root: str,
        replicate_count: int,
        scene_idx: int,
    ) -> str:
        """
        导出该场景所有目标 body 的最终位姿（xyz + quat[wxyz]），顺序按 qpos_addr 升序稳定。
        文件路径：out_root/<part_name>/<replicate_count>/scene_{scene_idx:06d}.npy
        """
        from utils import ModelAnalyzer

        analyzer = ModelAnalyzer(model)
        body_info = analyzer.find_target_bodies(part_name)
        body_info = sorted(body_info, key=lambda x: x["qpos_addr"])
        poses = np.zeros((len(body_info), 7), dtype=np.float64)
        for i, info in enumerate(body_info):
            adr = info["qpos_addr"]
            poses[i, :3] = data.qpos[adr : adr + 3]
            poses[i, 3:7] = data.qpos[adr + 3 : adr + 7]  # [w,x,y,z]
        out_dir = os.path.join(out_root, part_name, str(int(replicate_count)))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"scene_{scene_idx:06d}.npy")
        np.save(out_path, poses)
        return out_path

    @staticmethod
    def reconstruct_qpos_from_npy(
        model: mujoco.MjModel, body_info: List[Dict], npy_path: str
    ) -> np.ndarray:
        poses = np.load(npy_path)  # shape (N,7)
        bis = sorted(body_info, key=lambda x: x["qpos_addr"])
        assert poses.shape[0] == len(
            bis
        ), f"恢复失败：npy数量({poses.shape[0]})与body数({len(bis)})不一致"
        qpos = model.qpos0.copy()
        for i, info in enumerate(bis):
            adr = info["qpos_addr"]
            qpos[adr : adr + 3] = poses[i, :3]
            qpos[adr + 3 : adr + 7] = poses[i, 3:7]  # [w,x,y,z]
        return qpos


class MultiprocessSimulator:
    """多进程仿真器"""

    @staticmethod
    def worker_simulate_scenes(
        xml_path: str,
        scene_indices: List[int],
        qpos_batch: np.ndarray,
        qvel_batch: np.ndarray,
        steps: int,
        progress_queue: Queue,
        part_name: str,
        npy_out_dir: str,
        replicate_count: int,
    ) -> Dict:
        """工作进程函数"""
        from model_loader import ModelLoader

        loader = ModelLoader()
        model, data = loader.load_model(xml_path)
        simulator = Simulator(model, data)
        exporter = PoseExporter()

        worker_pid = os.getpid()

        # 随机选择一个场景用于可视化
        rng = np.random.default_rng(abs(hash((worker_pid, time.time()))) % (2**32))
        viz_scene_idx = int(rng.choice(scene_indices)) if scene_indices else None

        viz_initial_qpos = None
        viz_final_qpos = None
        viz_npy_path = None

        start_time = time.perf_counter()
        scenes_processed = 0
        scenes_failed = 0

        for scene_idx in scene_indices:
            if scene_idx == viz_scene_idx:
                viz_initial_qpos = qpos_batch[scene_idx].copy()

            ok = simulator.simulate_single_scene(
                qpos_batch[scene_idx],
                qvel_batch[scene_idx],
                steps,
                use_viewer=False,
            )

            # —— 无论是否为可视化场景，都导出 .npy —— #
            npy_path = exporter.export_final_poses_npy(
                model,
                data,
                part_name,
                npy_out_dir,
                replicate_count,
                scene_idx,
            )
            if scene_idx == viz_scene_idx:
                viz_final_qpos = data.qpos.copy()
                viz_npy_path = npy_path

            scenes_processed += 1 if ok else 0
            scenes_failed += 0 if ok else 1
            progress_queue.put(1)

        elapsed = time.perf_counter() - start_time
        return {
            "scenes_processed": scenes_processed,
            "scenes_failed": scenes_failed,
            "elapsed_time": elapsed,
            "steps_per_scene": steps,
            "worker_pid": worker_pid,
            "viz_scene_index": viz_scene_idx,
            "viz_initial_qpos": viz_initial_qpos,
            "viz_final_qpos": viz_final_qpos,
            "viz_npy_path": viz_npy_path,
        }

    @staticmethod
    def distribute_scenes(total_scenes: int, num_workers: int) -> List[List[int]]:
        """将场景分配给不同的工作进程"""
        scenes_per_worker = total_scenes // num_workers
        remainder = total_scenes % num_workers

        scene_assignments = []
        start_idx = 0

        for i in range(num_workers):
            current_batch_size = scenes_per_worker + (1 if i < remainder else 0)
            end_idx = start_idx + current_batch_size
            scene_assignments.append(list(range(start_idx, end_idx)))
            start_idx = end_idx

        return scene_assignments

    @staticmethod
    def run_multiprocess_simulation(
        xml_path: str,
        qpos_batch: np.ndarray,
        qvel_batch: np.ndarray,
        steps: int,
        num_workers: int,
        part_name: str = None,
        npy_out_dir: str = None,
        replicate_count: int = None,
    ) -> List[Dict]:
        """运行多进程仿真"""
        total_scenes = len(qpos_batch)
        scene_assignments = MultiprocessSimulator.distribute_scenes(
            total_scenes, num_workers
        )

        print(f"使用 {num_workers} 个进程进行仿真...")
        for i, assignment in enumerate(scene_assignments):
            print(f"进程 {i+1}: {len(assignment)} 个场景")

        manager = mp.Manager()
        progress_queue = manager.Queue()

        with mp.Pool(processes=num_workers) as pool:
            tasks = []
            for scene_indices in scene_assignments:
                if scene_indices:
                    task = pool.apply_async(
                        MultiprocessSimulator.worker_simulate_scenes,
                        (
                            xml_path,
                            scene_indices,
                            qpos_batch,
                            qvel_batch,
                            steps,
                            progress_queue,
                            part_name,
                            npy_out_dir,
                            replicate_count,
                        ),
                    )
                    tasks.append(task)

            with tqdm(total=total_scenes, desc="多进程仿真") as pbar:
                processed = 0
                while processed < total_scenes:
                    try:
                        progress_queue.get(timeout=0.1)
                        pbar.update(1)
                        processed += 1
                    except Empty:
                        if any(t.ready() and not t.successful() for t in tasks):
                            print("\n有子进程失败，终止池...")
                            for t in tasks:
                                if t.ready() and not t.successful():
                                    try:
                                        t.get()
                                    except Exception as e:
                                        print(f"子进程错误: {e}")
                            pool.terminate()
                            return []
            results = [t.get() for t in tasks]

        return results
