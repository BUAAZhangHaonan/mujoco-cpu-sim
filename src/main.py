import mujoco
import numpy as np
import time
import multiprocessing as mp
import os
import glob
import re

from src.core.utils import MuJoCoUtils, ModelAnalyzer, XmlProcessor
from src.core.model_loader import ModelLoader
from src.core.initialization import StateInitializer
from src.core.simulation import Simulator, MultiprocessSimulator
from src.core.visualization import Visualizer
from src.config.config_manager import ConfigManager


class MuJoCoSimulationManager:
    """MuJoCo仿真管理器"""

    def __init__(self):
        # 设置显示环境
        MuJoCoUtils.setup_display()

        # 加载配置
        self.config_manager = ConfigManager()
        self.config = self.config_manager.parse_args_and_merge_config()

        # 初始化组件
        self.model_loader = ModelLoader()
        self.model = None
        self.data = None
        self.body_info = None

    @staticmethod
    def get_replicate_count_from_xml(xml_path: str) -> int:
        """从XML文件中读取replicate数量"""
        try:
            with open(xml_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 查找 <replicate count="数字"> 模式
            match = re.search(
                r'<replicate\b[^>]*\bcount\s*=\s*["\'](\d+)["\']', content
            )
            if match:
                return int(match.group(1))
            else:
                print(f"警告: 在 {xml_path} 中未找到 replicate count，默认使用1")
                return 1
        except Exception as e:
            print(f"读取XML文件 {xml_path} 时出错: {e}，默认使用1")
            return 1

    def setup(self):
        """设置仿真环境"""
        # 设置随机种子
        np.random.seed(self.config.simulation.seed)

        # 解析模型路径并加载模型
        xml_path = self.model_loader.resolve_model_path(self.config.model.path)
        print(f"正在从 '{xml_path}' 加载模型...")

        # 处理replicate count
        if self.config.model.replicate_count is not None:
            # 使用现有的 patch_replicate_count 方法创建修改后的XML文件
            modified_xml_path = XmlProcessor.patch_replicate_count(
                xml_path, self.config.model.replicate_count
            )

            # 检查是否保存修改后的XML文件（确保属性存在）
            save_modified_xml = getattr(self.config.model, "save_modified_xml", False)

            if save_modified_xml:
                # 如果要保存，就使用修改后的文件路径
                xml_path = modified_xml_path
                print(
                    f"已将 XML 中的 replicate 数量修改为: {self.config.model.replicate_count}，并保存到: {xml_path}"
                )
            else:
                # 如果不保存，使用修改后的文件但在仿真完成后删除
                xml_path = modified_xml_path
                self._temp_xml_path = modified_xml_path  # 保存临时文件路径用于后续清理
                print(
                    f"已将 XML 中的 replicate 数量修改为: {self.config.model.replicate_count}（临时文件，仿真后将删除）"
                )
        else:
            print("未指定 replicate-count，保持 XML 原有数量。")
            self._temp_xml_path = None

        self.model, self.data = self.model_loader.load_model(xml_path)
        print("模型加载成功。")

        # 分析模型
        analyzer = ModelAnalyzer(self.model)
        part_name = (
            self.config.model.part_name
            if self.config.model.part_name
            else (
                os.path.splitext(os.path.basename(xml_path))[0].split(
                    "_scene_replicate"
                )[0]
                if "_scene_replicate" in os.path.basename(xml_path)
                else "ec8"
            )
        )
        print(f"零件名称: '{part_name}'")

        self.body_info = analyzer.find_target_bodies(part_name)
        n_inst = len(self.body_info)
        print(f"找到 {n_inst} 个 '{part_name}' 实例进行随机化")

        return xml_path, part_name

    def process_xml_files(self):
        """处理XML文件"""
        print("\n开始处理XML文件...")

        # 处理 *_dependencies.xml 文件
        dep_files = glob.glob(
            os.path.join(
                self.config.xml_processing.models_dir, "**", "*_dependencies.xml"
            ),
            recursive=True,
        )
        changed = 0
        for f in dep_files:
            try:
                changed += (
                    1
                    if XmlProcessor.process_dependencies_xml(
                        f, self.config.xml_processing.mesh_scale
                    )
                    else 0
                )
            except Exception as e:
                print(f"[error] {f}: {e}")
        print(f"处理 *_dependencies.xml 完成，修改 {changed} 个文件")

        # 处理 *_body.xml 文件
        body_files = glob.glob(
            os.path.join(self.config.xml_processing.models_dir, "**", "*_body.xml"),
            recursive=True,
        )
        changed = 0
        for f in body_files:
            try:
                changed += 1 if XmlProcessor.process_body_xml(f) else 0
            except Exception as e:
                print(f"[error] {f}: {e}")
        print(f"处理 *_body.xml 完成，修改 {changed} 个文件")

        # 处理 *_replicate.xml 文件
        scene_files = glob.glob(
            os.path.join(
                self.config.xml_processing.scenes_dir, "**", "*_replicate.xml"
            ),
            recursive=True,
        )
        changed = 0
        for f in scene_files:
            try:
                changed += (
                    1
                    if XmlProcessor.process_scene_xml(
                        f,
                        self.config.xml_processing.texture_width,
                        self.config.xml_processing.texture_height,
                        self.config.xml_processing.ground_size,
                    )
                    else 0
                )
            except Exception as e:
                print(f"[error] {f}: {e}")
        print(f"处理 *_replicate.xml 完成，修改 {changed} 个文件")

    def generate_initial_states(self, part_name):
        """生成初始状态"""
        initializer = StateInitializer(self.model, self.data)

        # 使用六角密堆初始化
        qpos_batch, qvel_batch = initializer.generate_hex_packed_states(
            self.body_info,
            self.config.simulation.batch,
            xy_extent=tuple(self.config.initialization.xy_extent),
            z_low=self.config.initialization.z_low,
            min_clearance=self.config.initialization.min_clearance,
            layer_gap=self.config.initialization.layer_gap,
            yaw_only=self.config.initialization.yaw_only,
            allow_3d_rot=self.config.initialization.allow_3d_rot,
            jitter_frac=self.config.initialization.jitter_frac,
            fit_mode=self.config.initialization.fit_mode,
        )
        print("所有场景的高密度无碰撞初始位姿已生成。")
        return qpos_batch, qvel_batch

    def run_simulation(self, xml_path, qpos_batch, qvel_batch, part_name):
        """运行仿真"""
        num_workers = self.config.multiprocessing.workers
        if num_workers == -1:
            num_workers = mp.cpu_count()
        elif num_workers == 0:
            num_workers = 0
        else:
            num_workers = min(num_workers, mp.cpu_count())

        print(f"CPU核心数: {mp.cpu_count()}")
        if num_workers > 0:
            print(f"使用进程数: {num_workers}")
        else:
            print("使用单进程模式")

        start_time = time.perf_counter()

        if num_workers == 0 or (
            self.config.rendering.viewer and self.config.simulation.batch == 1
        ):
            # 单进程模式
            print(
                f"开始依次仿真 {self.config.simulation.batch} 个场景，每个场景 {self.config.simulation.steps} 步..."
            )
            simulator = Simulator(self.model, self.data)

            for i in range(self.config.simulation.batch):
                use_viewer = (
                    self.config.rendering.viewer and self.config.simulation.batch == 1
                )
                simulator.simulate_single_scene(
                    qpos_batch[i],
                    qvel_batch[i],
                    self.config.simulation.steps,
                    use_viewer,
                )
            results = [
                {
                    "scenes_processed": self.config.simulation.batch,
                    "elapsed_time": time.perf_counter() - start_time,
                    "viz_initial_qpos": qpos_batch[0],
                    "viz_final_qpos": self.data.qpos.copy(),
                }
            ]
        else:
            # 多进程模式
            # 确定实际的replicate数量：优先使用配置，否则从XML读取
            actual_replicate_count = (
                self.config.model.replicate_count
                if self.config.model.replicate_count is not None
                else self.get_replicate_count_from_xml(xml_path)
            )

            results = MultiprocessSimulator.run_multiprocess_simulation(
                xml_path,
                qpos_batch,
                qvel_batch,
                self.config.simulation.steps,
                num_workers,
                part_name,
                self.config.paths.npy_out_dir,
                actual_replicate_count,
            )

            if self.config.rendering.save_images:
                visualizer = Visualizer(self.model)
                visualizer.save_random_scene_renders(
                    results, self.config.paths.viz_out_dir, part_name, self.body_info
                )

        return results, time.perf_counter() - start_time

    def print_results(self, results, elapsed):
        """打印结果统计"""
        total_steps = self.config.simulation.batch * self.config.simulation.steps
        sim_time_total = total_steps * self.model.opt.timestep
        total_scenes_processed = sum(r.get("scenes_processed", 0) for r in results)
        real_time_factor = sim_time_total / elapsed

        print("\n---------- 仿真完成 ----------")
        print(f"模型时间步长 (Timestep): {self.model.opt.timestep:.4f} s")
        print(f"场景数量 (Batch): {self.config.simulation.batch}")
        print(f"实际处理场景数: {total_scenes_processed}")
        print(f"每场景步数 (Steps): {self.config.simulation.steps}")
        print(f"总仿真步数 (Total Steps): {total_steps:,}")
        print(f"总仿真时间 (Simulated Time): {sim_time_total:.2f} s")
        print(f"真实耗时 (Wall-Clock Time): {elapsed:.2f} s")
        print(f"性能 (Steps/s): {total_steps / elapsed:,.0f}")
        print(f"实时倍率 (Real-Time Factor): {real_time_factor:.1f}x")

        if self.config.multiprocessing.workers > 0 and len(results) > 1:
            print(f"\n--- 进程详情 ---")
            for i, result in enumerate(results):
                if "worker_pid" in result:
                    print(
                        f"进程 {i+1} (PID: {result['worker_pid']}): "
                        f"{result['scenes_processed']} 场景, "
                        f"{result.get('scenes_failed', 0)} 失败, "
                        f"{result['elapsed_time']:.2f}s"
                    )
        print("--------------------------------")

    def run(self):
        """运行完整的仿真流程"""
        try:
            # 检查是否需要处理XML文件
            if (
                hasattr(self.config_manager, "args")
                and self.config_manager.args.process_xml
            ):
                self.process_xml_files()
                return

            # 设置环境
            xml_path, part_name = self.setup()

            # 生成初始状态
            qpos_batch, qvel_batch = self.generate_initial_states(part_name)

            # 运行仿真
            results, elapsed = self.run_simulation(
                xml_path, qpos_batch, qvel_batch, part_name
            )

            # 打印结果
            self.print_results(results, elapsed)

        finally:
            # 清理临时XML文件
            save_modified_xml = getattr(self.config.model, "save_modified_xml", False)
            if (
                hasattr(self, "_temp_xml_path")
                and self._temp_xml_path
                and not save_modified_xml
            ):
                try:
                    if os.path.exists(self._temp_xml_path):
                        os.remove(self._temp_xml_path)
                        print(f"已删除临时XML文件: {self._temp_xml_path}")
                except Exception as e:
                    print(f"删除临时XML文件时出错: {e}")


def main():
    """主函数"""
    manager = MuJoCoSimulationManager()
    manager.run()


if __name__ == "__main__":
    main()
