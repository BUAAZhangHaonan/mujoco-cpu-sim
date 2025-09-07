import yaml
import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SimulationConfig:
    """仿真配置"""
    steps: int
    batch: int
    seed: int


@dataclass
class ModelConfig:
    """模型配置"""
    path: str
    part_name: Optional[str]
    replicate_count: Optional[int]
    save_modified_xml: bool


@dataclass
class MultiprocessingConfig:
    """多进程配置"""
    workers: int


@dataclass
class InitializationConfig:
    """初始化配置"""
    # 六角密堆参数
    xy_extent: List[float]
    z_low: float
    layer_gap: float
    min_clearance: float
    jitter_frac: float
    
    # 姿态控制
    allow_3d_rot: bool
    yaw_only: bool
    
    # 拟合模式
    fit_mode: str


@dataclass
class RenderingConfig:
    """渲染配置"""
    viewer: bool
    save_images: bool
    image_resolution: dict


@dataclass
class PathsConfig:
    """路径配置"""
    output_dir: str
    npy_out_dir: str
    viz_out_dir: str


@dataclass
class XmlProcessingConfig:
    """XML处理配置"""
    # add_orign_obj_to_xml.py 相关配置
    mesh_scale: List[float]
    models_dir: str
    
    # change_replicate_xml_ground.py 相关配置
    scenes_dir: str
    texture_width: int
    texture_height: int
    ground_size: str


@dataclass
class Config:
    """总配置"""
    model: ModelConfig
    simulation: SimulationConfig
    multiprocessing: MultiprocessingConfig
    initialization: InitializationConfig
    rendering: RenderingConfig
    paths: PathsConfig
    xml_processing: XmlProcessingConfig


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.args = None

    def load_config(self) -> Config:
        """加载配置文件"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return Config(
            model=ModelConfig(**config_dict["model"]),
            simulation=SimulationConfig(**config_dict["simulation"]),
            multiprocessing=MultiprocessingConfig(**config_dict["multiprocessing"]),
            initialization=InitializationConfig(**config_dict["initialization"]),
            rendering=RenderingConfig(**config_dict["rendering"]),
            paths=PathsConfig(**config_dict["paths"]),
            xml_processing=XmlProcessingConfig(**config_dict["xml_processing"]),
        )

    def parse_args_and_merge_config(self) -> Config:
        """解析命令行参数并与配置文件合并"""
        parser = argparse.ArgumentParser(description="MuJoCo多进程仿真与XML处理工具")

        # 配置文件路径
        parser.add_argument(
            "--config", type=str, default="config.yaml", help="配置文件路径"
        )

        # 模型相关
        parser.add_argument("--model_path", type=str, help="MuJoCo XML模型文件路径")
        parser.add_argument("--part_name", type=str, help="要随机化的零件名前缀")
        parser.add_argument("--replicate-count", type=int, help="动态覆盖XML中的replicate数量")

        # 仿真相关
        parser.add_argument("--steps", type=int, help="每个场景要仿真的步数")
        parser.add_argument("--batch", type=int, help="要仿真的场景数量")
        parser.add_argument("--workers", type=int, help="并行进程数量")
        parser.add_argument("--seed", type=int, help="随机种子")

        # 初始化相关
        parser.add_argument("--xy-extent", type=float, nargs=2, help="XY放置范围：[min, max]")
        parser.add_argument("--z-low", type=float, help="第一层的z起始高度")
        parser.add_argument("--layer-gap", type=float, help="不同Z层之间的额外安全间隙")
        parser.add_argument("--min-clearance", type=float, help="不同零件之间的最小XY间隙")
        parser.add_argument("--jitter-frac", type=float, help="单元内XY抖动占间隙比例")
        
        # 姿态控制
        parser.add_argument("--allow-3d-rot", action="store_true", help="允许三轴随机旋转")
        parser.add_argument("--no-allow-3d-rot", dest="allow_3d_rot", action="store_false", help="禁用三轴随机旋转")
        parser.add_argument("--yaw-only", action="store_true", help="仅绕Z轴旋转")
        
        # 拟合模式
        parser.add_argument("--fit-mode", choices=["sphere", "aabb"], help="几何拟合模式")

        # 渲染相关
        parser.add_argument("--viewer", action="store_true", help="是否显示可视化界面")
        
        # 路径相关
        parser.add_argument("--output-dir", type=str, help="输出目录")
        parser.add_argument("--npy-out-dir", type=str, help=".npy输出目录")
        parser.add_argument("--viz-out-dir", type=str, help="可视化输出目录")

        # XML处理相关
        parser.add_argument("--process-xml", action="store_true", help="运行XML处理功能")
        parser.add_argument("--mesh-scale", type=float, nargs=3, help="网格缩放")
        parser.add_argument("--models-dir", type=str, help="模型目录")
        parser.add_argument("--scenes-dir", type=str, help="场景目录")
        parser.add_argument("--texture-width", type=int, help="纹理宽度")
        parser.add_argument("--texture-height", type=int, help="纹理高度")
        parser.add_argument("--ground-size", type=str, help="地面尺寸")

        self.args = parser.parse_args()

        # 更新配置文件路径
        self.config_path = self.args.config

        # 加载基础配置
        config = self.load_config()

        # 用命令行参数覆盖配置
        if self.args.model_path is not None:
            config.model.path = self.args.model_path
        if self.args.part_name is not None:
            config.model.part_name = self.args.part_name
        if self.args.replicate_count is not None:
            config.model.replicate_count = self.args.replicate_count

        if self.args.steps is not None:
            config.simulation.steps = self.args.steps
        if self.args.batch is not None:
            config.simulation.batch = self.args.batch
        if self.args.workers is not None:
            config.multiprocessing.workers = self.args.workers
        if self.args.seed is not None:
            config.simulation.seed = self.args.seed

        if self.args.xy_extent is not None:
            config.initialization.xy_extent = self.args.xy_extent
        if self.args.z_low is not None:
            config.initialization.z_low = self.args.z_low
        if self.args.layer_gap is not None:
            config.initialization.layer_gap = self.args.layer_gap
        if self.args.min_clearance is not None:
            config.initialization.min_clearance = self.args.min_clearance
        if self.args.jitter_frac is not None:
            config.initialization.jitter_frac = self.args.jitter_frac

        if self.args.allow_3d_rot is not None:
            config.initialization.allow_3d_rot = self.args.allow_3d_rot
        if self.args.yaw_only is not None:
            config.initialization.yaw_only = self.args.yaw_only
        if self.args.fit_mode is not None:
            config.initialization.fit_mode = self.args.fit_mode

        if self.args.viewer is not None:
            config.rendering.viewer = self.args.viewer

        if self.args.output_dir is not None:
            config.paths.output_dir = self.args.output_dir
        if self.args.npy_out_dir is not None:
            config.paths.npy_out_dir = self.args.npy_out_dir
        if self.args.viz_out_dir is not None:
            config.paths.viz_out_dir = self.args.viz_out_dir

        if self.args.mesh_scale is not None:
            config.xml_processing.mesh_scale = self.args.mesh_scale
        if self.args.models_dir is not None:
            config.xml_processing.models_dir = self.args.models_dir
        if self.args.scenes_dir is not None:
            config.xml_processing.scenes_dir = self.args.scenes_dir
        if self.args.texture_width is not None:
            config.xml_processing.texture_width = self.args.texture_width
        if self.args.texture_height is not None:
            config.xml_processing.texture_height = self.args.texture_height
        if self.args.ground_size is not None:
            config.xml_processing.ground_size = self.args.ground_size

        return config
