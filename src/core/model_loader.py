import mujoco
import os


class ModelLoader:
    """MuJoCo模型加载器"""

    @staticmethod
    def load_model(xml_path: str):
        """加载MuJoCo模型"""
        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)
            return model, data
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\n请确保：")
            print("1. 你的文件结构与说明一致。")
            print(
                "2. `assets/mesh` 和 `assets/texture` 目录中包含了所有的 .obj 和 .png 文件。"
            )
            exit()

    @staticmethod
    def resolve_model_path(model_path: str) -> str:
        """解析模型路径"""
        try:
            # 获取当前脚本所在目录 (0821/)
            script_dir = os.path.dirname(os.path.realpath(__file__))
            # 获取项目根目录 (mjwarp-test/)
            project_root = os.path.dirname(script_dir)
            return os.path.join(project_root, model_path)
        except NameError:
            return model_path
