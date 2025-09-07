import numpy as np
import subprocess
import os
import time
import math
import re
from typing import List, Dict


class MuJoCoUtils:
    """MuJoCo相关工具函数"""

    @staticmethod
    def setup_display():
        """设置虚拟显示和MuJoCo渲染环境"""
        display_num = ":1"

        # 检查是否已有 Xvfb 进程运行在指定显示上
        if MuJoCoUtils._is_xvfb_running(display_num):
            print(f"Xvfb 已在显示 {display_num} 上运行，跳过启动")
        else:
            print(f"启动 Xvfb 显示 {display_num}")
            try:
                subprocess.Popen(
                    ["Xvfb", display_num, "-screen", "0", "3840x2160x24"])
                # 等待 Xvfb 启动
                time.sleep(1)
            except Exception as e:
                print(f"启动 Xvfb 失败: {e}")

        os.environ["DISPLAY"] = display_num
        os.environ["MUJOCO_GL"] = "glfw"

    @staticmethod
    def _is_xvfb_running(display_num):
        """检查指定显示编号的 Xvfb 是否正在运行"""
        try:
            # 尝试使用 xdpyinfo 连接到 X 服务器
            # 如果成功，说明 X 服务器正在运行
            subprocess.check_call(
                ["xdpyinfo", "-display", display_num],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果 xdpyinfo 返回非零退出码 (表示无法连接) 或未找到命令
            return False
        except Exception as e:
            print(f"检查 Xvfb 状态时发生未知错误: {e}")
            return False

    @staticmethod
    def uniform_quat_wxyz() -> np.ndarray:
        """在 SO(3) 上均匀采样四元数，返回 [w,x,y,z]"""
        u1, u2, u3 = np.random.rand(3)
        qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        return np.array([qw, qx, qy, qz], dtype=np.float64)

    @staticmethod
    def yaw_to_quat(yaw: float) -> np.ndarray:
        """MuJoCo四元数顺序: [w, x, y, z]，仅绕Z轴旋转"""
        half = yaw * 0.5
        return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


class ModelAnalyzer:
    """模型分析工具"""

    def __init__(self, model):
        self.model = model

    def find_target_bodies(self, part_name: str) -> List[Dict]:
        """查找所有以 part_name 开头且带有关节的 body"""
        import mujoco

        body_info = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name or not name.startswith(part_name):
                continue
            if self.model.body_jntnum[i] <= 0:
                continue
            jnt_adr = self.model.body_jntadr[i]
            qpos_adr = self.model.jnt_qposadr[jnt_adr]
            body_info.append({"id": i, "name": name, "qpos_addr": qpos_adr})
        return body_info

    def get_ec8_collision_geoms(self, body_id: int) -> List[int]:
        """返回属于该 body 的、用于碰撞的 geoms"""
        gadr, gnum = self.model.body_geomadr[body_id], self.model.body_geomnum[body_id]
        geoms = []
        for g in range(gadr, gadr + gnum):
            if self.model.geom_group[g] == 3:
                geoms.append(g)
        return geoms

    def get_collision_geoms(self, body_id: int) -> List[int]:
        """返回属于该 body 的碰撞 geoms（按 group 判断）"""
        gadr, gnum = self.model.body_geomadr[body_id], self.model.body_geomnum[body_id]
        geoms = []
        for g in range(gadr, gadr + gnum):
            if self.model.geom_contype[g] | self.model.geom_conaffinity[g]:
                geoms.append(g)
        return geoms


class XmlProcessor:
    """XML处理工具"""

    @staticmethod
    def patch_replicate_count(xml_path: str, new_count: int) -> str:
        """
        在 scene_replicate.xml 中把第一个 <replicate count="..."> 改为 new_count，
        生成同目录下的副本文件并返回其路径，保证相对 include 路径不失效。
        """
        with open(xml_path, "r", encoding="utf-8") as f:
            txt = f.read()
        patched, n = re.subn(
            r'(<replicate\b[^>]*\bcount\s*=\s*")[0-9]+(")',
            rf"\g<1>{int(new_count)}\2",
            txt,
            count=1,
        )
        if n == 0:
            raise ValueError("未在XML中找到 <replicate count=...>；请检查文件。")
        dirn = os.path.dirname(xml_path)
        base, ext = os.path.splitext(os.path.basename(xml_path))
        out_path = os.path.join(dirn, f"{base}__rep{int(new_count)}{ext}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(patched)
        return out_path

    @staticmethod
    def indent_xml(elem, level=0):
        """简单美化XML输出，防止全在一行"""
        import xml.etree.ElementTree as ET

        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for e in elem:
                XmlProcessor.indent_xml(e, level + 1)
                if not e.tail or not e.tail.strip():
                    e.tail = i + "  "
            if not elem[-1].tail or not elem[-1].tail.strip():
                elem[-1].tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    @staticmethod
    def update_asset_mesh_scales(asset, scale_xyz=None) -> bool:
        """
        将 <asset> 下所有 <mesh> 的 scale 设为指定三元值（字符串如 '0.001 0.001 0.001'）。
        参数:
          - asset: <asset> 元素
          - scale_xyz: 长度为3的列表/元组，默认 [0.001, 0.001, 0.001]
        返回:
          - 是否发生了修改
        """
        if scale_xyz is None:
            scale_xyz = [0.001, 0.001, 0.001]
        if len(scale_xyz) != 3:
            raise ValueError("scale_xyz must have 3 elements")
        scale_str = " ".join(f"{float(v):g}" for v in scale_xyz)

        changed = False
        for m in asset.findall("mesh"):
            if m.get("scale") != scale_str:
                m.set("scale", scale_str)
                changed = True
        return changed

    @staticmethod
    def process_dependencies_xml(path, mesh_scale=None) -> bool:
        """处理 *_dependencies.xml 文件"""
        import xml.etree.ElementTree as ET

        if mesh_scale is None:
            mesh_scale = [0.001, 0.001, 0.001]

        name = os.path.basename(path).removesuffix("_dependencies.xml")
        tree = ET.parse(path)
        root = tree.getroot()

        asset = root.find(".//asset")
        if asset is None:
            print(f"[skip] no <asset> in {path}")
            return False

        meshes = [m for m in asset if m.tag == "mesh"]
        if not meshes:
            print(f"[skip] no <mesh> entries in {path}")
            return False

        # 找到 asset 中第一个 <mesh> 的索引
        first_mesh_idx = None
        for idx, child in enumerate(list(asset)):
            if child.tag == "mesh":
                first_mesh_idx = idx
                break

        first_mesh = meshes[0]
        # 推断目录与 scale
        file_attr = first_mesh.get("file", "")
        mesh_dir = os.path.dirname(file_attr)
        scale = first_mesh.get("scale") or "1 1 1"
        target_file = os.path.join(mesh_dir, f"{name}.obj").replace("\\", "/")

        # 查找是否已有目标 mesh 以及其在 asset 中的位置
        target_mesh = None
        target_idx = None
        for idx, child in enumerate(list(asset)):
            if child.tag != "mesh":
                continue
            f = child.get("file", "")
            if os.path.basename(f) == f"{name}.obj":
                target_mesh = child
                target_idx = idx
                break

        changed = False
        if target_mesh is None:
            # 不存在则插入到第一个 <mesh> 之前
            new_mesh = ET.Element(
                "mesh", {"file": target_file, "scale": scale})
            if first_mesh_idx is None:
                asset.append(new_mesh)
            else:
                asset.insert(first_mesh_idx, new_mesh)
            changed = True
            print(
                f"[write] inserted original mesh into {path}: file={target_file}")
        else:
            # 已存在但不在最前则移动到第一个 <mesh> 之前
            if first_mesh_idx is not None and target_idx != first_mesh_idx:
                asset.remove(target_mesh)
                # 重新定位当前第一个 <mesh> 的索引
                new_first_idx = None
                for idx, child in enumerate(list(asset)):
                    if child.tag == "mesh":
                        new_first_idx = idx
                        break
                if new_first_idx is None:
                    asset.append(target_mesh)
                else:
                    asset.insert(new_first_idx, target_mesh)
                changed = True
                print(
                    f"[reorder] moved original mesh to front in {path}: file={target_file}"
                )
            else:
                print(f"[ok] {path} original mesh already at front")

        # 统一所有 mesh 的 scale
        if XmlProcessor.update_asset_mesh_scales(asset, mesh_scale):
            changed = True
            print(f"[write] normalized mesh scales in {path}")

        if changed:
            XmlProcessor.indent_xml(root)
            tree.write(path, encoding="utf-8", xml_declaration=True)
        return changed

    @staticmethod
    def process_body_xml(path) -> bool:
        """处理 *_body.xml 文件"""
        import xml.etree.ElementTree as ET

        name = os.path.basename(path).removesuffix("_body.xml")
        tree = ET.parse(path)
        root = tree.getroot()

        body = root.find(".//body")
        if body is None:
            print(f"[skip] no <body> in {path}")
            return False

        children = list(body)

        # 定位第一个 <geom> 的索引
        first_geom_idx = None
        for idx, c in enumerate(children):
            if c.tag == "geom":
                first_geom_idx = idx
                break

        # 查找是否已有 mesh=name 的 geom 及其索引
        target_geom = None
        target_idx = None
        for idx, c in enumerate(children):
            if c.tag == "geom" and c.get("mesh") == name:
                target_geom = c
                target_idx = idx
                break

        changed = False
        if target_geom is None:
            # 不存在则创建并插入到第一个 <geom> 之前
            new_geom = ET.Element(
                "geom", {"mesh": name, "class": "visual",
                         "rgba": "1.0 1.0 1.0 0"}
            )
            if first_geom_idx is None:
                body.append(new_geom)
            else:
                body.insert(first_geom_idx, new_geom)
            changed = True
            print(f"[write] inserted original geom into {path}: mesh={name}")
        else:
            # 已存在但不在第一个 <geom> 位置则移动
            if first_geom_idx is not None and target_idx != first_geom_idx:
                body.remove(target_geom)
                # 重新定位当前第一个 <geom>
                new_first_idx = None
                for idx, c in enumerate(list(body)):
                    if c.tag == "geom":
                        new_first_idx = idx
                        break
                if new_first_idx is None:
                    body.append(target_geom)
                else:
                    body.insert(new_first_idx, target_geom)
                changed = True
                print(
                    f"[reorder] moved original geom to front in {path}: mesh={name}")
            else:
                print(f"[ok] {path} original geom already at front")

        if changed:
            XmlProcessor.indent_xml(root)
            tree.write(path, encoding="utf-8", xml_declaration=True)
        return changed

    @staticmethod
    def process_scene_xml(path: str, texture_width: int = 100, texture_height: int = 100,
                          ground_size: str = "1 1 0.1") -> bool:
        """处理 *_replicate.xml 文件"""
        import xml.etree.ElementTree as ET

        tree = ET.parse(path)
        root = tree.getroot()
        changed = False

        # 1) 处理 <asset>/<texture name="grid">
        asset = root.find(".//asset")
        if asset is None:
            print(f"[skip] no <asset> in {path}")
        else:
            # 找到名为 grid 的 texture
            tex = None
            for child in asset.findall("texture"):
                if child.get("name") == "grid":
                    tex = child
                    break
            if tex is None:
                print(f"[skip] no <texture name='grid'> in {path}")
            else:
                w = tex.get("width")
                h = tex.get("height")
                need_w = w != str(texture_width)
                need_h = h != str(texture_height)
                if need_w or need_h:
                    tex.set("width", str(texture_width))
                    tex.set("height", str(texture_height))
                    changed = True
                    print(
                        f"[write] update texture size in {path} -> width={texture_width}, height={texture_height}"
                    )

            # 2) 删除 *_model.xml 的 <model>
            removed_any = False
            for child in list(asset):
                if child.tag != "model":
                    continue
                file_attr = child.get("file", "")
                if file_attr.endswith("_model.xml"):
                    asset.remove(child)
                    removed_any = True
            if removed_any:
                changed = True
                print(f"[write] remove <model ... _model.xml> in {path}")

        # 3) 处理 <worldbody>/<geom name="ground" type="plane">
        worldbody = root.find(".//worldbody")
        if worldbody is None:
            print(f"[skip] no <worldbody> in {path}")
        else:
            ground = None
            for g in worldbody.findall("geom"):
                if g.get("name") == "ground" and g.get("type") == "plane":
                    ground = g
                    break
            if ground is None:
                print(f"[skip] no ground plane geom in {path}")
            else:
                if ground.get("size") != ground_size:
                    ground.set("size", ground_size)
                    changed = True
                    print(
                        f"[write] update ground size in {path} -> {ground_size}")

        if changed:
            XmlProcessor.indent_xml(root)
            tree.write(path, encoding="utf-8", xml_declaration=True)
        else:
            print(f"[ok] {path} already satisfied")
        return changed
