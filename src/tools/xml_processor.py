import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


class XMLProcessor:
    def __init__(self, assets_path, count: int=5, scale: float=0.01):
        """
        初始化XML处理器
        
        Args:
            assets_path (str): assets文件夹的路径
        """
        self.assets_path = assets_path
        self.count = count
        self.scale = scale
        self.mesh_path = os.path.join(assets_path, "mesh")
        self.mjcf_path = os.path.join(assets_path, "mjcf")
        self.texture_path = os.path.join(assets_path, "texture")
        self.models_path = os.path.join(self.mjcf_path, "models")
        self.scenes_path = os.path.join(self.mjcf_path, "scenes")
        
        # 确保目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录结构存在"""
        for path in [self.mjcf_path, self.models_path, self.scenes_path]:
            os.makedirs(path, exist_ok=True)
    
    def process_all_parts(self):
        """处理所有零件并生成XML文件"""
        # 获取mesh文件夹下的所有子文件夹（零件）
        part_folders = [f for f in os.listdir(self.mesh_path) 
                       if os.path.isdir(os.path.join(self.mesh_path, f))]
        
        # 使用tqdm显示进度
        success_count = 0
        for part_name in tqdm(part_folders, desc="处理零件"):
            success = self.process_part(part_name)
            if success:
                success_count += 1
        
        print(f"成功处理 {success_count}/{len(part_folders)} 个零件")
        return success_count
    
    def process_part(self, part_name):
        """
        处理单个零件，生成所需的XML文件
        
        Args:
            part_name (str): 零件名称
            
        Returns:
            bool: 处理是否成功
        """
        # 检查零件文件夹下的XML文件
        part_xml_path = os.path.join(self.mesh_path, part_name, f"{part_name}.xml")
        
        if not os.path.exists(part_xml_path):
            print(f"警告: 零件 {part_name} 的XML文件未找到: {part_xml_path}")
            return False
        
        # 解析XML文件
        part_info = self._parse_part_xml(part_xml_path, part_name)
        if not part_info:
            return False
        
        # 生成三个XML文件
        deps_success = self._generate_dependencies_xml(part_name, part_info)
        body_success = self._generate_body_xml(part_name, part_info)
        scene_success = self._generate_replicate_xml(part_name)
        
        # 测试生成的文件
        test_success = self._test_xml_files(part_name)
        
        return deps_success and body_success and scene_success and test_success
    
    def _parse_part_xml(self, xml_path, part_name):
        """
        解析零件的XML文件，提取所需信息
        
        Args:
            xml_path (str): XML文件路径
            part_name (str): 零件名称
            
        Returns:
            dict: 包含零件信息的字典
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 收集mesh文件和材质信息
            meshes = []
            materials = {}
            collision_meshes = []
            visual_meshes = []
            mesh_material_mapping = {}
            
            asset_node = root.find(".//asset")
            if asset_node is None:
                raise ValueError(f"XML文件中未找到asset节点: {xml_path}")
            
            for mesh in asset_node.findall("mesh"):
                file_path = mesh.get("file")
                if file_path:
                    file_name = os.path.basename(file_path)
                    mesh_name = os.path.splitext(file_name)[0]
                    scale_value = getattr(self, "scale", 0.001)
                    scale = mesh.get("scale", f"{scale_value} {scale_value} {scale_value}")
                    meshes.append({
                        "name": mesh_name,
                        "file": f"{part_name}/{file_name}",
                        "scale": scale
                    })
                    if "collision" in mesh_name:
                        collision_meshes.append(mesh_name)
                    else:
                        visual_meshes.append(mesh_name)
        
            for material in asset_node.findall("material"):
                name = material.get("name")
                rgba = material.get("rgba")
                specular = material.get("specular", "0.2")
                shininess = material.get("shininess", "0.2")
                
                if name and rgba:
                    materials[name] = {
                        "rgba": rgba,
                        "specular": specular,
                        "shininess": shininess
                    }
            
            worldbody = root.find(".//worldbody")
            if worldbody is not None:
                body = worldbody.find(".//body")
                if body is not None:
                    for geom in body.findall("geom"):
                        mesh_name = geom.get("mesh")
                        material_name = geom.get("material")
                        if mesh_name and material_name and "collision" not in mesh_name:
                            mesh_material_mapping[mesh_name] = material_name
            
            return {
                "meshes": meshes,
                "materials": materials,
                "collision_meshes": collision_meshes,
                "visual_meshes": visual_meshes,
                "mesh_material_mapping": mesh_material_mapping
            }
        
        except Exception as e:
            print(f"解析XML文件 {xml_path} 时出错: {e}")
            return None
    
    def _generate_dependencies_xml(self, part_name, part_info):
        """
        生成_dependencies.xml文件
        
        Args:
            part_name (str): 零件名称
            part_info (dict): 零件信息
            
        Returns:
            bool: 生成是否成功
        """
        output_path = os.path.join(self.models_path, f"{part_name}_dependencies.xml")
        
        root = ET.Element("mujocoinclude")
        
        default = ET.SubElement(root, "default")
        
        visual_default = ET.SubElement(default, "default", attrib={"class": "visual"})
        ET.SubElement(visual_default, "geom", attrib={
            "group": "2",
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0"
        })
        
        collision_default = ET.SubElement(default, "default", attrib={"class": "collision"})
        ET.SubElement(collision_default, "geom", attrib={
            "group": "3",
            "type": "mesh",
            "density": "2000",
            "condim": "3",
            "friction": "0.6 0.01 0.005",
            "margin": "0.001",
            "solimp": "0.95 0.995 0.001",
            "solref": "0.01 2"
        })
        
        asset = ET.SubElement(root, "asset")
        
        for name, material in part_info["materials"].items():
            ET.SubElement(asset, "material", attrib={
                "name": name,
                "rgba": material["rgba"],
                "specular": material["specular"],
                "shininess": material["shininess"]
            })
        
        scale_value = getattr(self, "scale", 0.001)
        ET.SubElement(asset, "mesh", attrib={
            "file": f"{part_name}/{part_name}.obj",
            "scale": f"{scale_value} {scale_value} {scale_value}"
        })
        
        for mesh in part_info["meshes"]:
            ET.SubElement(asset, "mesh", attrib={
                "file": mesh["file"],
                "scale": mesh["scale"]
            })
        
        self._write_xml_file(root, output_path)
        return True
    
    def _generate_body_xml(self, part_name, part_info):
        """
        生成_body.xml文件
        
        Args:
            part_name (str): 零件名称
            part_info (dict): 零件信息
            
        Returns:
            bool: 生成是否成功
        """
        output_path = os.path.join(self.models_path, f"{part_name}_body.xml")
        
        root = ET.Element("mujocoinclude")
        
        body = ET.SubElement(root, "body", attrib={"name": part_name})
        ET.SubElement(body, "freejoint")
        
        ET.SubElement(body, "geom", attrib={
            "mesh": part_name,
            "class": "visual",
            "rgba": "1.0 1.0 1.0 0"
        })
        
        for mesh_name in part_info["visual_meshes"]:
            if mesh_name != part_name and "collision" not in mesh_name:
                material = part_info["mesh_material_mapping"].get(mesh_name)
                
                if material:
                    ET.SubElement(body, "geom", attrib={
                        "mesh": mesh_name,
                        "material": material,
                        "class": "visual"
                    })
        
        for mesh_name in part_info["collision_meshes"]:
            ET.SubElement(body, "geom", attrib={
                "mesh": mesh_name,
                "class": "collision"
            })
        
        self._write_xml_file(root, output_path)
        return True
    
    def _generate_replicate_xml(self, part_name):
        """
        生成_replicate.xml文件
        
        Args:
            part_name (str): 零件名称
            
        Returns:
            bool: 生成是否成功
        """
        output_path = os.path.join(self.scenes_path, f"{part_name}_scene_replicate.xml")
        
        root = ET.Element("mujoco", attrib={"model": f"Falling {part_name}s Scene"})
        
        ET.SubElement(root, "compiler", attrib={
            "meshdir": "../../mesh/",
            "texturedir": "../../texture/"
        })
        
        ET.SubElement(root, "include", attrib={
            "file": f"../models/{part_name}_dependencies.xml"
        })
        
        ET.SubElement(root, "option", attrib={
            "gravity": "0 0 -9.81",
            "timestep": "0.0005",
            "solver": "Newton"
        })
        
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", attrib={
            "offheight": "1080",
            "offwidth": "1920"
        })
        ET.SubElement(visual, "quality", attrib={
            "offsamples": "8",
            "shadowsize": "4096"
        })
        
        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "texture", attrib={
            "name": "grid",
            "type": "2d",
            "builtin": "checker",
            "rgb1": ".1 .2 .3",
            "rgb2": ".2 .3 .4",
            "width": "100",
            "height": "100"
        })
        ET.SubElement(asset, "material", attrib={
            "name": "grid_material",
            "texture": "grid",
            "texrepeat": "10 10",
            "texuniform": "true",
            "reflectance": ".2"
        })
        
        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light", attrib={
            "pos": "0 0 20",
            "dir": "0 0 -1",
            "diffuse": "1 1 1",
            "specular": "1 1 1",
            "castshadow": "true"
        })
        ET.SubElement(worldbody, "geom", attrib={
            "name": "ground",
            "type": "plane",
            "size": "10 10 0.1",
            "material": "grid_material"
        })
        
        replicate = ET.SubElement(worldbody, "replicate", attrib={"count": str(self.count)})
        ET.SubElement(replicate, "include", attrib={
            "file": f"../models/{part_name}_body.xml"
        })
        
        self._write_xml_file(root, output_path)
        return True
    
    def _write_xml_file(self, root, output_path):
        """
        将XML树写入文件，确保正确的XML声明和格式化
        
        Args:
            root (ET.Element): XML根节点
            output_path (str): 输出文件路径
        """
        def indent(elem, level=0):
            i = "\n" + level*"    "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "    "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for child in elem:
                    indent(child, level+1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i
        
        indent(root)
        xml_str = ET.tostring(root, encoding='unicode')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("<?xml version='1.0' encoding='utf-8'?>\n")
            f.write(xml_str)
    
    def _test_xml_files(self, part_name):
        """
        测试生成的XML文件是否有效
        
        Args:
            part_name (str): 零件名称
            
        Returns:
            bool: 测试是否通过
        """
        try:
            dependencies_path = os.path.join(self.models_path, f"{part_name}_dependencies.xml")
            body_path = os.path.join(self.models_path, f"{part_name}_body.xml")
            replicate_path = os.path.join(self.scenes_path, f"{part_name}_scene_replicate.xml")
            
            assert os.path.exists(dependencies_path), f"未找到{part_name}的依赖文件"
            assert os.path.exists(body_path), f"未找到{part_name}的body文件"
            assert os.path.exists(replicate_path), f"未找到{part_name}的场景文件"
            
            ET.parse(dependencies_path)
            ET.parse(body_path)
            ET.parse(replicate_path)
            
            return True
        except Exception as e:
            print(f"零件{part_name}的测试失败: {e}")
            return False
