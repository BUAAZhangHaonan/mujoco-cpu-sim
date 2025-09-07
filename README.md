# 基于MuJoCo引擎的高性能CPU物理模拟框架 (mujoco-cpu-sim)

这是一个基于MuJoCo物理引擎的CPU模拟项目，专注于高效的多场景物理模拟、物体堆叠和动态放置优化。项目采用模块化设计，支持多进程并行计算，适用于机器人学、计算机视觉和物理模拟研究。

## 项目主要内容

### 核心功能

- **物理模拟**：基于MuJoCo引擎进行高精度物理模拟，支持刚体动力学、碰撞检测和约束求解
- **多场景批量处理**：支持同时运行多个模拟场景，提高计算效率
- **动态模型加载**：自动解析和加载MuJoCo XML模型文件，支持路径解析和模型分析
- **状态初始化**：智能生成非重叠初始状态，支持随机和规则放置策略
- **可视化与导出**：提供实时可视化和位姿数据导出功能，便于结果分析

### 应用场景

- 物体堆叠模拟
- 机器人抓取规划
- 场景布局优化
- 物理参数验证
- 数据集生成

## 创新点

### 1. 六角密堆初始化算法

- **创新描述**：采用六角密堆(Hexagonal Close Packing)算法生成高密度非重叠初始状态，相较于随机放置可显著提高空间利用率和模拟稳定性
- **技术优势**：
  - 减少碰撞冲突，提高模拟成功率
  - 优化空间布局，模拟更真实的物理场景
  - 支持多层堆叠和姿态随机化

### 2. 多进程并行模拟框架

- **创新描述**：基于Python multiprocessing模块实现多进程并行模拟，支持动态负载均衡
- **技术优势**：
  - 充分利用多核CPU资源，显著加速批量模拟
  - 内存隔离设计，避免进程间干扰
  - 进度监控和错误处理机制

### 3. 动态XML处理系统

- **创新描述**：支持运行时动态修改XML模型参数，无需预先编辑文件
- **技术优势**：
  - 灵活配置replicate数量，支持大规模场景生成
  - 临时文件管理，保持原始文件完整性
  - 正则表达式解析，提高处理效率

### 4. 自适应几何优化

- **创新描述**：结合包围球(Bounding Sphere)和轴对齐包围盒(AABB)进行几何计算和碰撞检测
- **技术优势**：
  - 对复杂网格模型提供稳定的几何近似
  - 自适应选择最优拟合模式(sphere/aabb)
  - 高效的碰撞检测算法

## 核心技术实现

### 架构设计

```text
src/
├── main.py              # 主入口，仿真管理器
├── core/
│   ├── simulation.py    # 模拟引擎，多进程支持
│   ├── model_loader.py  # 模型加载和路径解析
│   ├── geometry.py      # 几何计算和边界检测
│   ├── initialization.py# 状态初始化算法
│   ├── visualization.py # 可视化和渲染
│   └── utils.py         # 工具函数和辅助类
├── config/
│   ├── config_manager.py# 配置管理和CLI解析
│   └── config.yaml      # 默认配置文件
└── tools/
    ├── xml_processor.py # XML处理工具
    └── obj_selector.py  # 物体选择器
```

### 关键技术栈

- **物理引擎**：MuJoCo - 高性能多关节动力学模拟
- **并行计算**：Python multiprocessing - 多进程并行框架
- **数值计算**：NumPy - 高效数组操作和数学计算
- **配置管理**：PyYAML - YAML配置文件解析
- **进度监控**：tqdm - 进度条和状态显示
- **可视化**：matplotlib, mediapy - 结果可视化

### 核心算法实现

#### 六角密堆初始化

```python
# 使用六角密堆算法生成非重叠位置
initializer = HexPackedInitializer(
    model=self.model,
    body_info=body_info,
    xy_min=xy_min, xy_max=xy_max,
    clearance=min_clearance,
    layer_gap=layer_gap
)
qpos_batch, qvel_batch = initializer.generate(batch_size, z_low)
```

##### 背景 & 必要性

- 目标：在 **MuJoCo** 中为大量电子元件做“**高密度且绝不穿模**”的随机初始化，再进行自由落体仿真与可视化导出。
- 现实问题：传统“随机投点→`mj_forward`→局部松弛/推开”的套路，一旦同时体数多、初始重叠多，会瞬间制造海量接触约束，导致内存/栈分配在构建/求解约束系统时爆表（典型错误里会看到 `mj_stackAlloc`、`nefc`、`ncon` 爆增）。MuJoCo 的接触与约束是按牛顿法求解的，接触多就会拉高系统规模与内存压力，这是机制问题，不是“调几个参数能彻底解决”的小毛病。
- 同时，你的 XML 利用 `<replicate count="N">` 一次性实例化很多零件；视觉几何（`class="visual"`）通过 `contype="0" conaffinity="0"` 从碰撞管线里剔除，仅作渲染/位姿导出，这个建模方式完全合理。

##### 灵感来源（设计基石）

1. **二维最密堆积**：平面上同尺度圆的极限密度是**六角密堆**（蜂窝格）；把每个零件的“占地”近似成等效圆，先在 XY 平面六角铺设，再做多层 Z 叠放，天生零重叠、密度高。
2. **蓝噪声/泊松盘**：需要“更随机的均匀性”时，采用 Bridson 的 **Poisson-disk** 采样（邻域栅格 + 有限候选，均摊 O(N)），在单层/局部区域内生成互不靠太近的中心。
3. **SO(3) 随机姿态**：完整的三轴随机，用 Shoemake 的 **均匀四元数采样**（Graphics Gems III）。MuJoCo 四元数约定是 **(w, x, y, z)**。
4. **安全尺寸估计**：直接读每个碰撞 geom 的 **`geom_rbound`**（包围球半径，用于快速排除碰撞）当作保守占地尺度；AABB 不稳就回退 rbound，保证零重叠的**几何构造**先行。
5. **视觉透明控制**：`rgba` 的 alpha=0 可让对象不可见（用于“仅导出位姿、不参与渲染/传感”的视觉几何）。(

##### 解决方案（我们现在采用的方案）

###### **HexPackedInitializer（六角密堆+分层）**

- 用 `geom_rbound` 为每个零件取“占地等效半径”与“等效高度”的保守上界；取全局最大半径 R 与最大高度 H。
- 在 \[xy\_min, xy\_max]^2 上按 **六角格** 以步距 `2R + clearance` 铺设圆心，加入**微抖动**（不破坏不相交下界），得到**每层**的可用中心列表。
- 按容量把 N 个零件分配到若干 **Z 层**（层间距 `H + layer_gap`），写入 `qpos=[x,y,z, quat]`。
- 姿态：默认启用 **全三轴**（Shoemake）；如需稳定落料，可退化为仅绕 Z 的 yaw 随机。
- **完全不调用** `mj_forward` 做“推开/松弛”。**零重叠是几何保证**，这样避免了初始化阶段制造巨量接触与 Hessian 代价。

###### **可视化与数据导出**

- 碰撞由凸分解子 obj 负责；原始 obj 设为 `class="visual"` + `rgba="1 1 1 0"`（不可见），仅用于**最终位姿导出**。碰撞/渲染与导出职责彻底解耦。
- 仿真结束后，逐个 body 读取 **自由关节的 7D `qpos`**（pos+quat）保存为 `npy`。还原函数加载 `npy`，把相应 body 的 `qpos` 批量写回，即可 1:1 复原“视觉零件”场景用于**误差对比渲染**。
- 目录组织：`<npy根>/<零件名>/<数量>/scene_<idx>.npy`，保持**数量可追溯**与**恢复便捷**。

**动态修改数量（加载前覆盖 `<replicate>` 的 `count`）**

- 在加载 XML 前以解析或正则方式，把场景 `<replicate count="K">` 的 `K` **替换为命令行传入的数量**，再加载模型。MuJoCo 官方 XML 参考明确了 `replicate` 的作用与语法。

##### 可调参数（现在脚本里已经有）

- `xy_extent`（放置边界）、`min_clearance`（安全间隙）、`layer_gap`（层间距）、`z_low`（底层高度）、`jitter_frac`（单元内抖动幅度）、`seed`；
- `yaw_only`（=False 开启三轴随机）/`use_poisson`（在局部使用 Bridson 采样替换规则格点，获得更“蓝噪声”的分布）。

##### 伪代码（核心流程）

```pseudo
function LOAD_AND_PATCH_XML(path, part_count):
    xml_str = read_file(path)
    xml_str = replace_first('<replicate count="[^"]+">', f'<replicate count="{part_count}">', xml_str)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data  = mujoco.MjData(model)
    return model, data

function ESTIMATE_SAFE_SIZE(model, body):
    # collect colliding geoms of this body
    r = max(geom.rbound for geom in colliding_geoms(body))  # conservative
    H = 2 * r
    return (r, H)  # XY radius, Z height

function BUILD_HEX_GRID(xy_min, xy_max, R, clearance):
    step = 2R + clearance
    dx = step
    dy = sqrt(3)/2 * step
    margin = R + 0.5*clearance
    centers = []
    for row in 0..:
        y = xy_min + margin + row*dy
        if y > xy_max - margin: break
        x0 = xy_min + margin + (0 if row even else 0.5*dx)
        for col in 0..:
            x = x0 + col*dx
            if x > xy_max - margin: break
            centers.append((x, y))
    return centers  # one layer

function RANDOM_QUATERNION_SHOEMAKE():
    # sample uniform over SO(3) (wxyz)
    u1, u2, u3 ~ Uniform(0,1)
    q = [cos(2πu2)*sqrt(1-u1), sin(2πu2)*sqrt(1-u1),
         cos(2πu3)*sqrt(u1),   sin(2πu3)*sqrt(u1)]
    return [w,x,y,z] = [q3, q0, q1, q2]  # ensure wxyz ordering in MuJoCo

function INIT_SCENES(model, bodies, B, xy_min, xy_max, z_low, clearance, layer_gap, jitter_frac, use_poisson):
    # 1) size upper-bounds
    R_list, H_list = [], []
    for b in bodies:
        r, h = ESTIMATE_SAFE_SIZE(model, b)
        R_list.add(r); H_list.add(h)
    R = max(R_list); H = max(H_list)

    # 2) centers per layer
    C_layer = BUILD_HEX_GRID(xy_min, xy_max, R, clearance)

    # 3) how many layers
    cap = len(C_layer)
    L = ceil(len(bodies)/cap)

    qpos_batch = zeros((B, model.nq))
    qvel_batch = zeros((B, model.nv))

    for b_idx in 0..B-1:
        qpos = model.qpos0.copy()
        perm_bodies = shuffle(bodies)
        perm_centers = shuffle(C_layer)

        k = 0
        for layer in 0..L-1:
            z = z_low + layer*(H + layer_gap)
            take = min(cap, len(bodies)-k)
            for i in 0..take-1:
                body = perm_bodies[k]; cx, cy = perm_centers[i]
                (jx) = body.qpos_addr
                # jitter within safe disk
                θ ~ U[0,2π); rj = jitter_frac*clearance
                x = cx + rj*cos(θ); y = cy + rj*sin(θ)
                qpos[jx:jx+3] = [x, y, z]
                quat = RANDOM_QUATERNION_SHOEMAKE()
                qpos[jx+3:jx+7] = quat
                k += 1; if k == len(bodies): break

        qpos_batch[b_idx] = qpos
        qvel_batch[b_idx] = 0
    return qpos_batch, qvel_batch

function RUN_AND_EXPORT(model, scenes, steps, save_root, part_name, count):
    # run each scene; for each scene dump final body poses (7D per body)
    for s_idx, qpos0 in enumerate(scenes):
        reset(model, qpos0)
        step_forward(model, steps)

        npy_dir = f"{save_root}/{part_name}/{count}"
        makedirs(npy_dir, exist_ok=True)
        Q = []   # [(body_name, pos(3), quat(4))]
        for b in target_bodies(part_name):
            addr = b.qpos_addr
            pos = data.qpos[addr:addr+3]
            quat= data.qpos[addr+3:addr+7]  # wxyz in MuJoCo
            Q.append((b.name, pos, quat))
        save_npy(f"{npy_dir}/scene_{s_idx:06d}.npy", Q)

function RESTORE_FROM_NPY(model, npy_path):
    Q = load_npy(npy_path)
    qpos = model.qpos0.copy()
    for (name, pos, quat) in Q:
        b = body_by_name(name)
        addr = b.qpos_addr
        qpos[addr:addr+3] = pos
        qpos[addr+3:addr+7] = quat
    return qpos
```

##### 与常见方法的对比

| 方法                    | 思路                                                          | 优点                                               | 痛点/适用边界                                                                                           |
| --------------------- | ----------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **本方案：六角密堆+分层**       | 以 `geom_rbound` 保守等效圆，六角格 XY + 多层 Z，一次性构造零重叠初态；姿态用 Shoemake | **确定性容量上界、高密度、零重叠、不触发 `mj_forward` 求解**，对大批量实例稳定 | 需要合理 `xy_extent` 与 `layer_gap`；不同尺寸混合时目前按“全局上界”略保守                                                |
| Poisson-disk（Bridson） | 蓝噪声，最小间距约束，栅格加速                                             | 随机性强、分布自然                                        | 容量不易达极限密度；多层控制与不同半径需要额外实现；仍需避免 3D 穿模                                       |
| 贪心随机 + 反复拒绝           | 随机采样，检测碰撞冲突就重采                                              | 实现最简单                                            | 随实例数增长极易退化；高密度时重采次数爆炸                                                                             |
| 随机+物理“沉降”预滚           | 高处丢下，让物理自己推开                                                | 直观                                               | 初态就会爆出巨量接触，Newton/PGS 会拖慢甚至 OOM/栈溢出；且不可控的终态结构（随机堆放不可重复） |
| 全局优化/ILP/packing      | 把布局当优化问题求解                                                  | 理论上更紧                                            | 工程复杂度高、求解时间不可控，不适合批量数据管线                                                                          |

**为什么我们更好**：

- **不让引擎擦屁股**：把“无碰撞”这个约束在**几何构造阶段**解决，而不是扔给接触求解器；这直接回避了接触规模→内存/时间灾难的根因。
- **密度接近上限**：六角密堆逼近平面圆堆积上界（π/√12），在给定 `xy_extent` 下更容易塞满。
- **姿态正确**：SO(3) 的均匀采样（Shoemake）避免欧拉角偏置；与 MuJoCo 的 **wxyz** 约定一致，导出/恢复无坑。
- **工程可控**：不依赖 `mj_forward` 的碰撞松弛，不会因为模型规模变化而“偶发”失控。

##### 未来优化（可选路线）

- **异构尺寸更紧密**：按每件 r\_i 自适应“多半径六角堆”或把大件先布局、小件填空（hex + local Poisson）。
- **倾角约束**：为料箱场景设定 pitch/roll 上限（如 ≤15°），在 Shoemake 后做拒绝采样或投影修正。
- **分层自适应**：依据实时装载率动态计算层数与 `layer_gap`，降低高度与初速度。
- **几何近似改良**：有把握时用稳定的 AABB（可从 `geom_aabb` 读取）替代 rbound 以减少保守度；读不到/不可信再回退 rbound。
- **I/O 管线**：把 `npy` 增加 JSON meta（模型版本、part\_name、count、时间戳），强化可追溯性。
- **渲染核对**：在“仿真终态渲染”旁边始终放一张“`npy` 还原渲染”做 QA。

#### 多进程模拟

```python
# 并行运行多个模拟场景
with mp.Pool(processes=workers) as pool:
    results = pool.starmap(
        simulate_single_scene,
        [(scene_idx, initial_qpos, initial_qvel, steps)
         for scene_idx in range(batch_size)]
    )
```

#### 动态XML修改

```python
# 运行时修改replicate数量
modified_xml_path = XmlProcessor.patch_replicate_count(
    xml_path, replicate_count
)
```

## 快速开始

### 环境要求

- Python 3.8+
- MuJoCo 2.3+
- NumPy, matplotlib, mediapy, tqdm, PyYAML

### 安装依赖

```bash
pip install mujoco numpy matplotlib mediapy tqdm pyyaml
```

### 运行示例

```bash
# 从项目根目录运行
python -m src.main

# 或使用配置文件
python -m src.main --config src/config/config.yaml
```

### 配置说明

主要配置项位于 `src/config/config.yaml`：

- `model.path`: MuJoCo XML模型文件路径
- `simulation.batch`: 批量模拟场景数量
- `multiprocessing.workers`: 并行进程数
- `initialization`: 初始化参数（六角密堆、随机化等）

## 输出结果

- **位姿数据**：`.npy`格式，包含每个物体的最终位置和姿态
- **可视化图像**：可选保存模拟过程截图
- **日志信息**：详细的仿真状态和性能统计

## 性能优化

- 多进程并行：充分利用CPU资源
- 内存管理：及时释放临时文件和数据
- 几何优化：高效的碰撞检测算法
- 进度监控：实时显示仿真进度

## 扩展性

项目采用模块化设计，便于扩展：

- 添加新的初始化算法
- 集成不同的物理引擎
- 支持更多输出格式
- 自定义可视化组件
