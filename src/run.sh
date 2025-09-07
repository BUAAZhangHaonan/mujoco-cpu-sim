#!/bin/bash

# =================================================================
# 这是一个运行仿真并生成NPY数据的脚本。
# 请根据需要修改下面的配置变量。
# =================================================================

# --- 1. 配置路径 ---
# 定义包含MuJoCo场景文件（.xml）的目录
SCENE_DIR="assets_0907_0.001/mjcf/scenes"

# 定义用于存储输出NPY文件的目录
NPY_OUT_DIR="output/npy/npy_0907_0.001"



mkdir -p "$NPY_OUT_DIR"

echo "开始运行仿真..."
echo "场景文件目录: $SCENE_DIR"
echo "NPY输出目录: $NPY_OUT_DIR"


python3 0821/run.py \
    --scenes-dir "$SCENE_DIR" \
    --npy-out-dir "$NPY_OUT_DIR" \
    --batch 10 \
    --workers 20 \
    --replicate-counts 25 50 100 \
    --replicate-weights 0.25 0.5 0.25 \
    --allow-3d-rot \
    --disable-stability-check \
    --z-low 0.65 \
    --xy-extent -0.1 0.1


# --- 3. 完成 ---
echo "脚本执行完毕！"
echo "输出数据已保存至: $NPY_OUT_DIR"
