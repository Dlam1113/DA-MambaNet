"""
论文实验部分定性对比图生成脚本（纯视觉对比版）
================================================
布局：上方列标题 + 下方图片网格
标注：红框 + 黄框标记两个关键区域（不显示指标）
输出：qualitative_weather.pdf / qualitative_lolv1.pdf

使用方法：
  1. 配置路径、选图、标注区域
  2. python generate_qualitative.py
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
#  CONFIG
# ============================================================

class PathConfig:
    """路径配置"""
    BASE_DIR = "/home/Bjj/HVI-CIDNet-clean"
    COMPARISON_DIR = "/home/Bjj/comparison_models/comparison_results"

    # 恶劣天气数据集
    INPUT_DIR = os.path.join(BASE_DIR, "filtered/combined_pedestrian_val_noll_input")
    GT_DIR = os.path.join(BASE_DIR, "filtered/combined_pedestrian_val_noll_gt")
    OURS_DIR = os.path.join(BASE_DIR, "results/fid_best")

    # LOLv1 数据集
    LOLV1_INPUT_DIR = "/home/Bjj/HVI-CIDNet-clean/datasets/LOLdataset/eval15/low"
    LOLV1_GT_DIR = "/home/Bjj/HVI-CIDNet-clean/datasets/LOLdataset/eval15/high"
    LOLV1_OURS_DIR = os.path.join(BASE_DIR, "results/lolv1_best")

    OUTPUT_DIR = os.path.join(BASE_DIR, "results/qualitative")


# --- 对比模型列表 ---
WEATHER_COLUMNS = [
    ("Input",           "__INPUT__"),
    ("AirNet",          "airnet_scratch"),
    ("Restormer",       "restormer_scratch"),
    ("PromptIR",        "promptir_scratch"),
    ("Histoformer",     "histoformer_scratch"),
    ("MoCE-IR",         "moceir_scratch"),
    ("Ours",            "__OURS__"),
    ("GT",              "__GT__"),
]

LOLV1_COLUMNS = [
    ("Input",           "__INPUT__"),
    ("AirNet",          "airnet_scratch"),
    ("Restormer",       "restormer_scratch"),
    ("PromptIR",        "promptir_scratch"),
    ("Histoformer",     "histoformer_scratch"),
    ("MoCE-IR",         "moceir_scratch"),
    ("Ours",            "__OURS__"),
    ("GT",              "__GT__"),
]


# --- 图1: 雨天+雾天场景 ---
# 每行支持两个标注框：红框(box_red) + 黄框(box_yellow)
# 格式: [x%, y%, w%, h%] 归一化坐标
WEATHER_ROWS = [
    {
        "label": "Rain-1",
        "filename": "rain_v6_munster_000049_000019.png",
        "box_red":    [0.05, 0.10, 0.25, 0.35],   # ← 调整红框位置
        "box_yellow": [0.55, 0.30, 0.25, 0.35],   # ← 调整黄框位置
    },
    {
        "label": "Rain-2",
        "filename": "rain_v5_munster_000044_000019.png",
        "box_red":    [0.30, 0.15, 0.25, 0.35],
        "box_yellow": [0.60, 0.40, 0.25, 0.30],
    },
    {
        "label": "Fog-1",
        "filename": "foggy_frankfurt_000001_012519.png",
        "box_red":    [0.10, 0.20, 0.25, 0.30],
        "box_yellow": [0.50, 0.25, 0.25, 0.35],
    },
    {
        "label": "Fog-2",
        "filename": "foggy_munster_000145_000019.png",
        "box_red":    [0.35, 0.15, 0.25, 0.30],
        "box_yellow": [0.60, 0.35, 0.25, 0.30],
    },
]

# --- 图2: LOLv1 低光照 ---
LOLV1_ROWS = [
    {
        "label": "Low-1",
        "filename": "748.png",
        "box_red":    [0.10, 0.15, 0.30, 0.35],
        "box_yellow": [0.55, 0.30, 0.25, 0.35],
    },
    {
        "label": "Low-2",
        "filename": "22.png",
        "box_red":    [0.20, 0.10, 0.25, 0.30],
        "box_yellow": [0.50, 0.40, 0.30, 0.30],
    },
    {
        "label": "Low-3",
        "filename": "493.png",
        "box_red":    [0.05, 0.20, 0.25, 0.35],
        "box_yellow": [0.55, 0.25, 0.25, 0.35],
    },
]


# --- 视觉参数 ---
CELL_W, CELL_H = 320, 240    # 每个小图尺寸
BOX_WIDTH = 3                 # 标注框线宽
RED_COLOR = (255, 0, 0)       # 红框颜色
YELLOW_COLOR = (255, 220, 0)  # 黄框颜色
DPI = 300


# ============================================================
#  工具函数
# ============================================================

def load_and_resize(image_path, size):
    """加载并调整图片尺寸"""
    if not os.path.exists(image_path):
        print(f"  ⚠ 缺失: {os.path.basename(image_path)}")
        return np.ones((size[1], size[0], 3), dtype=np.uint8) * 128
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.LANCZOS)
    return np.array(img)


def draw_boxes(image, box_red, box_yellow, line_w=3):
    """
    在图片上绘制红框和黄框标注区域（仅画框，不放大）
    
    参数:
        image: numpy [H, W, 3]
        box_red: [x%, y%, w%, h%] 红框区域
        box_yellow: [x%, y%, w%, h%] 黄框区域
        line_w: 线宽
    返回:
        标注后的图片 numpy array
    """
    H, W = image.shape[:2]
    img_pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img_pil)
    
    # 画红框
    if box_red:
        xr, yr, wr, hr = box_red
        x1, y1 = int(xr * W), int(yr * H)
        x2, y2 = int((xr + wr) * W), int((yr + hr) * H)
        draw.rectangle([x1, y1, x2, y2], outline=RED_COLOR, width=line_w)
    
    # 画黄框
    if box_yellow:
        xr, yr, wr, hr = box_yellow
        x1, y1 = int(xr * W), int(yr * H)
        x2, y2 = int((xr + wr) * W), int((yr + hr) * H)
        draw.rectangle([x1, y1, x2, y2], outline=YELLOW_COLOR, width=line_w)
    
    return np.array(img_pil)


def resolve_path(col_type, col_dir, filename, path_cfg, dataset="weather"):
    """根据列类型和数据集解析图片路径"""
    if dataset == "lolv1":
        if col_type == "__INPUT__":
            return os.path.join(path_cfg.LOLV1_INPUT_DIR, filename)
        elif col_type == "__GT__":
            return os.path.join(path_cfg.LOLV1_GT_DIR, filename)
        elif col_type == "__OURS__":
            return os.path.join(path_cfg.LOLV1_OURS_DIR, filename)
        else:
            return os.path.join(path_cfg.COMPARISON_DIR, col_dir, filename)
    else:
        if col_type == "__INPUT__":
            return os.path.join(path_cfg.INPUT_DIR, filename)
        elif col_type == "__GT__":
            return os.path.join(path_cfg.GT_DIR, filename)
        elif col_type == "__OURS__":
            return os.path.join(path_cfg.OURS_DIR, filename)
        else:
            return os.path.join(path_cfg.COMPARISON_DIR, col_dir, filename)


def generate_qualitative_figure(rows, columns, path_cfg, dataset, output_name):
    """
    生成一张纯视觉定性对比图（无指标标注）
    
    参数:
        rows: 行配置列表
        columns: 列配置列表
        path_cfg: 路径配置
        dataset: "weather" 或 "lolv1"
        output_name: 输出文件名前缀
    """
    n_rows = len(rows)
    n_cols = len(columns)

    print(f"\n{'='*60}")
    print(f"生成: {output_name} ({n_rows} 行 × {n_cols} 列)")
    print(f"{'='*60}")

    fig_w = n_cols * CELL_W / DPI * 3
    fig_h = n_rows * CELL_H / DPI * 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=DPI)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    plt.subplots_adjust(wspace=0.01, hspace=0.04)

    for ri, row_cfg in enumerate(rows):
        fname = row_cfg["filename"]
        box_r = row_cfg.get("box_red")
        box_y = row_cfg.get("box_yellow")

        print(f"\n--- 第 {ri+1} 行: {row_cfg['label']} ({fname}) ---")

        for ci, (col_label, col_dir) in enumerate(columns):
            ax = axes[ri, ci]

            # 解析路径并加载
            img_path = resolve_path(col_dir, col_dir, fname, path_cfg, dataset)
            print(f"  [{col_label}] {os.path.basename(img_path)}")
            img = load_and_resize(img_path, (CELL_W, CELL_H))

            # 画红框 + 黄框
            img = draw_boxes(img, box_r, box_y, BOX_WIDTH)

            ax.imshow(img)
            ax.axis('off')

            # 列标题（仅第一行显示）
            if ri == 0:
                weight = 'bold' if col_label == "Ours" else 'normal'
                color = '#d32f2f' if col_label == "Ours" else 'black'
                ax.set_title(col_label, fontsize=9, fontweight=weight, color=color, pad=3)

    # 保存
    os.makedirs(path_cfg.OUTPUT_DIR, exist_ok=True)
    for ext in ['png', 'pdf']:
        out_path = os.path.join(path_cfg.OUTPUT_DIR, f"{output_name}.{ext}")
        fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0.03,
                    facecolor='white', edgecolor='none')
        print(f"  ✅ 已保存: {out_path}")
    plt.close(fig)


# ============================================================
#  主函数
# ============================================================

if __name__ == '__main__':
    cfg = PathConfig()

    # 文件自检
    print("🔍 文件自检...")
    missing = 0
    for rows, cols, ds in [(WEATHER_ROWS, WEATHER_COLUMNS, "weather"),
                            (LOLV1_ROWS, LOLV1_COLUMNS, "lolv1")]:
        for row in rows:
            for col_label, col_dir in cols:
                p = resolve_path(col_dir, col_dir, row["filename"], cfg, ds)
                if not os.path.exists(p):
                    missing += 1
                    print(f"  ❌ [{ds}/{col_label}] {p}")

    if missing:
        print(f"\n⚠ {missing} 个文件缺失\n")
    else:
        print("  所有文件就绪 ✅\n")

    # 图1: 雨天+雾天
    generate_qualitative_figure(
        rows=WEATHER_ROWS,
        columns=WEATHER_COLUMNS,
        path_cfg=cfg,
        dataset="weather",
        output_name="qualitative_weather"
    )

    # 图2: LOLv1
    generate_qualitative_figure(
        rows=LOLV1_ROWS,
        columns=LOLV1_COLUMNS,
        path_cfg=cfg,
        dataset="lolv1",
        output_name="qualitative_lolv1"
    )

    print(f"\n{'='*60}")
    print("🎉 全部定性对比图生成完毕！")
    print(f"   输出目录: {cfg.OUTPUT_DIR}")
    print(f"{'='*60}")
