"""
论文实验部分定性对比图生成脚本
================================
生成两张图：
  1. qualitative_weather.pdf — 雨天+雾天场景的多模型视觉对比
  2. qualitative_lolv1.pdf   — LOLv1低光照场景的多模型视觉对比

布局（每张图）：
  每行 = 一个场景图片
  每列 = Input | Model1 | Model2 | ... | Ours | GT
  红框 zoom-in 放大关键区域
  左下角标注 PSNR/SSIM

使用方法：
  1. 在 CONFIG 区域配置路径、选图、填入指标
  2. python generate_qualitative.py
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
#  CONFIG — 修改这里的路径和参数
# ============================================================

class PathConfig:
    """路径配置（服务器上的路径）"""
    BASE_DIR = "/home/Bjj/HVI-CIDNet-clean"
    COMPARISON_DIR = "/home/Bjj/comparison_models/comparison_results"
    
    INPUT_DIR = os.path.join(BASE_DIR, "filtered/combined_pedestrian_val_input")
    GT_DIR = os.path.join(BASE_DIR, "filtered/combined_pedestrian_val_gt")
    OURS_DIR = os.path.join(BASE_DIR, "results/fid_best")  # 最优模型输出
    
    # LOLv1 数据集路径（单独配置）
    LOLV1_INPUT_DIR = "/home/Bjj/comparison_models/LOLv1/input"
    LOLV1_GT_DIR = "/home/Bjj/comparison_models/LOLv1/gt"
    LOLV1_OURS_DIR = os.path.join(BASE_DIR, "results/lolv1_best")  # LOLv1 最优输出
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "results/qualitative")


# --- 对比模型列表（列顺序） ---
# 格式: (显示名, 结果子目录名)
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


# --- 图1: 雨天+雾天场景选图 ---
WEATHER_ROWS = [
    {
        "label": "Rain-1",
        "filename": "rain_v1_aachen_000004_000019.png",  # ← 替换为你选定的雨天图1
        "zoom_box": [0.35, 0.25, 0.25, 0.30],  # [x%, y%, w%, h%] 红框位置
    },
    {
        "label": "Rain-2",
        "filename": "rain_v1_aachen_000020_000019.png",  # ← 替换为雨天图2
        "zoom_box": [0.4, 0.3, 0.25, 0.30],
    },
    {
        "label": "Fog-1",
        "filename": "foggy_munster_000080_000019.png",    # ← 替换为雾天图1
        "zoom_box": [0.3, 0.2, 0.3, 0.35],
    },
    {
        "label": "Fog-2",
        "filename": "foggy_munster_000100_000019.png",    # ← 替换为雾天图2
        "zoom_box": [0.35, 0.25, 0.25, 0.30],
    },
]

# --- 图2: LOLv1 低光照场景选图 ---
LOLV1_ROWS = [
    {
        "label": "Low-1",
        "filename": "1.png",    # ← 替换为 LOLv1 eval15 中的图片名
        "zoom_box": [0.3, 0.2, 0.3, 0.35],
    },
    {
        "label": "Low-2",
        "filename": "22.png",   # ← 替换为第二张
        "zoom_box": [0.4, 0.3, 0.25, 0.30],
    },
    {
        "label": "Low-3",
        "filename": "79.png",   # ← 替换为第三张
        "zoom_box": [0.35, 0.25, 0.25, 0.30],
    },
]


# --- 预计算指标（从 measure.py 结果填入） ---
# 格式: METRICS[(filename, 列显示名)] = (PSNR, SSIM)
# 只需填对比模型和 Ours，Input/GT 自动跳过
WEATHER_METRICS = {
    # === Rain-1 ===
    # ("rain_v1_aachen_000004_000019.png", "AirNet"):      (0.00, 0.0000),  # ← 填入真实值
    # ("rain_v1_aachen_000004_000019.png", "Restormer"):   (0.00, 0.0000),
    # ("rain_v1_aachen_000004_000019.png", "PromptIR"):    (0.00, 0.0000),
    # ("rain_v1_aachen_000004_000019.png", "Histoformer"): (0.00, 0.0000),
    # ("rain_v1_aachen_000004_000019.png", "MoCE-IR"):     (0.00, 0.0000),
    # ("rain_v1_aachen_000004_000019.png", "Ours"):        (0.00, 0.0000),
    # ... 其他行同理 ...
}

LOLV1_METRICS = {
    # === Low-1 ===
    # ("1.png", "AirNet"):      (0.00, 0.0000),
    # ("1.png", "Restormer"):   (0.00, 0.0000),
    # ... 其他行同理 ...
}


# --- 视觉参数 ---
CELL_W, CELL_H = 320, 240         # 每个小图尺寸（像素）
ZOOM_SIZE = (110, 100)             # zoom-in 放大图尺寸
ZOOM_BORDER_WIDTH = 3              # 红框宽度
ZOOM_COLOR = (255, 0, 0)           # 红框颜色
METRIC_FONT_SIZE = 14              # 指标字号
DPI = 300                          # 输出分辨率


# ============================================================
#  工具函数
# ============================================================

def load_and_resize(image_path, size):
    """加载并调整图片尺寸，不存在则返回灰色占位图"""
    if not os.path.exists(image_path):
        print(f"  ⚠ 缺失: {os.path.basename(image_path)}")
        return np.ones((size[1], size[0], 3), dtype=np.uint8) * 128
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.LANCZOS)
    return np.array(img)


def add_zoom_patch(image, zoom_box, zoom_size, border_w=3, color=(255, 0, 0)):
    """
    在图片上添加红框标记和 zoom-in 放大插图
    
    参数:
        image: numpy [H, W, 3] uint8
        zoom_box: [x_ratio, y_ratio, w_ratio, h_ratio]
        zoom_size: (w, h) 放大图的像素尺寸
        border_w: 边框宽度
        color: 边框颜色 RGB
    返回:
        带有 zoom-in 的图片 numpy array
    """
    H, W = image.shape[:2]
    xr, yr, wr, hr = zoom_box
    cx, cy, cw, ch = int(xr*W), int(yr*H), int(wr*W), int(hr*H)
    
    # 裁剪并放大
    crop = image[cy:cy+ch, cx:cx+cw].copy()
    crop_pil = Image.fromarray(crop).resize(zoom_size, Image.LANCZOS)
    
    img_pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img_pil)
    
    # 原图上画红框
    draw.rectangle([cx, cy, cx+cw, cy+ch], outline=color, width=border_w)
    
    # zoom-in 图加红色边框
    zoom_img = Image.fromarray(np.array(crop_pil))
    zoom_draw = ImageDraw.Draw(zoom_img)
    zw, zh = zoom_size
    zoom_draw.rectangle([0, 0, zw-1, zh-1], outline=color, width=border_w)
    
    # 粘贴到右下角
    margin = 5
    paste_x = W - zw - margin
    paste_y = H - zh - margin
    img_pil.paste(zoom_img, (paste_x, paste_y))
    
    # 连接线
    draw2 = ImageDraw.Draw(img_pil)
    draw2.line([(cx+cw, cy+ch), (paste_x, paste_y)], fill=color, width=1)
    
    return np.array(img_pil)


def add_metric_text(image, psnr_val, ssim_val, font_size=14):
    """在图片左下角添加 PSNR/SSIM 标注"""
    img_pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img_pil)
    text = f"{psnr_val:.2f}/{ssim_val:.3f}"
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    H, W = image.shape[:2]
    tx, ty = 5, H - th - 6
    
    # 半透明黑色背景
    draw.rectangle([tx-2, ty-2, tx+tw+2, ty+th+2], fill=(0, 0, 0))
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    
    return np.array(img_pil)


def resolve_path(col_type, col_dir, filename, path_cfg, dataset="weather"):
    """
    根据列类型和数据集类型解析图片路径
    
    参数:
        col_type: 列标记
        col_dir: 目录名
        filename: 文件名
        path_cfg: 路径配置对象
        dataset: "weather" 或 "lolv1"
    返回:
        完整文件路径
    """
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


def generate_qualitative_figure(rows, columns, metrics, path_cfg, dataset, output_name):
    """
    生成一张定性对比图
    
    参数:
        rows: 行配置列表（每行一个场景）
        columns: 列配置列表（每列一个模型）
        metrics: 预计算指标字典
        path_cfg: 路径配置
        dataset: "weather" 或 "lolv1"
        output_name: 输出文件名前缀
    """
    n_rows = len(rows)
    n_cols = len(columns)
    
    print(f"\n{'='*60}")
    print(f"生成: {output_name} ({n_rows} 行 × {n_cols} 列)")
    print(f"{'='*60}")
    
    # 计算 figure 尺寸
    fig_w = n_cols * CELL_W / DPI * 3
    fig_h = n_rows * CELL_H / DPI * 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=DPI)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    
    plt.subplots_adjust(wspace=0.01, hspace=0.04)
    
    for ri, row_cfg in enumerate(rows):
        label = row_cfg["label"]
        fname = row_cfg["filename"]
        zbox = row_cfg["zoom_box"]
        
        print(f"\n--- 第 {ri+1} 行: {label} ({fname}) ---")
        
        for ci, (col_label, col_dir) in enumerate(columns):
            ax = axes[ri, ci]
            
            # 解析路径并加载
            img_path = resolve_path(col_dir, col_dir, fname, path_cfg, dataset)
            print(f"  [{col_label}] {os.path.basename(img_path)}", end="")
            img = load_and_resize(img_path, (CELL_W, CELL_H))
            
            # 添加 zoom-in
            img = add_zoom_patch(img, zbox, ZOOM_SIZE, ZOOM_BORDER_WIDTH, ZOOM_COLOR)
            
            # 添加指标标注
            is_model = col_dir not in ("__INPUT__", "__GT__")
            if is_model:
                key = (fname, col_label)
                if key in metrics:
                    p, s = metrics[key]
                    img = add_metric_text(img, p, s, METRIC_FONT_SIZE)
                    print(f" → {p:.2f}/{s:.4f}", end="")
            print()
            
            ax.imshow(img)
            ax.axis('off')
            
            # 列标题（第一行）
            if ri == 0:
                weight = 'bold' if col_label == "Ours" else 'normal'
                color = '#d32f2f' if col_label == "Ours" else 'black'
                ax.set_title(col_label, fontsize=9, fontweight=weight, color=color, pad=3)
            
            # 行标签（第一列）
            if ci == 0:
                ax.set_ylabel(label, fontsize=8, fontweight='bold', rotation=90, labelpad=6)
    
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
    
    # === 文件自检 ===
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
        print(f"\n⚠ {missing} 个文件缺失，将用灰色占位图代替\n")
    else:
        print("  所有文件就绪 ✅\n")
    
    # === 图1: 雨天+雾天 ===
    generate_qualitative_figure(
        rows=WEATHER_ROWS,
        columns=WEATHER_COLUMNS,
        metrics=WEATHER_METRICS,
        path_cfg=cfg,
        dataset="weather",
        output_name="qualitative_weather"
    )
    
    # === 图2: LOLv1 低光照 ===
    generate_qualitative_figure(
        rows=LOLV1_ROWS,
        columns=LOLV1_COLUMNS,
        metrics=LOLV1_METRICS,
        path_cfg=cfg,
        dataset="lolv1",
        output_name="qualitative_lolv1"
    )
    
    print(f"\n{'='*60}")
    print("🎉 全部定性对比图生成完毕！")
    print(f"   输出目录: {cfg.OUTPUT_DIR}")
    print(f"{'='*60}")
