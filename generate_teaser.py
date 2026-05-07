"""
论文 Teaser 视觉对比图生成脚本
===============================
功能：自动拼接 Input / 对比模型输出 / Ours / GT 图像为论文 Fig.1 格式
特点：
  - 支持多行（不同退化场景）× 多列（不同模型）
  - 红色框 zoom-in 局部放大对比
  - 每张小图标注 PSNR / SSIM（从 measure.py 结果中预填，保证与论文一致）
  - 输出高分辨率 PDF + PNG

使用方法：
  1. 先用 measure.py 计算各模型的单张图指标
  2. 把指标填入下方 CONFIG 的 METRICS 字典
  3. python generate_teaser.py
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')  # 无显示器环境也能用
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
# 指标优先从 Config.METRICS 读取；未预填时自动计算


# ============================================================
#  CONFIG — 在这里修改你的路径和参数
# ============================================================

class Config:
    """配置类：集中管理所有路径和参数"""
    
    # --- 基础路径（服务器上的路径，按需修改） ---
    BASE_DIR = "/home/Bjj/HVI-CIDNet-clean"
    COMPARISON_DIR = "/home/Bjj/comparison_models/comparison_results"
    
    # --- 恶劣天气数据集（雨天+雾天）的输入和GT目录 ---
    INPUT_DIR = os.path.join(BASE_DIR, "filtered/combined_pedestrian_val_noll_input")
    GT_DIR = os.path.join(BASE_DIR, "filtered/combined_pedestrian_val_noll_gt")
    OURS_DIR = os.path.join(BASE_DIR, "results/fid_best")
    
    # --- LOLv1 低光照数据集路径 ---
    LOLV1_INPUT_DIR = "/home/Bjj/HVI-CIDNet-clean/datasets/LOLdataset/eval15/low"   # ← 修改为实际路径
    LOLV1_GT_DIR = "/home/Bjj/HVI-CIDNet-clean/datasets/LOLdataset/eval15/high"         # ← 修改为实际路径
    LOLV1_OURS_DIR = os.path.join(BASE_DIR, "results/lolv1_best") # ← 修改为实际路径
    
    # --- 输出文件路径 ---
    OUTPUT_DIR = os.path.join(BASE_DIR, "results/teaser")
    OUTPUT_PNG = "teaser_figure.png"
    OUTPUT_PDF = "teaser_figure.pdf"
    
    # --- 每行配置：(场景标签, 图片文件名, zoom-in区域 [x, y, w, h] 归一化坐标) ---
    # zoom-in 区域坐标：[左上角x比例, 左上角y比例, 宽度比例, 高度比例]
    # 例如 [0.3, 0.2, 0.25, 0.3] 表示从图片 30%处开始, 20%处开始, 裁剪25%宽, 30%高
    # dataset 字段: "weather" = 恶劣天气数据集, "lolv1" = LOLv1 低光照数据集
    ROWS = [
        {
            "label": "Rain",
            "filename": "rain_v6_munster_000041_000019.png",  # 雨天场景示例
            "zoom_box": [0.35, 0.25, 0.25, 0.35],
            "dataset": "weather",
        },
        {
            "label": "Fog",
            "filename": "foggy_munster_000078_000019.png",  # 雾天场景示例
            "zoom_box": [0.4, 0.3, 0.25, 0.3],
            "dataset": "weather",
        },
        {
            "label": "Low-light",
            "filename": "780.png",  # LOLv1 eval15 中的图片 ← 替换为你选的图
            "zoom_box": [0.3, 0.2, 0.3, 0.35],
            "dataset": "lolv1",   # ← 使用 LOLv1 数据集路径
        },
    ]
    
    # --- 要对比的模型（按列顺序） ---
    # 格式：(列标题, 结果目录名或特殊标记)
    # 特殊标记: "__INPUT__" = 输入图, "__GT__" = 真值图, "__OURS__" = 我们的模型
    COLUMNS = [
        ("Input",       "__INPUT__"),
        ("PromptIR",    "promptir_scratch"),
        ("Histoformer", "histoformer_scratch"),
        ("MoCE-IR",     "moceir_scratch"),
        ("Ours",        "__OURS__"),
        ("GT",          "__GT__"),
    ]
    
    # --- 视觉参数 ---
    CELL_SIZE = (384, 256)       # 每个小图的像素尺寸 (宽, 高)
    ZOOM_SIZE = (128, 128)       # zoom-in 放大图的像素尺寸
    ZOOM_POSITION = "bottom-right"  # zoom-in 放大图的位置: bottom-right / bottom-left
    ZOOM_BORDER_WIDTH = 3        # 红框边框宽度
    ZOOM_BORDER_COLOR = (255, 0, 0)  # 红框颜色 RGB
    
    # --- 预计算指标（从 measure.py 的结果中手动填入） ---
    # 格式: METRICS[(行filename, 列模型名)] = (PSNR, SSIM)
    # 只需填写对比模型和 Ours 的指标，Input 和 GT 会自动跳过
    # ⚠ 先在服务器上运行 measure.py 得到每张图的指标，再填到这里
    METRICS = {
        # --- Rain 行 ---
        ("rain_v6_munster_000041_000019.png", "PromptIR"):    (0.00, 0.0000),  # ← 替换为真实值
        ("rain_v6_munster_000041_000019.png", "Histoformer"): (0.00, 0.0000),
        ("rain_v6_munster_000041_000019.png", "MoCE-IR"):     (0.00, 0.0000),
        ("rain_v6_munster_000041_000019.png", "Ours"):        (0.00, 0.0000),
        
        # --- Fog 行 ---
        ("foggy_munster_000078_000019.png", "PromptIR"):    (0.00, 0.0000),
        ("foggy_munster_000078_000019.png", "Histoformer"): (0.00, 0.0000),
        ("foggy_munster_000078_000019.png", "MoCE-IR"):     (0.00, 0.0000),
        ("foggy_munster_000078_000019.png", "Ours"):        (0.00, 0.0000),
        # --- Low-light 行 (LOLv1) ---
        ("1.png", "PromptIR"):    (0.00, 0.0000),  # ← LOLv1 图片指标
        ("1.png", "Histoformer"): (0.00, 0.0000),
        ("1.png", "MoCE-IR"):     (0.00, 0.0000),
        ("1.png", "Ours"):        (0.00, 0.0000),
    }
    
    LABEL_FONT_SIZE = 28         # 列标题字号
    METRIC_FONT_SIZE = 16        # PSNR/SSIM 标注字号
    ROW_LABEL_FONT_SIZE = 22     # 行标签字号
    
    PADDING = 4                  # 图片之间的间距(像素)
    TOP_HEADER_HEIGHT = 40       # 顶部列标题区域高度
    LEFT_LABEL_WIDTH = 80        # 左侧行标签区域宽度
    
    DPI = 300                    # 输出分辨率


def load_and_resize(image_path, target_size):
    """
    加载图片并调整尺寸
    
    参数:
        image_path: 图片文件路径
        target_size: 目标尺寸 (宽, 高)
    返回:
        numpy array [H, W, 3] uint8
    """
    if not os.path.exists(image_path):
        print(f"  ⚠ 文件不存在: {image_path}")
        # 生成灰色占位图
        placeholder = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 128
        return placeholder
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img)




def add_zoom_patch(image, zoom_box, zoom_size, position="bottom-right", 
                   border_width=3, border_color=(255, 0, 0)):
    """
    在图片上添加红框标记和zoom-in放大插图
    
    参数:
        image: numpy array [H, W, 3] uint8
        zoom_box: [x_ratio, y_ratio, w_ratio, h_ratio] 归一化坐标
        zoom_size: (zoom_w, zoom_h) 放大图的像素尺寸
        position: 放大图位置
        border_width: 边框宽度
        border_color: 边框颜色 RGB
    返回:
        image_with_zoom: 带有zoom-in的图片 numpy array
    """
    H, W = image.shape[:2]
    x_ratio, y_ratio, w_ratio, h_ratio = zoom_box
    
    # 计算裁剪区域的像素坐标
    crop_x = int(x_ratio * W)
    crop_y = int(y_ratio * H)
    crop_w = int(w_ratio * W)
    crop_h = int(h_ratio * H)
    
    # 裁剪并放大
    crop = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()
    crop_pil = Image.fromarray(crop).resize(zoom_size, Image.LANCZOS)
    crop_resized = np.array(crop_pil)
    
    # 转成 PIL 绘制
    img_pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img_pil)
    
    # 在原图上画红框（标记裁剪区域）
    draw.rectangle(
        [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h],
        outline=border_color, width=border_width
    )
    
    # 计算 zoom-in 图的放置位置
    margin = 6
    zoom_w, zoom_h = zoom_size
    if position == "bottom-right":
        paste_x = W - zoom_w - margin
        paste_y = H - zoom_h - margin
    elif position == "bottom-left":
        paste_x = margin
        paste_y = H - zoom_h - margin
    elif position == "top-right":
        paste_x = W - zoom_w - margin
        paste_y = margin
    else:
        paste_x = margin
        paste_y = margin
    
    # 给 zoom-in 图加红色边框
    zoom_with_border = Image.fromarray(crop_resized)
    zoom_draw = ImageDraw.Draw(zoom_with_border)
    zoom_draw.rectangle(
        [0, 0, zoom_w - 1, zoom_h - 1],
        outline=border_color, width=border_width
    )
    
    # 粘贴 zoom-in 图到主图
    img_pil.paste(zoom_with_border, (paste_x, paste_y))
    
    # 画一条连接线：从原图红框角到 zoom-in 图角
    draw2 = ImageDraw.Draw(img_pil)
    # 从红框右下角连到 zoom 区域左上角
    draw2.line(
        [(crop_x + crop_w, crop_y + crop_h), (paste_x, paste_y)],
        fill=border_color, width=1
    )
    
    return np.array(img_pil)


def add_metric_label(image, psnr_val, ssim_val, font_size=16):
    """
    在图片右下角添加 PSNR/SSIM 标注（半透明背景）
    
    参数:
        image: numpy array [H, W, 3] uint8
        psnr_val: PSNR 数值
        ssim_val: SSIM 数值
        font_size: 字号
    返回:
        标注后的图片 numpy array
    """
    img_pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img_pil)
    
    text = f"{psnr_val:.2f}/{ssim_val:.4f}"
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # 获取文本尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    H, W = image.shape[:2]
    # 放在左下角（避开 zoom-in 区域）
    text_x = 6
    text_y = H - text_h - 8
    
    # 画半透明黑色背景
    bg_padding = 3
    draw.rectangle(
        [text_x - bg_padding, text_y - bg_padding,
         text_x + text_w + bg_padding, text_y + text_h + bg_padding],
        fill=(0, 0, 0, 180)
    )
    
    # 画白色文字
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    return np.array(img_pil)


def get_image_path(col_type, col_dir, filename, cfg, dataset="weather"):
    """
    根据列类型和数据集类型获取图片的实际路径
    
    参数:
        col_type: 列标记（__INPUT__ / __GT__ / __OURS__ / 模型名）
        col_dir: 对应的目录名
        filename: 图片文件名
        cfg: Config 配置对象
        dataset: 数据集类型 "weather" 或 "lolv1"
    返回:
        图片的完整文件路径
    """
    if dataset == "lolv1":
        # LOLv1 数据集使用单独的路径
        if col_type == "__INPUT__":
            return os.path.join(cfg.LOLV1_INPUT_DIR, filename)
        elif col_type == "__GT__":
            return os.path.join(cfg.LOLV1_GT_DIR, filename)
        elif col_type == "__OURS__":
            return os.path.join(cfg.LOLV1_OURS_DIR, filename)
        else:
            return os.path.join(cfg.COMPARISON_DIR, col_dir, filename)
    else:
        # 恶劣天气数据集（雨天+雾天）
        if col_type == "__INPUT__":
            return os.path.join(cfg.INPUT_DIR, filename)
        elif col_type == "__GT__":
            return os.path.join(cfg.GT_DIR, filename)
        elif col_type == "__OURS__":
            return os.path.join(cfg.OURS_DIR, filename)
        else:
            return os.path.join(cfg.COMPARISON_DIR, col_dir, filename)


def generate_teaser(cfg):
    """
    主函数：生成 Teaser 视觉对比图
    
    参数:
        cfg: Config 配置对象
    """
    print("=" * 60)
    print("论文 Teaser 视觉对比图生成器")
    print("=" * 60)
    
    n_rows = len(cfg.ROWS)
    n_cols = len(cfg.COLUMNS)
    cell_w, cell_h = cfg.CELL_SIZE
    
    # 计算画布尺寸
    canvas_w = cfg.LEFT_LABEL_WIDTH + n_cols * (cell_w + cfg.PADDING) - cfg.PADDING
    canvas_h = cfg.TOP_HEADER_HEIGHT + n_rows * (cell_h + cfg.PADDING) - cfg.PADDING
    
    # 创建白色画布
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(canvas_w / cfg.DPI * 3, canvas_h / cfg.DPI * 3),
        dpi=cfg.DPI
    )
    
    # 确保 axes 是 2D 数组
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    print(f"\n布局: {n_rows} 行 × {n_cols} 列")
    print(f"单元格尺寸: {cell_w}×{cell_h} px")
    
    for row_idx, row_cfg in enumerate(cfg.ROWS):
        scene_label = row_cfg["label"]
        filename = row_cfg["filename"]
        zoom_box = row_cfg["zoom_box"]
        dataset = row_cfg.get("dataset", "weather")  # 读取数据集类型
        
        print(f"\n--- 第 {row_idx+1} 行: {scene_label} ({filename}) [dataset={dataset}] ---")
        
        # 先加载 GT
        gt_path = get_image_path("__GT__", "__GT__", filename, cfg, dataset)
        gt_img = load_and_resize(gt_path, cfg.CELL_SIZE)
        
        for col_idx, (col_label, col_dir) in enumerate(cfg.COLUMNS):
            ax = axes[row_idx, col_idx]
            
            # 获取图片路径（根据行的 dataset 字段路由到正确目录）
            img_path = get_image_path(col_dir, col_dir, filename, cfg, dataset)
            print(f"  [{col_label}] {img_path}")
            
            # 加载图片
            img = load_and_resize(img_path, cfg.CELL_SIZE)
            
            # 获取 PSNR/SSIM（跳过 Input 和 GT）
            # 优先使用 METRICS 预填值，否则自动计算
            show_metrics = col_dir not in ("__INPUT__", "__GT__")
            psnr_val, ssim_val = 0.0, 0.0
            if show_metrics:
                metric_key = (filename, col_label)
                if metric_key in cfg.METRICS and cfg.METRICS[metric_key] != (0.00, 0.0000):
                    # 使用手动预填的指标（来自 measure.py）
                    psnr_val, ssim_val = cfg.METRICS[metric_key]
                    print(f"    PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f} (预填值)")
                else:
                    # 自动计算指标
                    try:
                        psnr_val = calc_psnr(gt_img, img, data_range=255)
                        ssim_val = calc_ssim(gt_img, img, data_range=255, channel_axis=2)
                        print(f"    PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f} (自动计算)")
                    except Exception as e:
                        print(f"    ⚠ 自动计算失败: {e}")
            
            # 添加 zoom-in 放大图
            img = add_zoom_patch(
                img, zoom_box, cfg.ZOOM_SIZE,
                position=cfg.ZOOM_POSITION,
                border_width=cfg.ZOOM_BORDER_WIDTH,
                border_color=cfg.ZOOM_BORDER_COLOR
            )
            
            # 添加 PSNR/SSIM 标注
            if show_metrics:
                img = add_metric_label(img, psnr_val, ssim_val, cfg.METRIC_FONT_SIZE)
            
            # 显示图片
            ax.imshow(img)
            ax.axis('off')
            
            # 第一行加列标题
            if row_idx == 0:
                weight = 'bold' if col_label == "Ours" else 'normal'
                color = '#d32f2f' if col_label == "Ours" else 'black'
                ax.set_title(col_label, fontsize=10, fontweight=weight, color=color, pad=4)
            
            # 第一列加行标签
            if col_idx == 0:
                ax.set_ylabel(scene_label, fontsize=9, fontweight='bold',
                            rotation=90, labelpad=8)
    
    # 保存输出
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    png_path = os.path.join(cfg.OUTPUT_DIR, cfg.OUTPUT_PNG)
    pdf_path = os.path.join(cfg.OUTPUT_DIR, cfg.OUTPUT_PDF)
    
    plt.savefig(png_path, dpi=cfg.DPI, bbox_inches='tight', pad_inches=0.05,
                facecolor='white', edgecolor='none')
    plt.savefig(pdf_path, dpi=cfg.DPI, bbox_inches='tight', pad_inches=0.05,
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n{'=' * 60}")
    print(f"✅ Teaser 图已生成:")
    print(f"   PNG: {png_path}")
    print(f"   PDF: {pdf_path}")
    print(f"{'=' * 60}")


def generate_error_map(img, gt, colormap='hot'):
    """
    生成误差热力图（|Model - GT| 的彩色可视化）
    
    用于在 motivation figure 中直观展示模型输出与 GT 之间的差异。
    红色/亮色区域 = 误差大（细节丢失/色偏严重）；蓝色/暗色 = 误差小。
    
    参数:
        img: numpy array [H, W, 3] uint8 —— 模型输出图片
        gt:  numpy array [H, W, 3] uint8 —— Ground Truth 图片
        colormap: matplotlib colormap 名称（默认 'hot'）
    返回:
        error_map: numpy array [H, W, 3] uint8 —— 伪彩色误差图
    """
    # 计算绝对误差（灰度平均）
    diff = np.abs(img.astype(float) - gt.astype(float)).mean(axis=2)
    
    # 归一化到 [0, 1]
    diff_max = diff.max() if diff.max() > 0 else 1.0
    diff_norm = diff / diff_max
    
    # 应用 colormap
    cmap = plt.cm.get_cmap(colormap)
    error_colored = (cmap(diff_norm)[:, :, :3] * 255).astype(np.uint8)
    
    return error_colored


def export_teaser_materials(cfg):
    """
    导出 Motivation Figure 的底图素材
    
    为 PPT/Illustrator 组装三段式概念图提供高质量素材：
    1. 每个模型的独立输出图片（原尺寸 + 带 zoom-in 标注）
    2. zoom-in 区域的单独放大裁剪图（可在 PPT 中灵活放置）
    3. 误差热力图（|Model - GT|，用于可视化细节丢失/色偏程度）
    
    参数:
        cfg: Config 配置对象
    """
    print("=" * 60)
    print("Motivation Figure 素材导出器")
    print("=" * 60)
    
    # 创建素材输出目录
    mat_dir = os.path.join(cfg.OUTPUT_DIR, "materials")
    os.makedirs(mat_dir, exist_ok=True)
    
    for row_idx, row_cfg in enumerate(cfg.ROWS):
        scene_label = row_cfg["label"]
        filename = row_cfg["filename"]
        zoom_box = row_cfg["zoom_box"]
        dataset = row_cfg.get("dataset", "weather")
        
        print(f"\n--- 场景: {scene_label} ({filename}) ---")
        
        # 为每个场景创建子目录
        scene_dir = os.path.join(mat_dir, f"{row_idx+1}_{scene_label}")
        os.makedirs(scene_dir, exist_ok=True)
        
        # 先加载 GT（用于差异图计算）
        gt_path = get_image_path("__GT__", "__GT__", filename, cfg, dataset)
        gt_img = load_and_resize(gt_path, cfg.CELL_SIZE)
        
        for col_label, col_dir in cfg.COLUMNS:
            # 获取图片路径
            img_path = get_image_path(col_dir, col_dir, filename, cfg, dataset)
            img = load_and_resize(img_path, cfg.CELL_SIZE)
            
            # 安全的文件名（去掉特殊字符）
            safe_name = col_label.replace(" ", "_")
            
            # ---- 计算或读取 PSNR/SSIM（跳过 Input 和 GT） ----
            show_metrics = col_dir not in ("__INPUT__", "__GT__")
            psnr_val, ssim_val = 0.0, 0.0
            if show_metrics:
                metric_key = (filename, col_label)
                if metric_key in cfg.METRICS and cfg.METRICS[metric_key] != (0.00, 0.0000):
                    psnr_val, ssim_val = cfg.METRICS[metric_key]
                else:
                    try:
                        psnr_val = calc_psnr(gt_img, img, data_range=255)
                        ssim_val = calc_ssim(gt_img, img, data_range=255, channel_axis=2)
                    except Exception as e:
                        print(f"    ⚠ 指标计算失败: {e}")
            
            # ---- 1. 保存带 PSNR/SSIM 标注的图片 ----
            if show_metrics:
                img_labeled = add_metric_label(img, psnr_val, ssim_val, cfg.METRIC_FONT_SIZE)
            else:
                img_labeled = img
            out_clean = os.path.join(scene_dir, f"{safe_name}_clean.png")
            Image.fromarray(img_labeled).save(out_clean, quality=95)
            
            # ---- 2. 保存带 zoom-in 标注的版本 ----
            img_with_zoom = add_zoom_patch(
                img, zoom_box, cfg.ZOOM_SIZE,
                position=cfg.ZOOM_POSITION,
                border_width=cfg.ZOOM_BORDER_WIDTH,
                border_color=cfg.ZOOM_BORDER_COLOR
            )
            out_zoom = os.path.join(scene_dir, f"{safe_name}_with_zoom.png")
            Image.fromarray(img_with_zoom).save(out_zoom, quality=95)
            
            # ---- 3. 导出单独的 zoom-in 裁剪图 ----
            H, W = img.shape[:2]
            x_r, y_r, w_r, h_r = zoom_box
            cx, cy = int(x_r * W), int(y_r * H)
            cw, ch = int(w_r * W), int(h_r * H)
            crop = img[cy:cy+ch, cx:cx+cw]
            # 放大到更大尺寸（便于 PPT 中使用）
            crop_large = np.array(Image.fromarray(crop).resize((256, 256), Image.LANCZOS))
            out_crop = os.path.join(scene_dir, f"{safe_name}_crop_256.png")
            Image.fromarray(crop_large).save(out_crop, quality=95)
            
            # ---- 4. 生成误差热力图（跳过 Input 和 GT） ----
            if col_dir not in ("__INPUT__", "__GT__"):
                error_map = generate_error_map(img, gt_img, colormap='hot')
                out_err = os.path.join(scene_dir, f"{safe_name}_error_map.png")
                Image.fromarray(error_map).save(out_err, quality=95)
                
                # 误差 zoom-in 裁剪
                err_crop = error_map[cy:cy+ch, cx:cx+cw]
                err_crop_large = np.array(Image.fromarray(err_crop).resize((256, 256), Image.LANCZOS))
                out_err_crop = os.path.join(scene_dir, f"{safe_name}_error_crop_256.png")
                Image.fromarray(err_crop_large).save(out_err_crop, quality=95)
                
                print(f"  ✅ {col_label}: clean + zoom + crop + error_map")
            else:
                print(f"  ✅ {col_label}: clean + zoom + crop")
    
    print(f"\n{'=' * 60}")
    print(f"✅ 全部素材已导出到: {mat_dir}")
    print(f"{'=' * 60}")
    print(f"\n📋 素材目录结构:")
    print(f"   {mat_dir}/")
    for row_idx, row_cfg in enumerate(cfg.ROWS):
        scene_label = row_cfg["label"]
        print(f"   ├── {row_idx+1}_{scene_label}/")
        for col_label, _ in cfg.COLUMNS:
            safe = col_label.replace(" ", "_")
            print(f"   │   ├── {safe}_clean.png       (无标注原图)")
            print(f"   │   ├── {safe}_with_zoom.png   (带红框和zoom-in)")
            print(f"   │   ├── {safe}_crop_256.png    (zoom区域放大裁剪)")
            if col_label not in ("Input", "GT"):
                print(f"   │   ├── {safe}_error_map.png   (全图误差热力图)")
                print(f"   │   └── {safe}_error_crop_256.png (zoom区域误差)")


if __name__ == '__main__':
    config = Config()
    
    # ====== 模式选择 ======
    # 命令行参数: --materials 仅导出素材, 不加参数则生成完整 teaser
    export_mode = '--materials' in sys.argv
    
    # ====== 快速自检：列出所有需要的文件 ======
    print("\n🔍 文件自检:")
    missing = 0
    for row_cfg in config.ROWS:
        fn = row_cfg["filename"]
        ds = row_cfg.get("dataset", "weather")
        for col_label, col_dir in config.COLUMNS:
            path = get_image_path(col_dir, col_dir, fn, config, ds)
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            if not exists:
                missing += 1
                print(f"  {status} [{col_label}] {path}")
    
    if missing > 0:
        print(f"\n⚠ 共有 {missing} 个文件缺失，缺失位置会用灰色占位图代替。")
        print("  请检查 Config 中的路径配置是否正确。")
    else:
        print("  所有文件就绪! ✅")
    
    print()
    
    if export_mode:
        # 导出素材模式（用于 PPT/AI 组装 motivation figure）
        export_teaser_materials(config)
    else:
        # 传统模式（生成完整拼接 teaser 图）
        generate_teaser(config)
