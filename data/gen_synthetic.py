"""
合成退化数据集生成脚本
用途：为 DA-MambaNet 生成高斯噪声和 JPEG 压缩伪影的配对训练集

使用方法：
    # 生成高斯去噪数据集（从 BSD400 干净图像）
    python data/gen_synthetic.py --task noise --clean_dir ./datasets/BSD400_clean \
        --output_dir ./datasets/Noise_train --sigma 15 25 50

    # 生成 JPEG 压缩伪影数据集（从 DIV2K 干净图像）
    python data/gen_synthetic.py --task jpeg --clean_dir ./datasets/DIV2K_clean \
        --output_dir ./datasets/JPEG_train --quality 10 20 30 40

输出目录结构（与 AllInOneDataset 兼容）：
    <output_dir>/
    ├── low/     ← 退化图像（含噪声 or 含压缩伪影）
    └── high/    ← 干净 GT 图像（原图复制）

作者：DA-MambaNet 项目
"""

import os
import argparse
import random
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


# 支持的图像格式
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMG_EXTS


def collect_images(dir_path: str) -> list:
    """递归收集目录下所有图像文件路径"""
    files = []
    for root, _, fnames in os.walk(dir_path):
        for fname in sorted(fnames):
            if is_image(fname):
                files.append(os.path.join(root, fname))
    return files


def add_gaussian_noise(img_array: np.ndarray, sigma: float) -> np.ndarray:
    """
    给图像添加高斯白噪声（AWGN）
    
    参数：
        img_array: float32 数组，范围 [0, 255]
        sigma:     噪声标准差（越大噪声越强，常用值 15/25/50）
    返回：
        含噪图像 uint8 数组
    """
    noise = np.random.normal(0, sigma, img_array.shape).astype(np.float32)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return noisy


def gen_noise_dataset(clean_dir: str, output_dir: str,
                      sigmas: list, crop_size: int = 256,
                      patches_per_img: int = 5):
    """
    生成高斯去噪数据集
    
    策略：
    - 对每张干净图像随机裁剪 patches_per_img 个 patch
    - 对每个 patch 随机选择一个 sigma 加噪
    - 保存 (noisy, clean) 配对
    
    参数：
        clean_dir:       干净图像目录（如 BSD400_clean/）
        output_dir:      输出目录（将创建 low/ 和 high/ 子目录）
        sigmas:          噪声级别列表（如 [15, 25, 50]）
        crop_size:       裁剪尺寸（256×256）
        patches_per_img: 每张图裁剪多少个 patch
    """
    low_dir  = os.path.join(output_dir, 'low')
    high_dir = os.path.join(output_dir, 'high')
    os.makedirs(low_dir,  exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)

    clean_files = collect_images(clean_dir)
    print(f"[去噪] 找到 {len(clean_files)} 张干净图像，sigma={sigmas}")
    print(f"[去噪] 每张裁剪 {patches_per_img} 个 patch，共 {len(clean_files)*patches_per_img} 对")

    pair_idx = 0
    for img_path in tqdm(clean_files, desc='生成噪声数据'):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # 如果图像小于 crop_size，则跳过
        if w < crop_size or h < crop_size:
            continue

        for _ in range(patches_per_img):
            # 随机裁剪
            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)
            clean_patch = img.crop((x, y, x + crop_size, y + crop_size))
            clean_arr = np.array(clean_patch).astype(np.float32)

            # 随机选择 sigma
            sigma = random.choice(sigmas)
            noisy_arr = add_gaussian_noise(clean_arr, sigma)
            noisy_patch = Image.fromarray(noisy_arr)

            # 文件名：idx_sigma.png
            fname = f'{pair_idx:06d}_s{sigma}.png'
            clean_patch.save(os.path.join(high_dir, fname))
            noisy_patch.save(os.path.join(low_dir,  fname))
            pair_idx += 1

    print(f"[去噪] 完成！共 {pair_idx} 对保存至 {output_dir}")


def gen_jpeg_dataset(clean_dir: str, output_dir: str,
                     qualities: list, crop_size: int = 256,
                     patches_per_img: int = 5):
    """
    生成 JPEG 压缩伪影去除数据集
    
    策略：
    - 对每张干净图像随机裁剪 patch
    - 以不同质量因子（QF）进行 JPEG 压缩得到退化图
    - 保存 (compressed, clean) 配对
    
    参数：
        clean_dir:   干净图像目录（如 DIV2K_clean/）
        output_dir:  输出目录
        qualities:   JPEG 质量因子列表（10=重度压缩，40=轻度压缩）
        crop_size:   裁剪尺寸
        patches_per_img: 每张图的 patch 数
    """
    import io

    low_dir  = os.path.join(output_dir, 'low')
    high_dir = os.path.join(output_dir, 'high')
    os.makedirs(low_dir,  exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)

    clean_files = collect_images(clean_dir)
    print(f"[JPEG] 找到 {len(clean_files)} 张干净图像，质量因子={qualities}")
    print(f"[JPEG] 每张裁剪 {patches_per_img} 个 patch，共 {len(clean_files)*patches_per_img} 对")

    pair_idx = 0
    for img_path in tqdm(clean_files, desc='生成JPEG数据'):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        if w < crop_size or h < crop_size:
            continue

        for _ in range(patches_per_img):
            # 随机裁剪
            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)
            clean_patch = img.crop((x, y, x + crop_size, y + crop_size))

            # 随机质量因子
            qf = random.choice(qualities)

            # JPEG 编解码（内存中完成，避免中间文件）
            buf = io.BytesIO()
            clean_patch.save(buf, format='JPEG', quality=qf)
            buf.seek(0)
            compressed_patch = Image.open(buf).copy()   # .copy() 防止 buf 关闭后失效

            # 文件名：idx_qf.png（高质量保存避免二次压缩）
            fname = f'{pair_idx:06d}_q{qf}.png'
            clean_patch.save(os.path.join(high_dir, fname))
            compressed_patch.save(os.path.join(low_dir, fname))
            pair_idx += 1

    print(f"[JPEG] 完成！共 {pair_idx} 对保存至 {output_dir}")


# ==============================================================================
# 雪天数据集格式转换（CSD → low/high 格式）
# ==============================================================================
def convert_csd_format(csd_root: str, output_dir: str, split: str = 'train'):
    """
    将 CSD 数据集原始格式转换为 AllInOneDataset 的 low/high 格式
    
    CSD 原始结构：
        CSD/train/Snow/  ← 含雪图像
        CSD/train/Gt/    ← 清洁图像
    
    转换后：
        output_dir/low/   ← 含雪图像（复制）
        output_dir/high/  ← 清洁图像（复制）
    """
    import shutil

    snow_dir = os.path.join(csd_root, split, 'Snow')
    gt_dir   = os.path.join(csd_root, split, 'Gt')

    low_dir  = os.path.join(output_dir, 'low')
    high_dir = os.path.join(output_dir, 'high')
    os.makedirs(low_dir,  exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)

    snow_files = sorted([f for f in os.listdir(snow_dir) if is_image(f)])
    gt_files   = sorted([f for f in os.listdir(gt_dir)   if is_image(f)])

    print(f"[CSD 转换] Snow: {len(snow_files)} | GT: {len(gt_files)}")
    for sf, gf in tqdm(zip(snow_files, gt_files), desc='转换CSD格式'):
        shutil.copy2(os.path.join(snow_dir, sf), os.path.join(low_dir,  sf))
        shutil.copy2(os.path.join(gt_dir,   gf), os.path.join(high_dir, gf))

    print(f"[CSD 转换] 完成！输出: {output_dir}")


# ==============================================================================
# 命令行入口
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description='合成退化数据集生成工具（DA-MambaNet 专用）')
    p.add_argument('--task', choices=['noise', 'jpeg', 'convert_csd'], required=True,
                   help='任务类型: noise=高斯噪声 | jpeg=JPEG压缩 | convert_csd=转换CSD格式')
    p.add_argument('--clean_dir',  type=str, default='',
                   help='干净图像源目录（noise/jpeg 任务必填）')
    p.add_argument('--csd_root',   type=str, default='',
                   help='CSD 数据集根目录（convert_csd 任务必填）')
    p.add_argument('--output_dir', type=str, required=True,
                   help='输出目录（将创建 low/ 和 high/ 子目录）')
    p.add_argument('--split',      type=str, default='train',
                   help='CSD 数据集分割（train/test）')
    p.add_argument('--sigma',      type=int, nargs='+', default=[15, 25, 50],
                   help='高斯噪声标准差列表（noise 任务）')
    p.add_argument('--quality',    type=int, nargs='+', default=[10, 20, 30, 40],
                   help='JPEG 质量因子列表（jpeg 任务）')
    p.add_argument('--crop_size',  type=int, default=256,
                   help='随机裁剪大小')
    p.add_argument('--patches',    type=int, default=5,
                   help='每张图生成的 patch 数量')
    p.add_argument('--seed',       type=int, default=42,
                   help='随机种子（保证可复现）')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.task == 'noise':
        assert args.clean_dir, '--clean_dir 不能为空'
        gen_noise_dataset(
            clean_dir      = args.clean_dir,
            output_dir     = args.output_dir,
            sigmas         = args.sigma,
            crop_size      = args.crop_size,
            patches_per_img= args.patches,
        )
    elif args.task == 'jpeg':
        assert args.clean_dir, '--clean_dir 不能为空'
        gen_jpeg_dataset(
            clean_dir      = args.clean_dir,
            output_dir     = args.output_dir,
            qualities      = args.quality,
            crop_size      = args.crop_size,
            patches_per_img= args.patches,
        )
    elif args.task == 'convert_csd':
        assert args.csd_root, '--csd_root 不能为空'
        convert_csd_format(
            csd_root   = args.csd_root,
            output_dir = args.output_dir,
            split      = args.split,
        )
