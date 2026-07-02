"""
CSD 数据集重组采样脚本
====================
将 CSD (Comprehensive Snow Dataset) 原始结构重组为
AllInOneDataset 要求的 low/high 格式，并采样指定数量。

CSD 原始结构：
    CSD_raw/
    ├── Train/
    │   ├── Snow/   ← 含雪图像（退化输入）
    │   ├── Gt/     ← 清洁GT图像
    │   └── Mask/   ← 雪花掩码（本脚本忽略）
    └── Test/
        ├── Snow/
        ├── Gt/
        └── Mask/

重组后结构（与 AllInOneDataset 兼容）：
    Snow_train/
    ├── low/    ← 含雪图像（采样 n_train 张）
    └── high/   ← 清洁图像
    Snow_val/
    ├── low/    ← 含雪图像（采样 n_val 张）
    └── high/

注意：
    - Snow/ 和 Gt/ 中文件名相同，一一对应，直接按排序后的索引采样
    - Mask/ 目录不用于本项目，直接忽略

使用方法（服务器端执行）：
    cd ~/DA_Mamba
    python data/reorganize_csd.py \
        --csd_root  ./datasets/CSD_raw \
        --out_train ./datasets/Snow_train \
        --out_val   ./datasets/Snow_val \
        --n_train 600 \
        --n_val   100 \
        --seed 42

作者：DA-MambaNet 项目
"""

import os
import random
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


# 支持的图像格式（包含 .tif，CSD 数据集使用此格式）
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def is_image(fname: str) -> bool:
    return Path(fname).suffix.lower() in IMG_EXTS


def collect_csd_pairs(csd_root: str, split: str) -> list:
    """
    收集 CSD 数据集某个 split 下的所有 (snow_path, gt_path) 配对。

    配对方式：Snow/ 和 Gt/ 按文件名排序后一一对应（文件名相同）。

    参数：
        csd_root: CSD 根目录（含 Train/ 和 Test/ 子目录）
        split:    'Train' 或 'Test'（大写）
    返回：
        pairs: [(snow_path, gt_path), ...]
    """
    # 兼容大小写
    split_dir = None
    for candidate in [split, split.lower(), split.capitalize()]:
        d = os.path.join(csd_root, candidate)
        if os.path.isdir(d):
            split_dir = d
            break
    if split_dir is None:
        raise FileNotFoundError(f"找不到目录: {csd_root}/{split}，请检查路径")

    snow_dir = os.path.join(split_dir, 'Snow')
    gt_dir   = os.path.join(split_dir, 'Gt')

    if not os.path.isdir(snow_dir):
        raise FileNotFoundError(f"找不到 Snow 目录: {snow_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"找不到 Gt 目录: {gt_dir}")

    snow_files = sorted([f for f in os.listdir(snow_dir) if is_image(f)])
    gt_files   = sorted([f for f in os.listdir(gt_dir)   if is_image(f)])

    # 数量对齐
    if len(snow_files) != len(gt_files):
        print(f"  [警告] Snow({len(snow_files)}) ≠ Gt({len(gt_files)})，取最小值")
        n = min(len(snow_files), len(gt_files))
        snow_files = snow_files[:n]
        gt_files   = gt_files[:n]

    pairs = [
        (os.path.join(snow_dir, sf), os.path.join(gt_dir, gf))
        for sf, gf in zip(snow_files, gt_files)
    ]

    print(f"  [CSD {split}] 共收集到 {len(pairs)} 对")
    return pairs


def copy_pairs(pairs: list, out_dir: str, n: int, seed: int = 42):
    """
    从 pairs 中随机采样 n 对，复制到 out_dir/low/ 和 out_dir/high/

    参数：
        pairs:   [(snow_path, gt_path), ...]
        out_dir: 输出目录
        n:       采样数量
        seed:    随机种子
    """
    random.seed(seed)

    if len(pairs) < n:
        print(f"  [警告] 可用对数 {len(pairs)} < 目标 {n}，全部取用")
        selected = pairs
    else:
        selected = random.sample(pairs, n)

    low_dir  = os.path.join(out_dir, 'low')
    high_dir = os.path.join(out_dir, 'high')
    os.makedirs(low_dir,  exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)

    for i, (snow_path, gt_path) in enumerate(tqdm(selected, desc=f'复制到 {out_dir}')):
        # 统一命名避免冲突
        ext  = Path(snow_path).suffix
        fname = f'{i:06d}{ext}'
        shutil.copy2(snow_path, os.path.join(low_dir,  fname))
        shutil.copy2(gt_path,   os.path.join(high_dir, fname))

    print(f"  ✅ 完成！{len(selected)} 对 → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='CSD 数据集重组采样工具')
    parser.add_argument('--csd_root',  type=str, required=True,
                        help='CSD 解压后的根目录（含 Train/ 和 Test/ 子目录）')
    parser.add_argument('--out_train', type=str, default='./datasets/Snow_train',
                        help='训练集输出目录')
    parser.add_argument('--out_val',   type=str, default='./datasets/Snow_val',
                        help='验证集输出目录')
    parser.add_argument('--n_train',   type=int, default=600,
                        help='训练集采样数量（默认600，CSD Train共8000对）')
    parser.add_argument('--n_val',     type=int, default=100,
                        help='验证集采样数量（默认100，CSD Test共2000对）')
    parser.add_argument('--seed',      type=int, default=42,
                        help='随机种子（保证可复现）')
    args = parser.parse_args()

    print('=' * 50)
    print('CSD 数据集重组采样')
    print(f'  根目录: {args.csd_root}')
    print(f'  训练集: {args.n_train} 对 → {args.out_train}')
    print(f'  验证集: {args.n_val} 对  → {args.out_val}')
    print('=' * 50)

    # 训练集：从 CSD Train 中采样
    train_pairs = collect_csd_pairs(args.csd_root, split='Train')
    copy_pairs(train_pairs, args.out_train, n=args.n_train, seed=args.seed)

    # 验证集：从 CSD Test 中采样（避免与训练集重叠）
    val_pairs = collect_csd_pairs(args.csd_root, split='Test')
    copy_pairs(val_pairs, args.out_val, n=args.n_val, seed=args.seed + 1)

    print()
    print('✅ 全部完成！')
    print(f'   训练集: {args.out_train}/low/ + high/')
    print(f'   验证集: {args.out_val}/low/ + high/')
    print()
    print('下一步将这两个目录配置到 train.py:')
    print(f'   --data_snow_dirs {args.out_train}')
    print(f'   --data_snow_val  {args.out_val}')


if __name__ == '__main__':
    main()
