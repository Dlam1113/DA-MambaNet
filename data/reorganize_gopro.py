"""
GoPro Large 数据集重组采样脚本
============================

模块描述：
    深度学习数据预处理脚本。此脚本专门用于处理 GoPro 运动去模糊数据集（GOPRO_Large）。
    它的功能是将 GoPro 原始按照场景（scene）划分的复杂层级目录，展平并重组为 DA-MambaNet 
    中 AllInOneDataset 要求的统一 low/high 格式。
    此外，支持从海量数据中随机采样指定数量的图像对，构建训练和验证子集。

退化类型：
    运动模糊（Blur）-> 标签通常在主模型中映射为 4。

GOPRO_Large 原始结构（按场景划分）：
    GoPro_raw/
    ├── train/
    │   ├── GOPR0372_07_00/   ← 某个具体的视频场景
    │   │   ├── blur/         ← 模糊输入（low）
    │   │   │   ├── 000001.png
    │   │   │   └── ...
    │   │   └── sharp/        ← 清晰 GT（high）
    │   │       ├── 000001.png
    │   │       └── ...
    │   └── GOPR0xxx_xx_xx/   ← 其他场景
    └── test/
        └── GOPR0xxx_xx_xx/   ← 测试集场景

重组后结构约定（完全兼容 AllInOneDataset）：
    GoPro_train/
    ├── low/    ← 提取出的所有模糊图像（统一重命名并采样 n_train 张）
    └── high/   ← 提取出的所有清晰图像
    GoPro_val/
    ├── low/    ← 提取出的模糊图像（采样 n_val 张）
    └── high/   ← 对应的清晰图像

注意：
    由于 GoPro 数据集不同场景下可能存在同名文件（如 000001.png），
    本脚本在展平目录时会自动为文件重新编号为 `%06d.png`，避免在同一个 low 目录下发生文件名冲突。

使用方法（服务器终端执行）：
    python data/reorganize_gopro.py \
        --gopro_root ./datasets/GoPro_raw \
        --out_train  ./datasets/GoPro_train \
        --out_val    ./datasets/GoPro_val \
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


def collect_gopro_pairs(root: str, split: str = 'train') -> list:
    """
    收集 GoPro 数据集某个 split 下的所有 (blur_path, sharp_path) 配对。

    GoPro 结构：root/split/SCENE_NAME/blur/*.png + sharp/*.png
    文件名相同，一一对应。

    参数：
        root:   GOPRO_Large 根目录
        split:  'train' 或 'test'
    返回：
        pairs: [(blur_path, sharp_path), ...]  按场景排序后打乱
    """
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        # 兼容大写 Train/Test
        split_dir = os.path.join(root, split.capitalize())
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"找不到目录: {split_dir}，请检查 gopro_root 是否正确")

    pairs = []
    scene_dirs = sorted([d for d in os.listdir(split_dir)
                         if os.path.isdir(os.path.join(split_dir, d))])

    for scene in scene_dirs:
        scene_path = os.path.join(split_dir, scene)

        # 查找 blur 和 sharp 子目录（兼容大小写）
        blur_dir  = None
        sharp_dir = None
        for sub in os.listdir(scene_path):
            sub_lower = sub.lower()
            if sub_lower == 'blur':
                blur_dir = os.path.join(scene_path, sub)
            elif sub_lower == 'sharp':
                sharp_dir = os.path.join(scene_path, sub)

        if blur_dir is None or sharp_dir is None:
            print(f"  [警告] 场景 {scene} 缺少 blur/ 或 sharp/ 子目录，跳过")
            continue

        # 获取文件列表
        blur_files  = sorted([f for f in os.listdir(blur_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        sharp_files = sorted([f for f in os.listdir(sharp_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # 按文件名匹配（文件名应该相同）
        for bf, sf in zip(blur_files, sharp_files):
            pairs.append((
                os.path.join(blur_dir,  bf),
                os.path.join(sharp_dir, sf),
            ))

    print(f"  [GoPro {split}] 共收集到 {len(pairs)} 对")
    return pairs


def copy_pairs(pairs: list, out_dir: str, n: int, seed: int = 42):
    """
    从 pairs 中随机采样 n 对，复制到 out_dir/low/ 和 out_dir/high/

    参数：
        pairs:   [(blur_path, sharp_path), ...] 列表
        out_dir: 输出目录（会创建 low/ 和 high/ 子目录）
        n:       采样数量（如果 pairs 不足 n，则全取）
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

    for i, (blur_path, sharp_path) in enumerate(tqdm(selected, desc=f'复制到 {out_dir}')):
        # 统一命名为 000000.png 格式，避免文件名冲突
        ext = Path(blur_path).suffix
        fname = f'{i:06d}{ext}'
        shutil.copy2(blur_path,  os.path.join(low_dir,  fname))
        shutil.copy2(sharp_path, os.path.join(high_dir, fname))

    print(f"  ✅ 完成！{len(selected)} 对 → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='GoPro Large 数据集重组采样工具')
    parser.add_argument('--gopro_root', type=str, required=True,
                        help='GOPRO_Large 解压后的根目录（含 train/ 和 test/ 子目录）')
    parser.add_argument('--out_train',  type=str, default='./datasets/GoPro_train',
                        help='训练集输出目录')
    parser.add_argument('--out_val',    type=str, default='./datasets/GoPro_val',
                        help='验证集输出目录')
    parser.add_argument('--n_train',    type=int, default=600,
                        help='训练集采样数量（默认600）')
    parser.add_argument('--n_val',      type=int, default=100,
                        help='验证集采样数量（默认100）')
    parser.add_argument('--seed',       type=int, default=42,
                        help='随机种子（保证可复现）')
    args = parser.parse_args()

    print('=' * 50)
    print('GoPro Large 数据集重组采样')
    print(f'  根目录: {args.gopro_root}')
    print(f'  训练集: {args.n_train} 对 → {args.out_train}')
    print(f'  验证集: {args.n_val} 对 → {args.out_val}')
    print('=' * 50)

    # 收集训练集配对（从 GoPro train split）
    train_pairs = collect_gopro_pairs(args.gopro_root, split='train')
    copy_pairs(train_pairs, args.out_train, n=args.n_train, seed=args.seed)

    # 收集验证集配对（从 GoPro test split，避免与训练集重叠）
    val_pairs = collect_gopro_pairs(args.gopro_root, split='test')
    copy_pairs(val_pairs, args.out_val, n=args.n_val, seed=args.seed + 1)

    print()
    print('✅ 全部完成！')
    print(f'   训练集: {args.out_train}/low/ + high/')
    print(f'   验证集: {args.out_val}/low/ + high/')
    print()
    print('下一步将这两个目录配置到 train.py:')
    print(f'   --data_blur_dirs {args.out_train}')
    print(f'   --data_blur_val  {args.out_val}')


if __name__ == '__main__':
    main()
