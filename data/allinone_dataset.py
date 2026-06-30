"""
DA-MambaNet 专用多退化 All-in-One 数据集

功能：
    将三类退化数据集（低光照 / 雾天 / 雨天）混合为统一训练集，
    并为每张图像附加退化类型标签，供 DAM 分类辅助损失使用。

退化类型标签定义：
    0 → 低光照（Low-light，如 LOLv1/LOLv2/LoLI-Street）
    1 → 雾天（Fog/Haze，如 Cityscapes Foggy）
    2 → 雨天（Rain，如 Cityscapes Rain/Rain100H/Rain100L）

数据集目录结构（每种退化类型相同）：
    <dataset_root>/
    ├── low/     ← 退化输入图像（低光照 or 有雾 or 有雨）
    └── high/    ← 对应清洁 GT 图像

使用示例：
    dataset = AllInOneDataset(
        lol_dirs   = ['./datasets/LOLv1/train', './datasets/LOLv2/train'],
        fog_dirs   = ['./datasets/cityscapes_foggy/train'],
        rain_dirs  = ['./datasets/rain100H/train'],
        transform  = transform_train(256)
    )
    # dataset[i] 返回 (im_low, im_gt, filename_low, filename_gt, label)
    # label: 0=低光, 1=雾, 2=雨

作者：DA-MambaNet 项目
"""

import os
import random
import torch
import torch.utils.data as data
from os import listdir
from os.path import join
from data.util import is_image_file, load_img


# 退化类型标签映射
DEGRADATION_LABELS = {
    'lowlight': 0,   # 低光照
    'fog':      1,   # 雾天
    'rain':     2,   # 雨天
}


class AllInOneDataset(data.Dataset):
    """
    DA-MambaNet 多退化 All-in-One 训练数据集

    核心特性：
    1. 将多种退化数据集（低光/雾/雨）混合为单一 Dataset
    2. 为每张图像附加退化类型标签（整数0/1/2）
    3. 确保 low/high 图像对使用相同随机数种子（保证空间一致的数据增强）
    4. 支持数据集内部平衡（可选），避免类别严重不均衡

    参数：
        lol_dirs:   低光照数据集目录列表（每个目录含 low/ 和 high/）
        fog_dirs:   雾天数据集目录列表
        rain_dirs:  雨天数据集目录列表
        transform:  图像变换函数（随机裁剪、翻转等）
        balance:    是否按类别平衡（True=欠采样多数类, False=直接合并）
    """

    def __init__(self, lol_dirs=None, fog_dirs=None, rain_dirs=None,
                 transform=None, balance=False):
        super(AllInOneDataset, self).__init__()
        self.transform = transform

        # 存储所有样本 (low_path, high_path, label)
        self.samples = []

        # 按类别分别收集样本
        class_samples = {0: [], 1: [], 2: []}

        for label, dirs in [
            (DEGRADATION_LABELS['lowlight'], lol_dirs  or []),
            (DEGRADATION_LABELS['fog'],      fog_dirs  or []),
            (DEGRADATION_LABELS['rain'],     rain_dirs or []),
        ]:
            for data_dir in dirs:
                low_dir  = os.path.join(data_dir, 'low')
                high_dir = os.path.join(data_dir, 'high')

                if not os.path.isdir(low_dir) or not os.path.isdir(high_dir):
                    print(f"  [警告] 跳过无效目录 {data_dir}（缺少 low/ 或 high/）")
                    continue

                low_files  = sorted([f for f in listdir(low_dir)  if is_image_file(f)])
                high_files = sorted([f for f in listdir(high_dir) if is_image_file(f)])

                # 确保数量匹配
                if len(low_files) != len(high_files):
                    print(f"  [警告] {data_dir}: low({len(low_files)}) ≠ high({len(high_files)})，取最小值")
                    n = min(len(low_files), len(high_files))
                    low_files  = low_files[:n]
                    high_files = high_files[:n]

                label_name = [k for k, v in DEGRADATION_LABELS.items() if v == label][0]
                for lf, hf in zip(low_files, high_files):
                    class_samples[label].append((
                        os.path.join(low_dir,  lf),
                        os.path.join(high_dir, hf),
                        label
                    ))

                print(f"  ✅ {label_name}({label}): {data_dir} → {len(low_files)} 对")

        # 按类别平衡
        if balance:
            # 找到数量最多的类，其余类重复采样到相同数量
            max_count = max(len(v) for v in class_samples.values() if v)
            for label, samps in class_samples.items():
                if not samps:
                    continue
                while len(samps) < max_count:
                    samps.extend(samps[:max_count - len(samps)])
            print(f"  [平衡] 每类数量均衡到 {max_count}")

        for label, samps in class_samples.items():
            self.samples.extend(samps)

        # 打乱顺序（避免按类别顺序训练）
        random.shuffle(self.samples)

        # 统计
        n = {0: 0, 1: 0, 2: 0}
        for _, _, l in self.samples:
            n[l] += 1
        print(f"\n  [AllInOneDataset] 总计 {len(self.samples)} 对:")
        print(f"    低光照(0): {n[0]} | 雾天(1): {n[1]} | 雨天(2): {n[2]}")

    def __getitem__(self, index):
        """
        获取一个训练样本

        返回：
            im_low:     退化输入图像 Tensor (3, H, W)
            im_gt:      清洁 GT 图像 Tensor (3, H, W)
            file_low:   低质量图像文件名（调试用）
            file_gt:    GT 图像文件名（调试用）
            label:      退化类型标签 int (0=低光, 1=雾, 2=雨)
        """
        low_path, high_path, label = self.samples[index]

        im_low = load_img(low_path)
        im_gt  = load_img(high_path)

        _, file_low = os.path.split(low_path)
        _, file_gt  = os.path.split(high_path)

        # 同步随机数种子：确保 low/high 做完全相同的空间变换
        seed = random.randint(1, 1_000_000)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im_low = self.transform(im_low)

            random.seed(seed)
            torch.manual_seed(seed)
            im_gt = self.transform(im_gt)

        return im_low, im_gt, file_low, file_gt, label

    def __len__(self):
        return len(self.samples)


class AllInOneEvalDataset(data.Dataset):
    """
    DA-MambaNet 多退化 All-in-One 验证数据集

    功能：
        提供验证时的输入图像，以及对应的 GT 路径用于计算 PSNR/SSIM。
        可以一次评估所有退化类型，或按退化类型分别评估。

    参数：
        data_dirs:  验证数据目录列表（每个目录含 low/ 和 high/）
        labels:     与 data_dirs 等长的标签列表（0/1/2），None 表示混合评估
        transform:  图像变换（通常只有 ToTensor）
    """

    def __init__(self, data_dirs, labels=None, transform=None):
        super(AllInOneEvalDataset, self).__init__()
        self.transform   = transform
        self.low_files   = []
        self.high_files  = []
        self.labels_list = []

        for i, data_dir in enumerate(data_dirs):
            low_dir  = os.path.join(data_dir, 'low')
            high_dir = os.path.join(data_dir, 'high')

            if not os.path.isdir(low_dir):
                print(f"  [警告] 验证目录不存在: {low_dir}")
                continue

            lows  = sorted([os.path.join(low_dir,  f) for f in listdir(low_dir)  if is_image_file(f)])
            highs = sorted([os.path.join(high_dir, f) for f in listdir(high_dir) if is_image_file(f)])

            label = labels[i] if labels else -1   # -1 表示未知
            self.low_files.extend(lows)
            self.high_files.extend(highs)
            self.labels_list.extend([label] * len(lows))

        print(f"  [AllInOneEvalDataset] 验证集共 {len(self.low_files)} 张")

    def __getitem__(self, index):
        """
        返回：
            input_img:  退化输入 Tensor
            filename:   文件名
            h, w:       原始高宽（用于去掉 padding）
            label:      退化类型
        """
        import torch.nn.functional as F

        input_img = load_img(self.low_files[index])
        _, fname = os.path.split(self.low_files[index])
        label = self.labels_list[index]

        if self.transform:
            input_img = self.transform(input_img)
            # 填充到 8 的倍数（网络下采样要求）
            factor = 8
            h, w = input_img.shape[1], input_img.shape[2]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = F.pad(input_img.unsqueeze(0), (0, padw, 0, padh), 'reflect').squeeze(0)
        else:
            h = w = 0

        return input_img, fname, h, w, label

    def __len__(self):
        return len(self.low_files)

    def get_gt_path(self, index):
        """获取 GT 图像路径（用于计算 PSNR/SSIM）"""
        return self.high_files[index]


# ==============================================================================
# 便捷工厂函数（供 train.py 的 load_datasets 调用）
# ==============================================================================
def get_allinone_training_set(lol_dirs, fog_dirs, rain_dirs,
                               crop_size=256, balance=False):
    """
    构建 DA-MambaNet 多退化训练集。

    参数：
        lol_dirs:    低光照数据集目录列表
        fog_dirs:    雾天数据集目录列表
        rain_dirs:   雨天数据集目录列表
        crop_size:   随机裁剪大小
        balance:     是否按类别平衡
    返回：
        AllInOneDataset
    """
    from torchvision import transforms as T
    transform = T.Compose([
        T.RandomCrop(crop_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    return AllInOneDataset(
        lol_dirs  = lol_dirs,
        fog_dirs  = fog_dirs,
        rain_dirs = rain_dirs,
        transform = transform,
        balance   = balance,
    )


def get_allinone_eval_set(val_dirs, labels):
    """
    构建 DA-MambaNet 多退化验证集。

    参数：
        val_dirs:  验证数据目录列表
        labels:    对应的退化类型标签列表 [0, 0, 1, 2, ...]
    返回：
        AllInOneEvalDataset
    """
    from torchvision import transforms as T
    transform = T.ToTensor()
    return AllInOneEvalDataset(
        data_dirs = val_dirs,
        labels    = labels,
        transform = transform,
    )
