"""
DA-MambaNet 专用多退化 All-in-One 数据集

功能描述：
    本项目使用统一的框架处理多种图像退化问题。此模块负责将五类退化数据集
    （低光照 / 雾天 / 雨天 / 雪天 / 运动模糊）混合构建为统一的训练和验证集，
    并为每个样本自动生成退化类型标签，主要用于训练过程中的退化感知模块（DAM）
    的分类辅助损失，帮助网络感知当前的退化类型。

退化类型标签映射定义（5类）：
    0 → 低光照（Low-light，例如 LOLv1 数据集）
    1 → 雾天（Fog/Haze，例如 Cityscapes Foggy / RESIDE 数据集）
    2 → 雨天（Rain，例如 Rain100H / Rain100L 数据集）
    3 → 雪天（Snow，例如 CSD 去雪数据集）
    4 → 运动模糊（Blur，例如 GoPro 运动去模糊数据集）

数据集目录结构约定：
    所有的退化数据集（不论原格式如何）在输入此模块前，都需要被重新组织成相同的结构：
    <dataset_root>/
    ├── low/     ← 存放退化输入图像（低质量图像：有雾、有雨、模糊等）
    └── high/    ← 存放对应的清洁 GT（Ground Truth）图像（高质量清晰图像）
    注意：low 和 high 文件夹中的图片必须是一一对应的（通常要求文件名相同或按排序匹配）。

使用示例：
    dataset = AllInOneDataset(
        lol_dirs   = ['./datasets/LOLv1/train'],
        fog_dirs   = ['./datasets/Fog_train'],
        rain_dirs  = ['./datasets/Rain_train'],
        snow_dirs  = ['./datasets/Snow_train'],
        blur_dirs  = ['./datasets/GoPro/train'],
        transform  = transform_train(256)
    )
    # dataset[i] 将返回 (im_low, im_gt, filename_low, filename_gt, label)

作者：DA-MambaNet 项目
"""

import os
import random
import torch
import torch.utils.data as data
from os import listdir
from os.path import join
from data.util import is_image_file, load_img


# ============================================================
# 退化类型标签映射（5类）
# 修改这里会同步影响 DAM 的分类头输出维度
# 务必与 train.py 中的 --num_classes 参数保持一致
# ============================================================
DEGRADATION_LABELS = {
    'lowlight': 0,   # 低光照
    'fog':      1,   # 雾天
    'rain':     2,   # 雨天
    'snow':     3,   # 雪天
    'blur':     4,   # 运动模糊
}
NUM_CLASSES = len(DEGRADATION_LABELS)   # = 5



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
                 snow_dirs=None, blur_dirs=None,
                 transform=None, balance=False):
        """
        初始化五退化混合数据集

        参数：
            lol_dirs:   低光照数据集目录列表
            fog_dirs:   雾天数据集目录列表
            rain_dirs:  雨天数据集目录列表
            snow_dirs:  雪天数据集目录列表（新增，CSD等）
            blur_dirs:  运动模糊数据集目录列表（新增，GoPro等）
            transform:  图像变换函数
            balance:    是否按类别平衡样本数量
        """
        super(AllInOneDataset, self).__init__()
        self.transform = transform

        # 存储所有样本 (low_path, high_path, label)
        self.samples = []

        # 按类别分别收集样本（5类）
        class_samples = {i: [] for i in range(NUM_CLASSES)}

        for label, dirs in [
            (DEGRADATION_LABELS['lowlight'], lol_dirs  or []),
            (DEGRADATION_LABELS['fog'],      fog_dirs  or []),
            (DEGRADATION_LABELS['rain'],     rain_dirs or []),
            (DEGRADATION_LABELS['snow'],     snow_dirs or []),
            (DEGRADATION_LABELS['blur'],     blur_dirs or []),
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

        # 按类别平衡（过采样少数类到最多类的数量）
        if balance:
            counts = [len(v) for v in class_samples.values() if v]
            if counts:
                max_count = max(counts)
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

        # 统计各类数量
        n = {i: 0 for i in range(NUM_CLASSES)}
        for _, _, l in self.samples:
            n[l] += 1
        label_names = {v: k for k, v in DEGRADATION_LABELS.items()}
        print(f"\n  [AllInOneDataset] 总计 {len(self.samples)} 对（{NUM_CLASSES}类退化）:")
        for i in range(NUM_CLASSES):
            print(f"    {label_names[i]}({i}): {n[i]} 对")

    def __getitem__(self, index):
        """
        获取一个训练样本

        参数：
            index: 样本索引

        返回：
            im_low:     退化输入图像 Tensor，维度为 (C, H, W) -> (3, H, W)
            im_gt:      清洁 GT 图像 Tensor，维度为 (C, H, W) -> (3, H, W)
            file_low:   低质量图像文件名（用于调试和记录）
            file_gt:    GT 图像文件名（用于调试和记录）
            label:      退化类型标签 int（0=低光, 1=雾, 2=雨, 3=雪, 4=模糊）
        """
        low_path, high_path, label = self.samples[index]

        # 读取图像为张量 (C, H, W)
        im_low = load_img(low_path)
        im_gt  = load_img(high_path)

        _, file_low = os.path.split(low_path)
        _, file_gt  = os.path.split(high_path)

        # 同步随机数种子：深度学习中对于图像恢复任务，
        # 必须确保输入图像 (im_low) 和标签图像 (im_gt) 进行完全相同的空间变换（如随机裁剪、翻转），
        # 否则网络将无法学习到正确的像素级映射关系。
        seed = random.randint(1, 1_000_000)
        if self.transform:
            # 对退化图像进行变换
            random.seed(seed)
            torch.manual_seed(seed)
            im_low = self.transform(im_low)

            # 对清晰图像进行完全相同的变换
            random.seed(seed)
            torch.manual_seed(seed)
            im_gt = self.transform(im_gt)

        return im_low, im_gt, file_low, file_gt, label

    def __len__(self):
        return len(self.samples)


class AllInOneEvalDataset(data.Dataset):
    """
    DA-MambaNet 多退化快速验证集（Quick Validation Set）

    功能：
        提供训练期间快速验证所需的输入图像、GT 路径和类别标签。
        每个目录按固定排序截取前 max_samples_per_dir 张，用于趋势观察；
        该固定子集不是论文最终测试集。

    参数：
        data_dirs:  验证数据目录列表（每个目录含 low/ 和 high/）
        labels:     与 data_dirs 等长的标签列表（0/1/2），None 表示混合评估
        transform:  图像变换（通常只有 ToTensor）
    """

    def __init__(self, data_dirs, labels=None, transform=None, max_samples_per_dir=None):
        super(AllInOneEvalDataset, self).__init__()
        self.transform   = transform
        self.low_files   = []
        self.high_files  = []
        self.labels_list = []
        self.prefix_list = []

        for i, data_dir in enumerate(data_dirs):
            low_dir  = os.path.join(data_dir, 'low')
            high_dir = os.path.join(data_dir, 'high')

            if not os.path.isdir(low_dir):
                print(f"  [警告] 验证目录不存在: {low_dir}")
                continue

            lows  = sorted([os.path.join(low_dir,  f) for f in listdir(low_dir)  if is_image_file(f)])
            highs = sorted([os.path.join(high_dir, f) for f in listdir(high_dir) if is_image_file(f)])

            # 快速验证固定子集：按文件名排序后截取前 N 张。
            # 该策略保持跨 epoch 一致；LOLv1 只有15张，因此不会被扩充。
            if max_samples_per_dir and max_samples_per_dir > 0:
                lows  = lows[:max_samples_per_dir]
                highs = highs[:max_samples_per_dir]

            label = labels[i] if labels else -1   # -1 表示未知
            prefix = os.path.basename(os.path.normpath(data_dir))
            self.low_files.extend(lows)
            self.high_files.extend(highs)
            self.labels_list.extend([label] * len(lows))
            self.prefix_list.extend([prefix] * len(lows))

        print(
            f"  [Quick Validation Set] 共 {len(self.low_files)} 张 "
            f"(每类固定排序上限: {max_samples_per_dir if max_samples_per_dir else '无限制'})"
        )

    def __getitem__(self, index):
        """
        获取一个验证样本

        参数：
            index: 样本索引

        返回：
            input_img:  退化输入 Tensor，维度为 (3, H, W)
            filename:   文件名（带数据集前缀，防止同名文件覆盖）
            h, w:       原始图像的高度和宽度（用于评估后去掉 padding，计算真实的指标）
            label:      退化类型（用于分类统计指标）
        """
        import torch.nn.functional as F

        input_img = load_img(self.low_files[index])
        _, raw_fname = os.path.split(self.low_files[index])
        prefix = self.prefix_list[index]
        fname = f"{prefix}_{raw_fname}"
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
                               snow_dirs=None, blur_dirs=None,
                               crop_size=256, balance=False):
    """
    构建 DA-MambaNet 多退化训练集（5类退化）。

    参数：
        lol_dirs:    低光照数据集目录列表
        fog_dirs:    雾天数据集目录列表
        rain_dirs:   雨天数据集目录列表
        snow_dirs:   雪天数据集目录列表（CSD等）
        blur_dirs:   运动模糊数据集目录列表（GoPro等）
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
        snow_dirs = snow_dirs,
        blur_dirs = blur_dirs,
        transform = transform,
        balance   = balance,
    )


def get_allinone_eval_set(val_dirs, labels, max_samples_per_dir=None):
    """
    构建 DA-MambaNet 多退化快速验证集（Quick Validation Set）。

    参数：
        val_dirs:             验证数据目录列表
        labels:               对应的退化类型标签列表 [0, 1, 2, 3, 4]
        max_samples_per_dir: 每个目录按固定排序截取的最大样本数（例如30）
    返回：
        AllInOneEvalDataset
    """
    from torchvision import transforms as T
    transform = T.ToTensor()
    return AllInOneEvalDataset(
        data_dirs           = val_dirs,
        labels              = labels,
        transform           = transform,
        max_samples_per_dir = max_samples_per_dir,
    )
