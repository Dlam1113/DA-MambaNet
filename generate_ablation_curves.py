"""
消融实验曲线图生成脚本
========================
输入：各训练阶段的 metrics md 文件（训练内部 measure.py 数据）
输出：两张出版级曲线图（大数据集 + LOLv1）

注意：曲线图用于展示趋势，最终数值在消融表格（ablation_report.md）中。
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ============================================================
# 1. 从 md 文件解析逐 epoch 指标
# ============================================================

def parse_metrics_md(filepath):
    """
    解析 metrics md 文件中的第一个「整体 (Combined) 指标」表格
    （文件中可能有多个子集表格，只取第一个，遇到第二个表头就停止）
    
    返回: dict { 'epochs': [...], 'psnr': [...], 'ssim': [...], 'lpips': [...] }
    """
    epochs, psnr, ssim, lpips = [], [], [], []
    in_table = False
    found_first = False   # 是否已经找到并进入了第一个表格

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 检测表头行
            if '| Epochs |' in line and 'PSNR' in line:
                if found_first:
                    # 遇到第二个表头（子集表），立即停止
                    break
                in_table = True
                found_first = True
                continue
            if in_table:
                # 跳过分隔行
                if line.strip().startswith('|---'):
                    continue
                # 解析数据行
                m = re.match(r'\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', line)
                if m:
                    epochs.append(int(m.group(1)))
                    psnr.append(float(m.group(2)))

                    ssim.append(float(m.group(3)))
                    lpips.append(float(m.group(4)))
                elif line.strip() == '' or line.strip().startswith('##'):
                    # 表格结束
                    in_table = False

    return {'epochs': epochs, 'psnr': psnr, 'ssim': ssim, 'lpips': lpips}


# ============================================================
# 2. 数据配置
# ============================================================

# md 文件路径（相对于脚本位置）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(SCRIPT_DIR, 'results','metrics')

# --- 大数据集（雨天+雾天，200张验证） ---
WEATHER_BASELINE_MD = os.path.join(DOC_DIR, 'metrics2026-04-28-065833.md')  # CIDNet Baseline
WEATHER_REFINER_MD  = os.path.join(DOC_DIR, 'metrics2026-03-18-143143.md')  # +RGB Refiner only
WEATHER_FULL_MD     = os.path.join(DOC_DIR, 'metrics2026-03-27-040647.md')  # 完整模型（有曲线数据）

# --- LOLv1（eval15, 15张验证） ---
LOLV1_BASELINE_MD = os.path.join(DOC_DIR, 'metrics2026-03-04-094555.md')  # Baseline（有曲线数据）
LOLV1_REFINER_MD  = os.path.join(DOC_DIR, 'metrics2026-04-25-212641.md')  # +RGB Refiner only
LOLV1_FULL_MD     = os.path.join(DOC_DIR, 'metrics2026-04-23-221213.md')  # 完整模型

# 输出目录
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'ablation')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. 绘图样式
# ============================================================

# 配色方案（学术风格）
COLORS = {
    'baseline': '#808080',     # 灰色：Baseline
    'refiner':  '#2196F3',     # 蓝色：+RGB Refiner
    'full':     '#E53935',     # 红色：完整模型（Ours）
}

STYLES = {
    'baseline': {'lw': 1.8, 'ls': '-',  'alpha': 0.85},
    'refiner':  {'lw': 1.8, 'ls': '-',  'alpha': 0.85},
    'full':     {'lw': 2.2, 'ls': '-',  'alpha': 1.00},
}

# 平滑函数（简单滑动平均，减少训练噪声）
def smooth(y, window=7):
    """对序列做滑动平均平滑"""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    padded = np.pad(y, window//2, mode='reflect')
    return np.convolve(padded, kernel, mode='valid')[:len(y)]


def resample(epochs, values, step):
    """
    按固定步长对 epoch 序列降采样，统一不同模型的采样间隔。
    例如 step=50，则只保留 epoch 为 50 的倍数的点。
    若原数据没有恰好整除的点，则取最近的那个点。
    """
    epochs = np.array(epochs)
    values = np.array(values)
    target_epochs = np.arange(epochs.min(), epochs.max() + step, step)
    sampled_e, sampled_v = [], []
    for te in target_epochs:
        # 找最近的 epoch
        idx = np.argmin(np.abs(epochs - te))
        sampled_e.append(epochs[idx])
        sampled_v.append(values[idx])
    # 去重（防止同一个 idx 被取两次）
    result_e, result_v = [], []
    seen = set()
    for e, v in zip(sampled_e, sampled_v):
        if e not in seen:
            seen.add(e)
            result_e.append(e)
            result_v.append(v)
    return np.array(result_e), np.array(result_v)


def filter_epoch(data, min_epoch=0, max_epoch=None):
    """
    在数据层面过滤 epoch 范围，使 matplotlib 计算 Y 轴范围时
    只考虑可见区间内的值，从而消除 Y 轴底部的大量空白。

    参数:
        data: parse_metrics_md 返回的 dict
        min_epoch: 保留 epoch >= min_epoch 的数据
        max_epoch: 保留 epoch <= max_epoch 的数据（None 表示不限）
    返回: 过滤后的同结构 dict
    """
    filtered = {'epochs': [], 'psnr': [], 'ssim': [], 'lpips': []}
    for i, e in enumerate(data['epochs']):
        if e < min_epoch:
            continue
        if max_epoch is not None and e > max_epoch:
            continue
        filtered['epochs'].append(e)
        filtered['psnr'].append(data['psnr'][i])
        filtered['ssim'].append(data['ssim'][i])
        filtered['lpips'].append(data['lpips'][i])
    return filtered


def plot_one_metric(ax, data_list, metric_key, ylabel, invert=False,
                    resample_step=None, show_legend=True, subplot_tag=None):
    """
    在一个子图上绘制多条曲线

    参数:
        ax: matplotlib Axes
        data_list: list of dict, 每个dict有 name/color/style/data/hline
        metric_key: 'psnr' | 'ssim' | 'lpips'
        ylabel: Y轴标签
        invert: True = Y轴越低越好（LPIPS），翻转 Y 轴
        resample_step: int, 统一采样间隔（epoch步长）。None=不降采样
        show_legend: 是否显示图例
        subplot_tag: 子图标签，如 '(a)'，显示在左上角
    """
    for d in data_list:
        if d.get('hline'):
            # 水平虚线（仅最佳值，无曲线数据）
            ax.axhline(
                y=d['hline'][metric_key],
                color=d['color'], linestyle='--', linewidth=1.8,
                alpha=0.9, label=d['name']
            )
        else:
            epochs = np.array(d['data']['epochs'])
            values = np.array(d['data'][metric_key])
            # 降采样：统一步长
            if resample_step is not None:
                epochs, values = resample(epochs, values, step=resample_step)
            smoothed = smooth(values, window=5)
            # 原始曲线（淡色，仅在不降采样时显示以避免过于密集）
            if resample_step is None:
                ax.plot(epochs, values, color=d['color'], linewidth=0.6,
                        alpha=0.25, linestyle='-')
            # 平滑曲线（实色）
            ax.plot(epochs, smoothed, color=d['color'],
                    label=d['name'], **STYLES[d['style']])

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel('Epoch', fontsize=10)
    if invert:
        ax.invert_yaxis()  # LPIPS 越低越好
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 子图标签 (a)~(f)，显示在左上角
    if subplot_tag:
        ax.text(0.02, 0.95, subplot_tag, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='gray', alpha=0.8))

    # 图例：仅在指定位置显示
    if show_legend:
        ax.legend(fontsize=7.5, loc='lower right')


# ============================================================
# 4. 合并消融曲线图（3行×2列，一张大图）
# ============================================================

def plot_combined_ablation():
    """
    生成一张合并的消融曲线图：3行×2列
    - 左列：天气数据集（Rain + Fog）
    - 右列：LOLv1（eval15）
    - 第1行：PSNR
    - 第2行：SSIM
    - 第3行：LPIPS
    - 子图标签：(a)~(f)
    """

    # --- 读取天气数据集 ---
    print("读取天气数据集数据...")
    w_baseline = parse_metrics_md(WEATHER_BASELINE_MD)
    w_refiner  = parse_metrics_md(WEATHER_REFINER_MD)
    w_full     = parse_metrics_md(WEATHER_FULL_MD)

    print(f"  Baseline:    {len(w_baseline['epochs'])} 个检查点")
    print(f"  +Refiner:    {len(w_refiner['epochs'])} 个检查点")
    print(f"  Full Model:  {len(w_full['epochs'])} 个检查点")

    # Full Model 日志从 ep655 开始，过滤掉 ep<650 的点
    MIN_EP = 650
    w_baseline = filter_epoch(w_baseline, min_epoch=MIN_EP)
    w_refiner  = filter_epoch(w_refiner,  min_epoch=MIN_EP)

    weather_list = [
        {'name': 'CIDNet (Baseline)', 'color': COLORS['baseline'],
         'style': 'baseline', 'data': w_baseline},
        {'name': '+RGB Refiner only', 'color': COLORS['refiner'],
         'style': 'refiner', 'data': w_refiner},
        {'name': 'Full Model (Ours)', 'color': COLORS['full'],
         'style': 'full', 'data': w_full},
    ]

    # --- 读取 LOLv1 数据 ---
    print("\n读取 LOLv1 数据...")
    l_baseline = parse_metrics_md(LOLV1_BASELINE_MD)
    l_refiner  = parse_metrics_md(LOLV1_REFINER_MD)
    l_full     = parse_metrics_md(LOLV1_FULL_MD)

    print(f"  Baseline:   {len(l_baseline['epochs'])} 个检查点")
    print(f"  +Refiner:   {len(l_refiner['epochs'])} 个检查点")
    print(f"  Full Model: {len(l_full['epochs'])} 个检查点")

    lolv1_list = [
        {'name': 'CIDNet (Baseline)', 'color': COLORS['baseline'],
         'style': 'baseline', 'data': l_baseline},
        {'name': '+RGB Refiner only', 'color': COLORS['refiner'],
         'style': 'refiner', 'data': l_refiner},
        {'name': 'Full Model (Ours)', 'color': COLORS['full'],
         'style': 'full', 'data': l_full},
    ]

    # --- 创建 3行×2列大图 ---
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    # 列标题
    axes[0, 0].set_title('Weather Dataset (Rain + Fog)', fontsize=11, fontweight='bold', pad=10)
    axes[0, 1].set_title('LOLv1 Dataset (eval15)', fontsize=11, fontweight='bold', pad=10)

    # 第1行：PSNR
    # (a) 天气 PSNR
    plot_one_metric(axes[0, 0], weather_list, 'psnr', 'PSNR (dB)',
                    resample_step=25, show_legend=True, subplot_tag='(a)')
    # (b) LOLv1 PSNR
    plot_one_metric(axes[0, 1], lolv1_list, 'psnr', 'PSNR (dB)',
                    resample_step=50, show_legend=False, subplot_tag='(b)')

    # 第2行：SSIM
    # (c) 天气 SSIM
    plot_one_metric(axes[1, 0], weather_list, 'ssim', 'SSIM',
                    resample_step=25, show_legend=False, subplot_tag='(c)')
    # (d) LOLv1 SSIM
    plot_one_metric(axes[1, 1], lolv1_list, 'ssim', 'SSIM',
                    resample_step=50, show_legend=False, subplot_tag='(d)')

    # 第3行：LPIPS（越低越好）
    # (e) 天气 LPIPS
    plot_one_metric(axes[2, 0], weather_list, 'lpips', 'LPIPS',
                    invert=True, resample_step=25, show_legend=False, subplot_tag='(e)')
    # (f) LOLv1 LPIPS
    plot_one_metric(axes[2, 1], lolv1_list, 'lpips', 'LPIPS',
                    invert=True, resample_step=50, show_legend=False, subplot_tag='(f)')

    plt.tight_layout(h_pad=2.5, w_pad=2.0)

    # 保存
    out_png = os.path.join(OUTPUT_DIR, 'ablation_combined.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'ablation_combined.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  ✅ 已保存: {out_png}")
    print(f"  ✅ 已保存: {out_pdf}")


# ============================================================
# 5. 主函数
# ============================================================

if __name__ == '__main__':
    print("=" * 50)
    print("消融实验曲线图生成器（合并版 3×2 大图）")
    print("=" * 50)

    plot_combined_ablation()

    print()
    print("=" * 50)
    print("🎉 消融曲线图已生成！")
    print(f"   输出目录: {OUTPUT_DIR}")
    print("=" * 50)
