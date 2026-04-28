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
    解析 metrics md 文件中的"整体 (Combined) 指标"表格
    返回: dict { 'epochs': [...], 'psnr': [...], 'ssim': [...], 'lpips': [...] }
    """
    epochs, psnr, ssim, lpips = [], [], [], []
    in_table = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 检测表头
            if '| Epochs |' in line and 'PSNR' in line:
                in_table = True
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
DOC_DIR = os.path.join(SCRIPT_DIR, 'doc')

# --- 大数据集（雨天+雾天，200张验证） ---
WEATHER_BASELINE_MD = os.path.join(DOC_DIR, 'metrics2026-03-18-143143.md')  # CIDNet Baseline
WEATHER_REFINER_MD  = os.path.join(DOC_DIR, 'metrics2026-04-28-065833.md')  # +RGB Refiner only
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
def smooth(y, window=5):
    """对序列做滑动平均平滑"""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    # 边缘填充 (reflect)
    padded = np.pad(y, window//2, mode='reflect')
    return np.convolve(padded, kernel, mode='valid')[:len(y)]


def plot_one_metric(ax, data_list, metric_key, ylabel, invert=False):
    """
    在一个子图上绘制多条曲线

    参数:
        ax: matplotlib Axes
        data_list: list of dict, 每个dict有 name/color/style/data/hline
        metric_key: 'psnr' | 'ssim' | 'lpips'
        ylabel: Y轴标签
        invert: True = Y轴越低越好（LPIPS），画图时不翻转但可在注释中提示
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
            smoothed = smooth(values, window=7)
            # 原始曲线（淡色）
            ax.plot(epochs, values, color=d['color'], linewidth=0.6,
                    alpha=0.25, linestyle='-')
            # 平滑曲线（实色）
            ax.plot(epochs, smoothed, color=d['color'],
                    label=d['name'], **STYLES[d['style']])

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel('Epoch', fontsize=11)
    if invert:
        ax.invert_yaxis()  # LPIPS 越低越好
        ax.set_title(ylabel + ' ↓', fontsize=11, fontweight='bold')
    else:
        ax.set_title(ylabel + ' ↑', fontsize=11, fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9, loc='lower right' if not invert else 'upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================
# 4. 图1：大数据集（雨天+雾天）消融曲线
# ============================================================

def plot_weather_ablation():
    """绘制天气数据集消融曲线"""
    print("读取大数据集数据...")
    baseline_data = parse_metrics_md(WEATHER_BASELINE_MD)
    refiner_data  = parse_metrics_md(WEATHER_REFINER_MD)
    full_data     = parse_metrics_md(WEATHER_FULL_MD)

    print(f"  Baseline:    {len(baseline_data['epochs'])} 个检查点")
    print(f"  +Refiner:    {len(refiner_data['epochs'])} 个检查点")
    print(f"  Full Model:  {len(full_data['epochs'])} 个检查点")

    data_list = [
        {
            'name': 'CIDNet (Baseline)',
            'color': COLORS['baseline'], 'style': 'baseline',
            'data': baseline_data,
        },
        {
            'name': '+RGB Refiner only',
            'color': COLORS['refiner'], 'style': 'refiner',
            'data': refiner_data,
        },
        {
            'name': 'Full Model (Ours)',
            'color': COLORS['full'], 'style': 'full',
            'data': full_data,
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Ablation Study on Weather Dataset (Rain + Fog)',
                 fontsize=13, fontweight='bold', y=1.02)

    plot_one_metric(axes[0], data_list, 'psnr',  'PSNR (dB)')
    plot_one_metric(axes[1], data_list, 'ssim',  'SSIM')
    plot_one_metric(axes[2], data_list, 'lpips', 'LPIPS', invert=True)

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, 'ablation_weather.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'ablation_weather.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ 已保存: {out_png}")
    print(f"  ✅ 已保存: {out_pdf}")


# ============================================================
# 5. 图2：LOLv1 消融曲线
# ============================================================

def plot_lolv1_ablation():
    """绘制 LOLv1 消融曲线"""
    print("读取 LOLv1 数据...")
    baseline_data = parse_metrics_md(LOLV1_BASELINE_MD)
    refiner_data  = parse_metrics_md(LOLV1_REFINER_MD)
    full_data     = parse_metrics_md(LOLV1_FULL_MD)

    print(f"  Baseline:   {len(baseline_data['epochs'])} 个检查点")
    print(f"  +Refiner:   {len(refiner_data['epochs'])} 个检查点")
    print(f"  Full Model: {len(full_data['epochs'])} 个检查点")

    data_list = [
        {
            'name': 'CIDNet (Baseline)',
            'color': COLORS['baseline'], 'style': 'baseline',
            'data': baseline_data,
        },
        {
            'name': '+RGB Refiner only',
            'color': COLORS['refiner'], 'style': 'refiner',
            'data': refiner_data,
        },
        {
            'name': 'Full Model (Ours)',
            'color': COLORS['full'], 'style': 'full',
            'data': full_data,
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Ablation Study on LOLv1 (eval15)',
                 fontsize=13, fontweight='bold', y=1.02)

    plot_one_metric(axes[0], data_list, 'psnr',  'PSNR (dB)')
    plot_one_metric(axes[1], data_list, 'ssim',  'SSIM')
    plot_one_metric(axes[2], data_list, 'lpips', 'LPIPS', invert=True)

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, 'ablation_lolv1.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'ablation_lolv1.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ 已保存: {out_png}")
    print(f"  ✅ 已保存: {out_pdf}")


# ============================================================
# 6. 主函数
# ============================================================

if __name__ == '__main__':
    print("=" * 50)
    print("消融实验曲线图生成器")
    print("=" * 50)

    plot_weather_ablation()
    print()
    plot_lolv1_ablation()

    print()
    print("=" * 50)
    print("🎉 全部曲线图已生成！")
    print(f"   输出目录: {OUTPUT_DIR}")
    print("=" * 50)
