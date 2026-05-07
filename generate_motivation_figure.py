"""
generate_motivation_figure.py
=============================
生成论文 Introduction Fig.1 的三个子面板（单独保存，方便后续拼图）。

  (a) HVI 逆变换量化损失热力图
  (b) I 通道直方图：线性 vs 曲线拉伸
  (c) 11 控制点 Neural Curve 可视化

用法: python generate_motivation_figure.py --input <低光照图片路径>
"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 全局字体设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})


# ============================================================
# 色彩空间工具
# ============================================================

def rgb_to_hsv(img):
    """RGB [0,1] → H(0~6), S(0~1), V(0~1)"""
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    v = np.max(img, axis=2)
    delta = v - np.min(img, axis=2)
    s = np.where(v > 0, delta / (v + 1e-10), 0.0)
    h = np.zeros_like(v)
    m_r = (v == R) & (delta > 0)
    h[m_r] = ((G[m_r] - B[m_r]) / (delta[m_r] + 1e-10)) % 6
    m_g = (v == G) & (delta > 0) & ~m_r
    h[m_g] = ((B[m_g] - R[m_g]) / (delta[m_g] + 1e-10)) + 2
    m_b = (v == B) & (delta > 0) & ~m_r & ~m_g
    h[m_b] = ((R[m_b] - G[m_b]) / (delta[m_b] + 1e-10)) + 4
    return h, s, v


def hsv_to_rgb(h, s, v):
    """H(0~6), S, V → RGB [0,1]"""
    h_i = (h.astype(int)) % 6
    f = h - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    rgb = np.zeros((*h.shape, 3))
    for i, (c1, c2, c3) in enumerate([
        (v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)
    ]):
        mask = h_i == i
        rgb[mask, 0] = c1[mask]
        rgb[mask, 1] = c2[mask]
        rgb[mask, 2] = c3[mask]
    return np.clip(rgb, 0, 1)


def hvi_forward(h, s, v):
    """HSV → HVI 极化变换"""
    angle = np.pi * h / 3.0
    return s * np.cos(angle), s * np.sin(angle), v


def hvi_inverse(H, V, I):
    """HVI → HSV 逆变换"""
    h = np.arctan2(V, H) * 3.0 / np.pi
    h[h < 0] += 6.0
    s = np.sqrt(H**2 + V**2)
    return h, np.clip(s, 0, 1), I


def quantize(arr, bits=8):
    """模拟定点量化: float → N-bit 整数 → float"""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return arr.copy()
    scale = 2**bits - 1
    norm = (arr - mn) / (mx - mn)
    return np.round(norm * scale) / scale * (mx - mn) + mn


def piecewise_curve(x, ctrl_y):
    """用 N+1 个控制点构建分段线性曲线"""
    ctrl_x = np.linspace(0, 1, len(ctrl_y))
    return np.interp(x, ctrl_x, ctrl_y)


# ============================================================
# Panel (a): HVI 逆变换量化损失
# ============================================================

def panel_a(img_np, output_dir):
    """
    生成面板 (a): 原图 / Round-trip 重建 / 放大误差热力图。
    通过在 HVI 空间做 8-bit 量化来模拟逆变换精度损失。
    """
    h, s, v = rgb_to_hsv(img_np)
    H, V, I = hvi_forward(h, s, v)

    # 量化模拟
    Hq, Vq, Iq = quantize(H), quantize(V), quantize(I)
    h2, s2, v2 = hvi_inverse(Hq, Vq, Iq)
    recon = hsv_to_rgb(h2, s2, v2)

    # 误差 (per-pixel L1, 取通道均值)
    error = np.mean(np.abs(img_np - recon), axis=2)
    amp = 30  # 放大倍数

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)

    # 原图
    axes[0].imshow(img_np)
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')

    # Round-trip 重建
    axes[1].imshow(recon)
    axes[1].set_title('HVI Round-trip', fontweight='bold')
    axes[1].axis('off')

    # 放大误差热力图
    im = axes[2].imshow(np.clip(error * amp, 0, 1), cmap='inferno', vmin=0, vmax=1)
    axes[2].set_title(f'Amplified Error (×{amp})', fontweight='bold', color='#CC0000')
    axes[2].axis('off')
    # 色标
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Error Magnitude', fontsize=9)

    fig.suptitle('(a) HVI⁻¹ Inverse-Conversion Quantization Loss',
                 fontsize=14, fontweight='bold', y=1.02, color='#CC0000')
    fig.tight_layout()

    out = os.path.join(output_dir, 'motivation_panel_a.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Panel (a): {out}")


# ============================================================
# Panel (b): I 通道直方图分析
# ============================================================

def panel_b(img_np, output_dir):
    """
    生成面板 (b): I 通道直方图对比。
    展示线性拉伸产生截断，曲线拉伸分布更均匀。
    """
    _, _, v_channel = rgb_to_hsv(img_np)

    # 模拟 11 控制点的 learned curve
    ctrl_pts = [0.0, 0.15, 0.35, 0.52, 0.62, 0.70, 0.76, 0.82, 0.88, 0.93, 0.97, 1.0]
    v_linear = np.clip(v_channel * 2.5, 0, 1)
    v_curve = piecewise_curve(v_channel, ctrl_pts)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)

    # 三个直方图叠加
    ax.hist(v_channel.ravel(), bins=100, range=(0, 1), alpha=0.55,
            color='#666666', label='Original I-channel', density=True, edgecolor='none')
    ax.hist(v_linear.ravel(), bins=100, range=(0, 1), alpha=0.45,
            color='#DD4444', label='Linear ×2.5 (clipped)', density=True, edgecolor='none')
    ax.hist(v_curve.ravel(), bins=100, range=(0, 1), alpha=0.45,
            color='#2266CC', label='Neural Curve stretch', density=True, edgecolor='none')

    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)

    # 标注线性拉伸的截断问题
    ax.axvline(x=1.0, color='#DD4444', linestyle='--', linewidth=1, alpha=0.7)
    ax.annotate('Clipping loss\n(information destroyed)',
                xy=(0.95, 0.5), xytext=(0.65, 4.0),
                fontsize=9, color='#DD4444', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#DD4444', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEEEE', alpha=0.8))

    # 标注曲线拉伸的均匀分布
    ax.annotate('Smooth redistribution\n(detail preserved)',
                xy=(0.45, 1.5), xytext=(0.45, 5.5),
                fontsize=9, color='#2266CC', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2266CC', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#EEF4FF', alpha=0.8))

    ax.set_title('(b) Linear Mapping Cannot Redistribute\n'
                 'Low-Light Dynamic Range',
                 fontsize=13, fontweight='bold', color='#CC0000', pad=12)

    fig.tight_layout()
    out = os.path.join(output_dir, 'motivation_panel_b.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Panel (b): {out}")


# ============================================================
# Panel (c): Neural Curve 可视化
# ============================================================

def panel_c(output_dir):
    """
    生成面板 (c): 11 控制点分段线性曲线 vs 线性映射。
    """
    ctrl_pts = [0.0, 0.15, 0.35, 0.52, 0.62, 0.70, 0.76, 0.82, 0.88, 0.93, 0.97, 1.0]
    ctrl_x = np.linspace(0, 1, len(ctrl_pts))
    x = np.linspace(0, 1, 300)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=200)

    # 背景填充: 曲线与线性之间的差异区域
    y_identity = x
    y_curve = piecewise_curve(x, ctrl_pts)
    ax.fill_between(x, y_identity, y_curve, alpha=0.12, color='#2266CC',
                    label='Enhancement gain')

    # Identity 线
    ax.plot(x, y_identity, '--', color='#AAAAAA', linewidth=1.8, label='Identity (y = x)')

    # 线性 ×2.5
    ax.plot(x, np.clip(x * 2.5, 0, 1), ':', color='#DD4444', linewidth=2,
            label='Linear ×2.5 (clipped)')

    # 学习曲线
    ax.plot(x, y_curve, '-', color='#2266CC', linewidth=2.5,
            label='Learned 11-pt Curve')

    # 控制点
    ax.scatter(ctrl_x, ctrl_pts, c='#2266CC', s=50, zorder=5,
               edgecolors='white', linewidth=1.2)

    # 标注几个关键控制点
    for i in [1, 3, 5, 8]:
        ax.annotate(f'P{i}({ctrl_x[i]:.1f}, {ctrl_pts[i]:.2f})',
                    xy=(ctrl_x[i], ctrl_pts[i]),
                    xytext=(ctrl_x[i] + 0.06, ctrl_pts[i] - 0.06),
                    fontsize=7, color='#2266CC',
                    arrowprops=dict(arrowstyle='-', color='#2266CC', lw=0.8))

    ax.set_xlabel('Input Intensity')
    ax.set_ylabel('Output Intensity')
    ax.legend(fontsize=8.5, loc='lower right', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    ax.set_title('(c) 11-Control-Point Neural Curve Layer\n'
                 'vs. Linear Intensity Mapping',
                 fontsize=13, fontweight='bold', color='#006600', pad=12)

    # 绿色边框
    for sp in ax.spines.values():
        sp.set_edgecolor('#006600')
        sp.set_linewidth(2)

    fig.tight_layout()
    out = os.path.join(output_dir, 'motivation_panel_c.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Panel (c): {out}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='生成 Motivation 示意图子面板')
    parser.add_argument('--input', type=str, required=True,
                        help='低光照输入图片路径')
    parser.add_argument('--output-dir', type=str, default='paper/figures',
                        help='输出目录 (默认 paper/figures)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取并缩放图片
    img = Image.open(args.input).convert('RGB')
    w, h = img.size
    scale = 400 / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img_np = np.array(img).astype(np.float64) / 255.0
    print(f"📷 输入图片: {args.input}  →  尺寸 {img_np.shape[:2]}")

    # 依次生成三个面板
    panel_a(img_np, args.output_dir)
    panel_b(img_np, args.output_dir)
    panel_c(args.output_dir)

    print(f"\n🎉 全部完成！输出在: {args.output_dir}/motivation_panel_*.png/pdf")
    print("提示: 面板 (d) 流程概念图建议使用 draw.io 或 PPT 手绘后拼合。")


if __name__ == '__main__':
    main()
