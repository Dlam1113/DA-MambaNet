"""
条件化 Mamba 块（Conditional Mamba Block, CMB）

功能：
    替换原 DualSpaceCIDNet 中的 LCA（Lightweight Cross-Attention）模块，
    在 HVI 双流架构中实现基于 Mamba SSM 的全局建模 + FiLM 退化感知条件调制。

设计思路：
    原 LCA 结构：
        x = x + CAB(LN(x), LN(y))   # 交叉注意力：x 关注 y
        x = IEL(LN(x))               # 强度增强

    新 CMB 结构：
        x = x + MambaLayer(LN(x))    # Mamba 替代交叉注意力：全局建模
        x = x + LocalConv(LN(x))     # 局部卷积：补偿 Mamba 的局部像素遗忘（来自 MambaIR）
        x = FiLMLayer(x, d)          # FiLM 条件调制：根据退化信息调整特征
        x = x + IEL(LN(x))           # 强度增强（保留原有模块）

两种变体（基于 HVI 物理特性的异构扫描策略，这是创新点之一）：
    HV_CMB：色度流，使用 2 方向扫描（水平 + 垂直）
             原因：色度信息在局部区域内连续，2方向已足以捕捉色彩模式
    I_CMB：  光照流，使用 4 方向全扫描
             原因：光照变化具有全局性（阴影、光源），需要更全面的感受野

兼容说明：
    - 在没有 mamba-ssm 的 Windows 本地环境中，自动降级为伪 Mamba（线性层模拟）
    - 在 Linux 服务器安装 mamba-ssm 后，自动使用真实 Mamba
    - 接口设计与 HV_LCA / I_LCA 完全兼容，方便直接替换

作者：DA-MambaNet 项目
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------- 依赖导入（优雅降级处理）----------
# 在 Windows 本地开发环境中，mamba-ssm 不可用（需要 Linux + CUDA）
# 使用伪 Mamba 模块保证代码在 Windows 上可运行和测试
try:
    from mamba_ssm import Mamba as MambaSSM
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

# 从项目现有模块导入
from net.transformer_utils import LayerNorm
from net.LCA import IEL
from net.FiLM import FiLMLayer


# ==============================================================================
#   辅助模块
# ==============================================================================

class MockMamba(nn.Module):
    """
    伪 Mamba 模块（仅用于 Windows 本地代码调试和 shape 验证）

    用真实 Mamba 的接口（输入 (B, L, D)，输出 (B, L, D)），
    但内部只用 Linear + 残差实现，不涉及任何 CUDA 编译。

    警告：此模块不具备真实 Mamba 的选择性记忆能力，仅用于开发阶段验证。
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        # 用两层线性层模拟 Mamba 的输入-输出映射
        inner_dim = d_model * expand
        self.in_proj  = nn.Linear(d_model, inner_dim)
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.act      = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：x: (B, L, D)
        返回：(B, L, D)
        """
        return self.out_proj(self.act(self.in_proj(x)))


def build_mamba(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
    """
    构建 Mamba 层的工厂函数。
    
    在 Linux + mamba-ssm 环境中返回真实 Mamba；
    在 Windows 开发环境中返回 MockMamba。
    
    参数：
        d_model: 模型维度（通道数）
        d_state: SSM 状态空间维度（越大，记忆容量越大，计算越慢）
        d_conv:  局部卷积核大小（用于 Mamba 内部的局部上下文建模）
        expand:  通道扩展倍数（内部维度 = d_model * expand）
    
    返回：
        Mamba 层或 MockMamba 层（接口兼容）
    """
    if MAMBA_AVAILABLE:
        return MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    else:
        return MockMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class SS2D(nn.Module):
    """
    2D 选择性扫描模块（2D Selective Scan，参考 VMamba）

    核心思路：
        图像是二维结构，Mamba 只能处理一维序列。
        通过沿多个方向展平图像，让每个像素都能"看到"来自不同方向的邻居。

    支持的扫描方向数：
        num_scan=2：水平扫描（左→右）+ 垂直扫描（上→下）
                    适合 HV 色度流（局部色彩模式为主）
        num_scan=4：水平正+反 + 垂直正+反（4方向全扫）
                    适合 I 光照流（全局光照分布）

    参数：
        dim:      输入特征通道数
        d_state:  SSM 状态空间维度
        d_conv:   Mamba 内部局部卷积核大小
        expand:   Mamba 通道扩展倍数
        num_scan: 扫描方向数，2或4
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, num_scan: int = 4):
        super().__init__()
        assert num_scan in [2, 4], f"num_scan 必须为 2 或 4，当前：{num_scan}"
        self.num_scan = num_scan

        # 每个扫描方向对应一个独立的 Mamba 实例
        # 独立参数让每个方向学习方向特有的时序依赖
        self.mamba_layers = nn.ModuleList([
            build_mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_scan)
        ])

        # 多方向特征融合：将 num_scan 路结果合并为一路
        # 如果 num_scan=1 则不需要投影，直接输出
        if num_scan > 1:
            # 1×1 卷积实现通道融合（等价于逐点全连接）
            self.merge = nn.Conv2d(dim * num_scan, dim, kernel_size=1, bias=False)
        else:
            self.merge = nn.Identity()

    def _scan_horizontal(self, x: torch.Tensor) -> torch.Tensor:
        """
        水平方向扫描：按行展平，从左到右逐像素处理。
        
        参数：x: (B, C, H, W)
        返回：(B, C, H, W)
        """
        B, C, H, W = x.shape
        # (B, C, H, W) → (B, H*W, C)：按行展平为序列
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        # Mamba 处理序列
        out_seq = self.mamba_layers[0](x_seq)            # (B, H*W, C)
        # 恢复为图像形状
        return rearrange(out_seq, 'b (h w) c -> b c h w', h=H, w=W)

    def _scan_vertical(self, x: torch.Tensor, mamba_idx: int = 1) -> torch.Tensor:
        """
        垂直方向扫描：按列展平，从上到下逐像素处理。
        
        参数：
            x:          (B, C, H, W)
            mamba_idx:  使用第几个 Mamba 实例
        返回：(B, C, H, W)
        """
        B, C, H, W = x.shape
        # 转置空间维度后展平：(B, C, H, W) → (B, W, H, C) 按列扫描
        x_T = x.permute(0, 1, 3, 2)                          # (B, C, W, H)
        x_seq = rearrange(x_T, 'b c w h -> b (w h) c')        # 按列展平
        out_seq = self.mamba_layers[mamba_idx](x_seq)
        # 恢复
        out_T = rearrange(out_seq, 'b (w h) c -> b c w h', w=W, h=H)
        return out_T.permute(0, 1, 3, 2)                      # (B, C, H, W)

    def _scan_reverse(self, x: torch.Tensor, direction: str, mamba_idx: int) -> torch.Tensor:
        """
        反向扫描：将序列翻转后送入 Mamba，实现双向上下文建模。
        
        参数：
            x:          (B, C, H, W)
            direction:  'horizontal' 或 'vertical'
            mamba_idx:  使用的 Mamba 索引
        返回：(B, C, H, W)
        """
        B, C, H, W = x.shape
        if direction == 'horizontal':
            x_seq = rearrange(x, 'b c h w -> b (h w) c')
        else:
            x_T = x.permute(0, 1, 3, 2)
            x_seq = rearrange(x_T, 'b c w h -> b (w h) c')

        # 沿序列维度翻转，实现反向扫描
        x_seq_rev = x_seq.flip(dims=[1])
        out_seq = self.mamba_layers[mamba_idx](x_seq_rev)
        # 再次翻转恢复顺序
        out_seq = out_seq.flip(dims=[1])

        if direction == 'horizontal':
            return rearrange(out_seq, 'b (h w) c -> b c h w', h=H, w=W)
        else:
            out_T = rearrange(out_seq, 'b (w h) c -> b c w h', w=W, h=H)
            return out_T.permute(0, 1, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多方向扫描并融合。
        
        参数：x: (B, C, H, W)
        返回：(B, C, H, W)
        """
        if self.num_scan == 2:
            # 2 方向：水平正向 + 垂直正向（用于 HV 色度流）
            out_h = self._scan_horizontal(x)               # 水平
            out_v = self._scan_vertical(x, mamba_idx=1)    # 垂直
            # 通道拼接后融合：(B, 2C, H, W) → (B, C, H, W)
            return self.merge(torch.cat([out_h, out_v], dim=1))

        else:  # num_scan == 4
            # 4 方向：水平正+反 + 垂直正+反（用于 I 光照流）
            out_h      = self._scan_horizontal(x)                          # 水平正向
            out_h_rev  = self._scan_reverse(x, 'horizontal', mamba_idx=1)  # 水平反向
            out_v      = self._scan_vertical(x, mamba_idx=2)               # 垂直正向
            out_v_rev  = self._scan_reverse(x, 'vertical',   mamba_idx=3)  # 垂直反向
            # 4路拼接融合：(B, 4C, H, W) → (B, C, H, W)
            return self.merge(torch.cat([out_h, out_h_rev, out_v, out_v_rev], dim=1))


class LocalConvEnhancer(nn.Module):
    """
    局部卷积增强模块（来自 MambaIR 的设计，解决"局部像素遗忘"问题）

    Mamba 的扫描特性在处理 2D 图像时，空间相邻但序列上距离远的像素难以高效交互。
    局部卷积通过 3×3 感受野对局部邻域像素进行显式建模，弥补这一不足。

    结构：
        x → BN → 深度卷积(3×3) → 逐点卷积(1×1) → 残差连接 → 输出

    参数：
        dim:         输入输出通道数
        expand:      中间通道扩展倍数（默认2，平衡表达力与参数量）
    """

    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        inner_dim = dim * expand
        self.norm    = nn.BatchNorm2d(dim)
        # 深度卷积（空间信息聚合）
        self.dw_conv = nn.Conv2d(dim, inner_dim, kernel_size=3, padding=1,
                                  groups=dim, bias=False)
        # 逐点卷积（通道混合 + 降维）
        self.pw_conv = nn.Conv2d(inner_dim, dim, kernel_size=1, bias=False)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：x: (B, C, H, W)
        返回：(B, C, H, W)（与输入 shape 相同）
        """
        residual = x
        x = self.norm(x)
        x = self.act(self.dw_conv(x))
        x = self.pw_conv(x)
        return x + residual  # 残差连接，保护梯度流动


# ==============================================================================
#   核心模块：条件化 Mamba 块（CMB）
# ==============================================================================

class HV_CMB(nn.Module):
    """
    HV 色度流条件化 Mamba 块（Conditional Mamba Block for HV stream）

    替换 HV_LCA，用 Mamba SSM 替代交叉注意力，加入 FiLM 退化感知调制。
    使用 2 方向扫描（水平 + 垂直），适合色度信息的局部连续特性。

    前向流程：
        1. x = x + SS2D(LN(x), num_scan=2)    # Mamba 全局建模（2方向扫描）
        2. x = LocalConvEnhancer(x)             # 局部增强（补偿局部遗忘）
        3. x = FiLMLayer(x, d)                  # FiLM 退化感知调制
        4. x = x + IEL(LN(x))                   # 强度增强（保留原 HV_LCA 设计）

    参数：
        dim:       特征通道数
        cond_dim:  退化条件向量维度（来自 DAM，默认4）
        d_state:   Mamba SSM 状态维度
        d_conv:    Mamba 内部卷积核大小
        expand:    Mamba 通道扩展倍数
        bias:      是否使用偏置（保持与原 LCA 接口一致）
    
    接口对比：
        原 HV_LCA.forward(x, y) → 需要两个输入（x 和对方流的特征 y）
        新 HV_CMB.forward(x, d) → x 是当前特征，d 是退化条件向量
        
        注意：Mamba 取消了跨流的交叉注意力（CAB）。两流的信息交互
        将通过 FiLM 条件调制隐式实现（两流共享同一退化条件向量 d）。
    """

    def __init__(self, dim: int, cond_dim: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, bias: bool = False):
        super().__init__()

        # 层归一化（channels_first 格式，与原 LCA 保持一致）
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # 2方向 SS2D（HV 色度流使用水平+垂直）
        self.ss2d = SS2D(dim=dim, d_state=d_state, d_conv=d_conv,
                         expand=expand, num_scan=2)

        # 局部卷积增强（解决 Mamba 的局部像素遗忘）
        self.local_enhance = LocalConvEnhancer(dim=dim, expand=2)

        # FiLM 条件调制层（核心创新：退化感知）
        self.film = FiLMLayer(channels=dim, cond_dim=cond_dim)

        # 强度增强层（保留原 HV_LCA 的 IEL 设计）
        self.iel = IEL(dim=dim)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
            x: HV 色度流特征图 (B, C, H, W)
            d: 退化条件向量 (B, cond_dim)，来自 DAM 模块
        
        返回：
            经过 Mamba + FiLM 调制后的特征图 (B, C, H, W)
        
        与原 HV_LCA 的对比：
            原：x = x + CAB(LN(x), LN(y))  # 需要另一路特征 y
            新：x = x + SS2D(LN(x))         # 自注意力式全局建模
        """
        # 步骤1：Mamba 全局建模（2方向扫描，含残差）
        x = x + self.ss2d(self.norm1(x))

        # 步骤2：局部卷积增强（弥补 Mamba 对近邻像素的不足）
        x = self.local_enhance(x)

        # 步骤3：FiLM 退化感知条件调制（γ⊙x + β）
        x = self.film(x, d)

        # 步骤4：强度增强（保留原 IEL 设计）
        x = x + self.iel(self.norm2(x))

        return x


class I_CMB(nn.Module):
    """
    I 光照流条件化 Mamba 块（Conditional Mamba Block for I stream）

    替换 I_LCA，使用 4 方向全扫描，适合光照信息的全局连续特性。
    与 HV_CMB 的唯一区别：SS2D 使用 4 方向（而非 2 方向）扫描。

    前向流程：
        1. x = x + SS2D(LN(x), num_scan=4)    # Mamba 全局建模（4方向扫描）
        2. x = LocalConvEnhancer(x)             # 局部增强
        3. x = FiLMLayer(x, d)                  # FiLM 退化感知调制
        4. x = x + IEL(LN(x))                   # 强度增强（含残差，与原 I_LCA 一致）

    参数：
        dim:       特征通道数
        cond_dim:  退化条件向量维度（默认4）
        d_state:   Mamba SSM 状态维度
        d_conv:    Mamba 内部卷积核大小
        expand:    Mamba 通道扩展倍数
        bias:      是否使用偏置
    """

    def __init__(self, dim: int, cond_dim: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, bias: bool = False):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # 4方向 SS2D（I 光照流使用完整四方向扫描）
        self.ss2d = SS2D(dim=dim, d_state=d_state, d_conv=d_conv,
                         expand=expand, num_scan=4)

        self.local_enhance = LocalConvEnhancer(dim=dim, expand=2)
        self.film = FiLMLayer(channels=dim, cond_dim=cond_dim)
        self.iel = IEL(dim=dim)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
            x: I 光照流特征图 (B, C, H, W)
            d: 退化条件向量 (B, cond_dim)
        返回：
            (B, C, H, W)
        """
        # 步骤1：4方向 Mamba 全局建模（光照流需要更强的全局感受野）
        x = x + self.ss2d(self.norm1(x))

        # 步骤2：局部卷积增强
        x = self.local_enhance(x)

        # 步骤3：FiLM 退化感知条件调制
        x = self.film(x, d)

        # 步骤4：强度增强（I 通道保留残差连接，与原 I_LCA 一致）
        x = x + self.iel(self.norm2(x))

        return x


# ==============================================================================
#                              单元测试
# ==============================================================================
if __name__ == '__main__':
    import sys
    print("=" * 60)
    print(f"CMB（条件化 Mamba 块）单元测试")
    print(f"mamba-ssm 是否可用: {MAMBA_AVAILABLE}")
    if not MAMBA_AVAILABLE:
        print("[提示] 当前使用 MockMamba（Windows 开发模式），shape 验证有效，SSM 能力待服务器验证")
    print("=" * 60)

    batch_size = 2
    cond_dim   = 4   # DAM 输出维度

    # ---- 测试1：SS2D（2方向，HV流） ----
    print("\n[测试1] SS2D（2方向扫描，HV 色度流）")
    for ch, h, w in [(36, 64, 64), (72, 32, 32), (144, 16, 16)]:
        ss2d_hv = SS2D(dim=ch, num_scan=2)
        ss2d_hv.eval()
        x = torch.rand(batch_size, ch, h, w)
        with torch.no_grad():
            out = ss2d_hv(x)
        assert out.shape == x.shape, f"shape 错误: {out.shape}"
        print(f"  dim={ch:3d}, {h}×{w}: 输入 {x.shape} → 输出 {out.shape}  [OK]")

    # ---- 测试2：SS2D（4方向，I流） ----
    print("\n[测试2] SS2D（4方向扫描，I 光照流）")
    for ch, h, w in [(36, 64, 64), (72, 32, 32), (144, 16, 16)]:
        ss2d_i = SS2D(dim=ch, num_scan=4)
        ss2d_i.eval()
        x = torch.rand(batch_size, ch, h, w)
        with torch.no_grad():
            out = ss2d_i(x)
        assert out.shape == x.shape, f"shape 错误: {out.shape}"
        print(f"  dim={ch:3d}, {h}×{w}: 输入 {x.shape} → 输出 {out.shape}  [OK]")

    # ---- 测试3：LocalConvEnhancer ----
    print("\n[测试3] LocalConvEnhancer")
    lce = LocalConvEnhancer(dim=36)
    lce.eval()
    x = torch.rand(2, 36, 64, 64)
    with torch.no_grad():
        out = lce(x)
    assert out.shape == x.shape
    print(f"  输入 {x.shape} → 输出 {out.shape}  [OK]")

    # ---- 测试4：HV_CMB 完整测试 ----
    print("\n[测试4] HV_CMB（2方向，HV 色度流）")
    test_configs = [
        (36,  64, 64),   # 最浅层
        (72,  32, 32),   # 中间层
        (144, 16, 16),   # 最深层（瓶颈）
    ]
    for ch, h, w in test_configs:
        hv_cmb = HV_CMB(dim=ch, cond_dim=cond_dim)
        hv_cmb.eval()
        x = torch.rand(batch_size, ch, h, w)
        d = torch.rand(batch_size, cond_dim)
        with torch.no_grad():
            out = hv_cmb(x, d)
        assert out.shape == x.shape, f"HV_CMB shape 错误: {out.shape}"
        params = sum(p.numel() for p in hv_cmb.parameters())
        print(f"  dim={ch:3d}, {h}×{w}: 输入 {x.shape} → 输出 {out.shape} | 参数量 {params:,}  [OK]")

    # ---- 测试5：I_CMB 完整测试 ----
    print("\n[测试5] I_CMB（4方向，I 光照流）")
    for ch, h, w in test_configs:
        i_cmb = I_CMB(dim=ch, cond_dim=cond_dim)
        i_cmb.eval()
        x = torch.rand(batch_size, ch, h, w)
        d = torch.rand(batch_size, cond_dim)
        with torch.no_grad():
            out = i_cmb(x, d)
        assert out.shape == x.shape, f"I_CMB shape 错误: {out.shape}"
        params = sum(p.numel() for p in i_cmb.parameters())
        print(f"  dim={ch:3d}, {h}×{w}: 输入 {x.shape} → 输出 {out.shape} | 参数量 {params:,}  [OK]")

    # ---- 测试6：参数量汇总 ----
    print("\n[测试6] HV_CMB vs I_CMB 参数量对比")
    for ch in [36, 72, 144]:
        hv = HV_CMB(dim=ch)
        i  = I_CMB(dim=ch)
        hv_p = sum(p.numel() for p in hv.parameters())
        i_p  = sum(p.numel() for p in i.parameters())
        print(f"  dim={ch:3d}: HV_CMB={hv_p:,}  |  I_CMB={i_p:,}  |  I比HV多 {i_p-hv_p:,}")

    # ---- 测试7：梯度流检验 ----
    print("\n[测试7] 梯度流检验（确保退化条件向量 d 有梯度）")
    hv_cmb = HV_CMB(dim=36)
    x_g = torch.rand(2, 36, 64, 64)
    d_g = torch.rand(2, 4, requires_grad=True)
    out_g = hv_cmb(x_g, d_g)
    out_g.sum().backward()
    assert d_g.grad is not None, "d 没有梯度！"
    print(f"  d 的梯度 L2 范数: {d_g.grad.norm().item():.4f}  [OK]")

    print("\n" + "=" * 60)
    print("所有测试通过！CMB 模块就绪。")
    if not MAMBA_AVAILABLE:
        print("[提示] 上传至服务器安装 mamba-ssm 后，MockMamba 将自动替换为真实 Mamba。")
    print("=" * 60)
