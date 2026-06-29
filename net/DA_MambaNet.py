"""
DA-MambaNet：退化感知自适应 Mamba 图像恢复网络
（Degradation-Aware Mamba Restoration Network）

论文项目：基于 DualSpaceCIDNet 的扩展研究（DA-MambaNet）

核心架构（与 DualSpaceCIDNet 的关键区别）：
┌────────────────────────────────────────────────────────────┐
│                   DualSpaceCIDNet                          │
│  LCA 交叉注意力：x 通过 CAB 关注 y（跨流注意力）            │
│  无退化感知模块                                             │
│  标准 CAB（所有退化类型共用相同的权重）                      │
└────────────────────────────────────────────────────────────┘
                           ↓  改进为
┌────────────────────────────────────────────────────────────┐
│                    DA-MambaNet                              │
│  CMB Mamba 块：SS2D 全局建模 + FiLM 退化感知调制            │
│  DAM：自动预测退化类型（雨/雾/低光）+ 严重程度               │
│  异构扫描：HV 色度流 2方向 / I 光照流 4方向                  │
└────────────────────────────────────────────────────────────┘

三点创新：
    1. 退化感知条件机制（DAM + FiLM）：全自动，无需人工 Prompt
    2. 条件化异构双流 Mamba：基于 HVI 物理特性的差异化扫描
    3. 轻量化设计：目标参数量 < 4M（与 DualSpaceCIDNet 持平）

整体网络结构（forward 流程）：
    输入 x (B, 3, H, W)
    │
    ├─ DAM(x) → 条件向量 d (B, 4) = [p_rain, p_fog, p_lowlight, s_level]
    │
    └─ HVI 色彩空间变换 → (B, 3, H, W)
       │
       ├─ HV 色度流（2通道）  ─────────────────────────────────────┐
       │   HVE_block0→1→2→3 (下采样编码器)                         │
       │   ← HV_CMB 与 I_CMB 替代原来的 LCA，共6对 →              │
       │   HVD_block3→2→1→0 (上采样解码器)                         │
       │                                                           │
       ├─ I 光照流（1通道）   ───────────────────────────────────┘  │
       │   IE_block0→1→2→3 (编码器)                                 │
       │   ID_block3→2→1→0 (解码器)                                 │
       │
       └─ 拼接 HV+I → HVI→RGB 变换 → clamp[0,1] → 输出

文件：net/DA_MambaNet.py
作者：DA-MambaNet 项目
"""

import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import NormDownsample, NormUpsample
from net.CMB import HV_CMB, I_CMB
from net.DAM import DegradationAwareModule


class DA_MambaNet(nn.Module):
    """
    退化感知自适应 Mamba 图像恢复网络（DA-MambaNet）

    网络设计在 DualSpaceCIDNet 基础上做出以下改动：
    - ❌ 移除 LCA（HV_LCA / I_LCA）模块
    - ✅ 替换为 HV_CMB / I_CMB（Mamba + FiLM 调制）
    - ✅ 新增 DAM（退化感知模块）
    - ✅ 保持所有编解码器结构、HVI 变换、跳跃连接不变

    参数说明：
        channels:     各层通道数 [ch1, ch2, ch3, ch4]
                      默认 [36, 36, 72, 144]，与 DualSpaceCIDNet 一致
        norm:         是否在编解码器中使用 LayerNorm，默认 False
        num_classes:  DAM 退化类型分类数，默认 3（雨/雾/低光）
        cond_dim:     退化条件向量维度 = num_classes + 1（+1 为程度估计）
        d_state:      Mamba SSM 状态空间维度，默认 16
        d_conv:       Mamba 内部卷积核大小，默认 4
        expand:       Mamba 通道扩展倍数，默认 2
    """

    def __init__(self,
                 channels: list = None,
                 norm: bool = False,
                 num_classes: int = 3,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2):
        super().__init__()

        if channels is None:
            channels = [36, 36, 72, 144]

        [ch1, ch2, ch3, ch4] = channels
        # 退化条件向量维度 = 分类数 + 1（程度估计）
        cond_dim = num_classes + 1

        # =====================================================================
        #   模块1：退化感知模块（DAM）
        # =====================================================================
        self.dam = DegradationAwareModule(num_classes=num_classes, mid_ch=32)

        # =====================================================================
        #   模块2：HVI 色彩空间变换（不修改，直接复用）
        # =====================================================================
        self.trans = RGB_HVI()

        # =====================================================================
        #   模块3：HV 色度流编解码器（与 DualSpaceCIDNet 完全相同）
        # =====================================================================
        # 编码器（逐步下采样，通道数增加）
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        # 解码器（逐步上采样，通道数减少）
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # =====================================================================
        #   模块4：I 光照流编解码器（与 DualSpaceCIDNet 完全相同）
        # =====================================================================
        # 编码器
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        # 解码器
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        # =====================================================================
        #   模块5：HV-I 交互模块
        #   【核心改动】原 LCA（注意力）→ 新 CMB（Mamba + FiLM）
        #
        #   HV_CMB：2方向扫描（水平+垂直），适合色度流
        #    I_CMB：4方向扫描（完整四方向），适合光照流
        #
        #   6对位置（与 DualSpaceCIDNet 的 LCA 数量相同）：
        #   编码器3对 + 解码器3对
        # =====================================================================
        # --- 编码阶段（特征提取，从浅到深） ---
        # 第1对：ch2 尺度（1/2 分辨率）
        self.HV_CMB1 = HV_CMB(dim=ch2, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.I_CMB1  = I_CMB( dim=ch2, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        # 第2对：ch3 尺度（1/4 分辨率）
        self.HV_CMB2 = HV_CMB(dim=ch3, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.I_CMB2  = I_CMB( dim=ch3, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        # 第3对：ch4 尺度（1/8 分辨率，瓶颈层）
        self.HV_CMB3 = HV_CMB(dim=ch4, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.I_CMB3  = I_CMB( dim=ch4, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # --- 解码阶段（特征恢复，从深到浅）---
        # 第4对：ch4 尺度（瓶颈层底部）
        self.HV_CMB4 = HV_CMB(dim=ch4, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.I_CMB4  = I_CMB( dim=ch4, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        # 第5对：ch3 尺度（1/4 分辨率）
        self.HV_CMB5 = HV_CMB(dim=ch3, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.I_CMB5  = I_CMB( dim=ch3, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        # 第6对：ch2 尺度（1/2 分辨率）
        self.HV_CMB6 = HV_CMB(dim=ch2, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.I_CMB6  = I_CMB( dim=ch2, cond_dim=cond_dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        参数：
            x: 退化输入图像 (B, 3, H, W)，范围 [0, 1]

        返回：
            output_rgb: 增强后的图像 (B, 3, H, W)，范围 [0, 1]
            d:          退化条件向量 (B, num_classes+1)（训练时可用于计算辅助损失）

        与 DualSpaceCIDNet.forward 的区别：
            - 额外返回条件向量 d（用于计算 DAM 分类损失）
            - 将 LCA 调用替换为 CMB 调用（接口从 (x, y) 变为 (x, d)）
        """
        dtypes = x.dtype

        # ==============================================================
        # 步骤0：退化感知 → 生成条件向量 d
        # ==============================================================
        # DAM 从退化图像中提取类型概率和严重程度
        # d: (B, num_classes+1) = [p_rain, p_fog, p_lowlight, s_level]
        d = self.dam(x)

        # ==============================================================
        # 步骤1：RGB → HVI 色彩空间变换
        # ==============================================================
        hvi = self.trans.HVIT(x)                             # (B, 3, H, W)
        i   = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)       # I 通道 (B, 1, H, W)

        # ==============================================================
        # 步骤2：编码器（双流下采样）
        # ==============================================================
        # --- I 流编码器（光照） ---
        i_enc0 = self.IE_block0(i)               # (B, ch1, H,   W)
        i_enc1 = self.IE_block1(i_enc0)          # (B, ch2, H/2, W/2)

        # --- HV 流编码器（色度） ---
        hv_0   = self.HVE_block0(hvi)            # (B, ch1, H,   W)
        hv_1   = self.HVE_block1(hv_0)           # (B, ch2, H/2, W/2)

        # 保存跳跃连接（最浅层，直接跳过所有 CMB）
        i_jump0  = i_enc0    # (B, ch1, H,   W)
        hv_jump0 = hv_0      # (B, ch1, H,   W)

        # ==============================================================
        # 第1对 CMB（ch2 尺度，1/2 分辨率）
        # ==============================================================
        # CMB 接口：forward(x, d) → 每个流独立处理，通过 d 隐式联系
        # 注意：这里不再有"y"（对方流）的显式输入，两流通过共享 d 实现耦合
        i_enc2 = self.I_CMB1(i_enc1, d)         # I 流 Mamba 建模
        hv_2   = self.HV_CMB1(hv_1,   d)        # HV 流 Mamba 建模

        # 保存跳跃连接（第1层 CMB 后）
        i_jump1  = i_enc2    # (B, ch2, H/2, W/2)
        hv_jump1 = hv_2      # (B, ch2, H/2, W/2)

        # 继续下采样到 ch3 尺度
        i_enc2 = self.IE_block2(i_enc2)          # (B, ch3, H/4, W/4)
        hv_2   = self.HVE_block2(hv_2)           # (B, ch3, H/4, W/4)

        # ==============================================================
        # 第2对 CMB（ch3 尺度，1/4 分辨率）
        # ==============================================================
        i_enc3 = self.I_CMB2(i_enc2, d)
        hv_3   = self.HV_CMB2(hv_2,   d)

        # 保存跳跃连接（第2层 CMB 后）
        i_jump2  = i_enc3    # (B, ch3, H/4, W/4)
        hv_jump2 = hv_3      # (B, ch3, H/4, W/4)

        # 继续下采样到 ch4 尺度（瓶颈）
        # 注意：这里沿用 DualSpaceCIDNet 的原始逻辑，下采样输入是 i_enc2/hv_2（而非 CMB 输出）
        # 这是原代码的设计，跳跃连接和下采样输入相互独立
        i_enc3 = self.IE_block3(i_enc2)          # (B, ch4, H/8, W/8)
        hv_3   = self.HVE_block3(hv_2)           # (B, ch4, H/8, W/8)

        # ==============================================================
        # 第3、4对 CMB（ch4 尺度，瓶颈层，最重要的两对）
        # ==============================================================
        # 第3对：编码器瓶颈底部（特征最抽象，全局退化信息最丰富）
        i_enc4 = self.I_CMB3( i_enc3, d)
        hv_4   = self.HV_CMB3(hv_3,   d)

        # 第4对：解码器瓶颈顶部（开始信息恢复）
        i_dec4 = self.I_CMB4( i_enc4, d)
        hv_4   = self.HV_CMB4(hv_4,   d)

        # ==============================================================
        # 步骤3：解码器（双流上采样）
        # ==============================================================
        # 上采样到 ch3 尺度（融合跳跃连接）
        hv_3   = self.HVD_block3(hv_4,   hv_jump2)   # (B, ch3, H/4, W/4)
        i_dec3 = self.ID_block3( i_dec4, i_jump2)    # (B, ch3, H/4, W/4)

        # ==============================================================
        # 第5对 CMB（ch3 尺度，1/4 分辨率，解码阶段）
        # ==============================================================
        i_after_cmb5  = self.I_CMB5( i_dec3, d)     # (B, ch3, H/4, W/4)  I流 CMB5 输出
        hv_after_cmb5 = self.HV_CMB5(hv_3,   d)     # (B, ch3, H/4, W/4)  HV流 CMB5 输出

        # 上采样到 ch2 尺度：两流都用各自 CMB5 的输出 + 跳跃连接
        hv_2   = self.HVD_block2(hv_after_cmb5, hv_jump1)  # (B, ch2, H/2, W/2)
        i_dec2 = self.ID_block2( i_after_cmb5,  i_jump1)   # (B, ch2, H/2, W/2)

        # ==============================================================
        # 第6对 CMB（ch2 尺度，1/2 分辨率，最浅的解码 CMB）
        # ==============================================================
        i_dec1 = self.I_CMB6( i_dec2, d)   # (B, ch2, H/2, W/2)
        hv_1   = self.HV_CMB6(hv_2,   d)   # (B, ch2, H/2, W/2)

        # 上采样到原始分辨率（融合最浅层跳跃连接）
        i_dec1 = self.ID_block1(i_dec1, i_jump0)     # (B, ch1, H, W)
        i_dec0 = self.ID_block0(i_dec1)               # (B, 1,   H, W)



        hv_1   = self.HVD_block1(hv_1, hv_jump0)     # (B, ch1, H, W)
        hv_0   = self.HVD_block0(hv_1)                # (B, 2,   H, W)

        # ==============================================================
        # 步骤4：HVI → RGB
        # ==============================================================
        # 全局残差连接：在 HVI 空间直接相加（继承 DualSpaceCIDNet 设计）
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi   # (B, 3, H, W)
        output_rgb = self.trans.PHVIT(output_hvi)              # (B, 3, H, W)

        # 截断到 [0, 1]，防止极端值导致损失计算异常
        output_rgb = torch.clamp(output_rgb, 0, 1)

        # 返回：RGB 输出 + 条件向量（训练时用于辅助损失）
        return output_rgb, d

    def HVIT(self, x: torch.Tensor) -> torch.Tensor:
        """获取 HVI 变换结果（兼容评估脚本接口）"""
        return self.trans.HVIT(x)

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理专用接口（只返回 RGB 输出，不返回条件向量）
        用于 eval.py 等推理脚本，与 DualSpaceCIDNet 接口完全兼容。
        """
        output_rgb, _ = self.forward(x)
        return output_rgb


# ==============================================================================
#                              单元测试
# ==============================================================================
if __name__ == '__main__':
    print("=" * 65)
    print("DA-MambaNet 完整网络单元测试")
    print("=" * 65)

    batch_size  = 2
    H, W        = 256, 256
    num_classes = 3

    # ---- 测试1：基本前向传播 ----
    print("\n[测试1] 基本前向传播")
    model = DA_MambaNet(
        channels    = [36, 36, 72, 144],
        num_classes = num_classes,
        d_state     = 16,
        d_conv      = 4,
        expand      = 2
    )
    model.eval()

    x = torch.rand(batch_size, 3, H, W)
    with torch.no_grad():
        output_rgb, d = model(x)

    print(f"  输入:       {x.shape}")
    print(f"  输出 RGB:   {output_rgb.shape}")            # 期望 (2, 3, 256, 256)
    print(f"  条件向量 d: {d.shape}")                     # 期望 (2, 4)
    print(f"  输出范围:   [{output_rgb.min():.4f}, {output_rgb.max():.4f}]")  # 期望 [0, 1]

    assert output_rgb.shape == x.shape, "输出 shape 错误！"
    assert d.shape == (batch_size, num_classes + 1), "条件向量 shape 错误！"
    assert output_rgb.min() >= 0 and output_rgb.max() <= 1, "输出超出 [0,1] 范围！"
    print("  [OK] 通过")

    # ---- 测试2：推理接口测试 ----
    print("\n[测试2] 推理接口（forward_inference，与 DualSpaceCIDNet 兼容）")
    with torch.no_grad():
        out_infer = model.forward_inference(x)
    assert out_infer.shape == x.shape
    print(f"  输出 shape: {out_infer.shape}  [OK]")

    # ---- 测试3：总参数量统计 ----
    print("\n[测试3] 参数量统计（目标 < 4M）")
    total_params = sum(p.numel() for p in model.parameters())
    dam_params   = sum(p.numel() for p in model.dam.parameters())
    trans_params = sum(p.numel() for p in model.trans.parameters())
    # CMB 参数（所有 12 个 CMB 块）
    cmb_params   = sum(
        sum(p.numel() for p in getattr(model, f'HV_CMB{i}').parameters()) +
        sum(p.numel() for p in getattr(model, f'I_CMB{i}').parameters())
        for i in range(1, 7)
    )
    # 编解码器参数
    enc_dec_params = total_params - dam_params - trans_params - cmb_params

    print(f"  {'组件':<20} {'参数量':>12} {'占比':>8}")
    print(f"  {'-'*42}")
    print(f"  {'DAM（退化感知）':<20} {dam_params:>12,} {dam_params/total_params*100:>7.1f}%")
    print(f"  {'12个CMB块':<20} {cmb_params:>12,} {cmb_params/total_params*100:>7.1f}%")
    print(f"  {'编解码器':<20} {enc_dec_params:>12,} {enc_dec_params/total_params*100:>7.1f}%")
    print(f"  {'HVI变换':<20} {trans_params:>12,} {trans_params/total_params*100:>7.1f}%")
    print(f"  {'-'*42}")
    print(f"  {'总计':<20} {total_params:>12,} ({total_params/1e6:.3f}M)")

    if total_params < 4_000_000:
        print(f"\n  [OK] 参数量 {total_params/1e6:.3f}M < 4M，满足轻量化目标！")
    else:
        print(f"\n  [WARNING] 参数量 {total_params/1e6:.3f}M 超出 4M 目标，需考虑压缩策略。")

    # ---- 测试4：不同输入分辨率 ----
    print("\n[测试4] 不同分辨率测试")
    for h, w in [(128, 128), (256, 256), (512, 512)]:
        x_t = torch.rand(1, 3, h, w)
        with torch.no_grad():
            out_t, d_t = model(x_t)
        assert out_t.shape == x_t.shape
        print(f"  {h}×{w}: 输入 {x_t.shape} → 输出 {out_t.shape}  [OK]")

    # ---- 测试5：梯度流检验 ----
    print("\n[测试5] 梯度流检验")
    model.train()
    x_g = torch.rand(2, 3, 128, 128)
    out_g, d_g = model(x_g)
    # 主任务损失（图像恢复）
    loss_main = out_g.sum()
    loss_main.backward()

    # 统计梯度情况（分模块报告）
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    with_grad    = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    no_grad_names = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    print(f"  可训练参数总数:    {total_params}")
    print(f"  有梯度的参数数:    {with_grad}")
    print(f"  无梯度的参数数:    {total_params - with_grad}")
    if no_grad_names:
        print(f"  无梯度参数列表（前5个）:")
        for n in no_grad_names[:5]:
            print(f"    - {n}")
    # 说明：trans.density_k 通过 PHVIT 的布尔掩码索引无法回传梯度（原 CIDNet 已知问题）
    # 主干参数（DAM + CMB + 编解码器）应全部有梯度
    core_params_ok = all(
        p.grad is not None
        for name, p in model.named_parameters()
        if p.requires_grad and 'trans' not in name
    )
    print(f"\n  主干参数（排除 HVI 变换）梯度全覆盖: {core_params_ok}")
    assert core_params_ok, "主干参数（DAM/CMB/编解码器）存在无梯度参数！"
    print("  [OK] 通过")

    print("\n" + "=" * 65)
    print("所有测试通过！DA-MambaNet 主干网络就绪。")
    print("=" * 65)

