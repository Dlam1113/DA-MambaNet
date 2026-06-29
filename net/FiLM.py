"""
FiLM 生成器模块（Feature-wise Linear Modulation Generator）

功能：
    将退化感知模块（DAM）输出的4维条件向量 d ∈ R^4
    转换为用于调制特征图的缩放系数 γ 和平移系数 β。

    调制公式：FiLM(F | γ, β) = γ ⊙ F + β
    
    其中：
    - F:    待调制的特征图 (B, C, H, W)
    - γ:    通道级缩放系数 (B, C)，γ > 1 放大，0 < γ < 1 抑制，γ < 0 反转
    - β:    通道级平移系数 (B, C)，调整激活值分布中心

理论依据（来自原论文 Perez et al., AAAI 2018）：
    - γ 比 β 更重要：实验中去掉 γ（设为1）使准确率下降 1.5%，去掉 β 只下降 0.5%
    - γ 可以完全关闭不相关的特征通道（γ ≈ 0），实现特征级的"选择性遗忘"
    - FiLM 无需放在归一化层后，可灵活插入网络任意位置

在 DA-MambaNet 中的使用方式：
    d = DAM(x)                          # (B, 4)
    gamma, beta = FiLMGenerator(d)      # 各 (B, C)
    feat = Mamba(x)                     # (B, C, H, W)
    feat = gamma[:, :, None, None] * feat + beta[:, :, None, None]  # FiLM 调制

作者：DA-MambaNet 项目
"""

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """
    FiLM 生成器：从退化条件向量生成特征图调制参数 (γ, β)
    
    网络结构：
        d (B, cond_dim)
        │
        ├─ FC(cond_dim → hidden_dim) → SiLU → FC(hidden_dim → channels * 2)
        │
        └─ 分割：前 channels 维 → γ，后 channels 维 → β
    
    参数：
        channels:   待调制特征图的通道数（输出 γ 和 β 的维度各为 channels）
        cond_dim:   条件向量维度，默认4（来自 DAM 输出）
        hidden_dim: 隐层维度，默认 max(channels // 4, 16)，控制生成器复杂度
    
    注意：
        - γ 初始化为1附近（恒等映射），β 初始化为0附近（不偏移）
          确保训练初期 FiLM 接近恒等变换，不破坏预训练特征
    """

    def __init__(self, channels: int, cond_dim: int = 4, hidden_dim: int = None):
        super().__init__()

        self.channels = channels
        # 若不指定 hidden_dim，默认为 max(channels//4, 16)，兼顾表达力与参数量
        if hidden_dim is None:
            hidden_dim = max(channels // 4, 16)

        # 两层全连接网络生成 γ 和 β（输出维度 = channels * 2）
        self.generator = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),   # 条件压缩
            nn.SiLU(),                          # SiLU (Swish) 激活：平滑，避免 ReLU 死神经元
            nn.Linear(hidden_dim, channels * 2) # 输出 γ 和 β 拼接
        )

        # 初始化：让 γ ≈ 1，β ≈ 0（初始接近恒等变换）
        self._init_weights()

    def _init_weights(self):
        """
        初始化策略：
        - 最后一层（输出层）偏置初始化：
          前 channels 维（对应 γ）设为 1.0
          后 channels 维（对应 β）设为 0.0
        - 这样训练初期 FiLM 等价于恒等变换，保证梯度稳定流动
        """
        # 最后一层的偏置决定初始的 γ 和 β 值
        last_linear = self.generator[-1]  # 最后一个 Linear 层
        nn.init.zeros_(last_linear.weight)
        # bias 前半部分 → γ 初始为1，后半部分 → β 初始为0
        with torch.no_grad():
            last_linear.bias[:self.channels] = 1.0   # γ 初始化为 1
            last_linear.bias[self.channels:] = 0.0   # β 初始化为 0

    def forward(self, d: torch.Tensor):
        """
        前向传播：条件向量 → (γ, β)
        
        参数：
            d: 退化条件向量 (B, cond_dim)
        
        返回：
            gamma: 缩放系数 (B, channels)
            beta:  平移系数 (B, channels)
        
        使用示例：
            gamma, beta = film_gen(d)
            # 将 (B, C) 广播到特征图 (B, C, H, W)
            feat = gamma[:, :, None, None] * feat + beta[:, :, None, None]
        """
        # 生成器输出：(B, channels * 2)
        params = self.generator(d)
        # 沿最后一维切分为 γ 和 β
        gamma, beta = params.chunk(2, dim=-1)  # 各 (B, channels)
        return gamma, beta


class FiLMLayer(nn.Module):
    """
    FiLM 调制层（将 FiLMGenerator 和特征调制操作封装为一个整体）
    
    用法：
        film = FiLMLayer(channels=36, cond_dim=4)
        feat_out = film(feat, d)  # feat: (B,C,H,W), d: (B,4)
    
    参数：
        channels:  特征通道数
        cond_dim:  条件向量维度
        hidden_dim: FiLM 生成器隐层大小
    """

    def __init__(self, channels: int, cond_dim: int = 4, hidden_dim: int = None):
        super().__init__()
        self.film_gen = FiLMGenerator(channels, cond_dim, hidden_dim)

    def forward(self, feat: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        FiLM 条件调制：γ ⊙ F + β
        
        参数：
            feat: 输入特征图 (B, C, H, W)
            d:    退化条件向量 (B, cond_dim)
        
        返回：
            调制后的特征图 (B, C, H, W)
        """
        gamma, beta = self.film_gen(d)                # 各 (B, C)
        # 广播：(B, C) → (B, C, 1, 1)，自动广播到 (B, C, H, W)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)      # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)       # (B, C, 1, 1)
        return gamma * feat + beta                     # 元素级仿射变换


# ==============================================================================
#                              单元测试
# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("FiLM 生成器单元测试")
    print("=" * 60)

    batch_size = 2
    channels   = 36    # HV 流第一层通道数
    cond_dim   = 4     # DAM 输出维度

    # ---- 测试1：FiLMGenerator 基本测试 ----
    print("\n[测试1] FiLMGenerator 基本测试")
    gen = FiLMGenerator(channels=channels, cond_dim=cond_dim)
    gen.eval()
    d = torch.rand(batch_size, cond_dim)
    with torch.no_grad():
        gamma, beta = gen(d)
    print(f"  条件向量 d:  {d.shape}")
    print(f"  gamma shape: {gamma.shape}")   # 期望 (2, 36)
    print(f"  beta shape:  {beta.shape}")    # 期望 (2, 36)
    assert gamma.shape == (batch_size, channels), "gamma 维度错误！"
    assert beta.shape  == (batch_size, channels), "beta 维度错误！"
    print("  [OK] 通过")

    # ---- 测试2：初始化验证（γ≈1，β≈0） ----
    print("\n[测试2] 初始化验证（训练初期应接近恒等映射）")
    # 用固定输入测试，初始化后 γ 应接近1，β 应接近0
    # 由于网络权重也有影响，完全等于1/0的情况在全连接后不成立，但偏置贡献主导时近似
    last_bias = gen.generator[-1].bias
    gamma_bias = last_bias[:channels]
    beta_bias  = last_bias[channels:]
    print(f"  γ 偏置均值: {gamma_bias.mean().item():.4f}（期望≈1.0）")
    print(f"  β 偏置均值: {beta_bias.mean().item():.4f}（期望≈0.0）")
    assert abs(gamma_bias.mean().item() - 1.0) < 0.01, "γ 偏置初始化偏差过大！"
    assert abs(beta_bias.mean().item() - 0.0) < 0.01,  "β 偏置初始化偏差过大！"
    print("  [OK] 通过")

    # ---- 测试3：FiLMLayer 端到端测试 ----
    print("\n[测试3] FiLMLayer 端到端特征调制")
    H, W = 64, 64
    film = FiLMLayer(channels=channels, cond_dim=cond_dim)
    film.eval()
    feat = torch.rand(batch_size, channels, H, W)
    d_test = torch.rand(batch_size, cond_dim)
    with torch.no_grad():
        feat_out = film(feat, d_test)
    print(f"  输入特征: {feat.shape}")
    print(f"  输出特征: {feat_out.shape}")   # 期望与输入相同
    assert feat_out.shape == feat.shape, "FiLM 调制后 shape 变化！"
    print("  [OK] 通过")

    # ---- 测试4：参数量统计 ----
    print("\n[测试4] 参数量统计（不同通道数）")
    for ch in [36, 72, 144]:
        gen_test = FiLMGenerator(channels=ch, cond_dim=4)
        params = sum(p.numel() for p in gen_test.parameters())
        print(f"  channels={ch:3d}: FiLMGenerator 参数量 = {params:,}")
    print("  [OK] 通过")

    # ---- 测试5：梯度流测试 ----
    print("\n[测试5] 梯度流测试")
    film_grad = FiLMLayer(channels=36, cond_dim=4)
    feat_g = torch.rand(2, 36, 64, 64)
    d_g = torch.rand(2, 4, requires_grad=True)
    out_g = film_grad(feat_g, d_g)
    loss = out_g.sum()
    loss.backward()
    print(f"  d 的梯度是否存在: {d_g.grad is not None}")
    assert d_g.grad is not None, "条件向量没有梯度！"
    print("  [OK] 通过")

    print("\n" + "=" * 60)
    print("所有测试通过！FiLM 模块就绪。")
    print("=" * 60)
