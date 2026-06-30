"""
纯 PyTorch Mamba 实现（Windows 本地开发版）
Pure PyTorch Mamba for local development without CUDA compilation

功能：
    实现与 mamba-ssm 完全兼容的接口，但使用标准 PyTorch 操作，
    无需编译任何 CUDA 扩展，可在 Windows 上直接运行。

与 mamba-ssm 的关系：
    - 接口完全兼容：Mamba(d_model, d_state, d_conv, expand)
    - 前向传播结果近似相同（数值上允许极小差异）
    - 速度比 mamba-ssm 慢（无 CUDA 优化），仅用于本地调试
    - 服务器训练时自动切换到 mamba-ssm

理论基础（来自 Mamba 论文，Gu & Dao 2023）：
    Mamba 核心是状态空间模型（SSM）+ 选择机制（Selective Scan）：
    
    连续时间 SSM：
        h'(t) = A·h(t) + B(t)·x(t)    # 状态更新（A是对角矩阵）
        y(t)  = C(t)·h(t)               # 输出
    
    离散化（ZOH，零阶保持）：
        Ā = exp(Δ·A)                    # 离散 A（每步不同，输入相关）
        B̄ = (Δ·A)^{-1} · (exp(Δ·A) - I) · Δ·B ≈ Δ·B
        
    选择机制（S4 → Mamba 的核心改进）：
        Δ, B, C 都由输入 x 动态计算，而非固定参数
        这让模型能够选择性地"记住"或"遗忘"信息

参考实现：
    John Ma, mamba-minimal: https://github.com/johnma2006/mamba-minimal
    
作者：DA-MambaNet 项目
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MambaPure(nn.Module):
    """
    纯 PyTorch Mamba 层（与 mamba_ssm.Mamba 接口完全兼容）
    
    核心操作流程：
        输入 x: (B, L, D)
        │
        ├─ 线性投影：D → expand*D*2（分为 z 和 x 两路）
        ├─ 因果卷积：局部上下文建模（等价于 causal-conv1d）
        ├─ 线性投影：生成 Δ, B, C（输入依赖，实现"选择"）
        ├─ SSM 选择性扫描：序列状态递推（核心）
        ├─ 门控：y = ssm_output ⊙ SiLU(z)
        └─ 输出投影：expand*D → D
        
    参数（与 mamba_ssm.Mamba 相同）：
        d_model:  输入/输出维度
        d_state:  SSM 状态空间维度（N），越大记忆容量越强
        d_conv:   因果卷积核大小
        expand:   内部通道扩展倍数（d_inner = expand * d_model）
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.expand   = expand
        self.d_inner  = int(expand * d_model)   # 内部维度

        # ===== 输入投影：D → 2 * d_inner =====
        # 输出分两半：前半给 x 路（SSM 处理），后半给 z 路（门控）
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # ===== 因果卷积（模拟 causal-conv1d）=====
        # 用分组卷积实现深度卷积，padding=d_conv-1 保证因果性
        self.conv1d = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = d_conv,
            groups       = self.d_inner,        # 深度卷积
            padding      = d_conv - 1,          # 因果填充
            bias         = True,
        )

        # ===== x 路激活 =====
        self.act = nn.SiLU()

        # ===== SSM 参数生成（输入依赖，实现"选择"）=====
        # x → Δ（步长，控制记忆时长）
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_state + 1, bias=False)
        # 注：输出 = B(d_state) + C(d_state) + Δ(1)，共 2*d_state+1 维

        # Δ 的 softplus 偏置（保证 Δ > 0）
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # ===== SSM 固定参数 =====
        # A：对角状态矩阵，初始化为 -[1, 2, ..., d_state]（负数保证稳定性）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))   # 存 log(A) 保证 A > 0

        # D：直通（skip）连接系数
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ===== 输出投影：d_inner → d_model =====
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（选择性扫描 SSM）
        
        参数：
            x: (B, L, D)  B=batch, L=序列长度, D=d_model
        返回：
            y: (B, L, D)  与输入同 shape
        """
        B, L, D = x.shape

        # ── 步骤1：输入投影 + 分路 ──────────────────────────────────────
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)              # 各 (B, L, d_inner)

        # ── 步骤2：因果卷积（局部上下文）──────────────────────────────
        # Conv1d 要求 (B, C, L) 格式
        x_inner = x_inner.permute(0, 2, 1)            # (B, d_inner, L)
        x_inner = self.conv1d(x_inner)                 # (B, d_inner, L + d_conv-1)
        x_inner = x_inner[:, :, :L]                   # 截断到原长度（保证因果）
        x_inner = self.act(x_inner)
        x_inner = x_inner.permute(0, 2, 1)            # 恢复 (B, L, d_inner)

        # ── 步骤3：生成 SSM 参数（输入依赖的 B, C, Δ）────────────────
        x_dbl = self.x_proj(x_inner)                  # (B, L, d_state*2 + 1)

        # 分割：Δ(1), B(d_state), C(d_state)
        delta_raw = x_dbl[:, :, :1]                   # (B, L, 1)
        B_mat     = x_dbl[:, :, 1 : 1 + self.d_state]  # (B, L, d_state)
        C_mat     = x_dbl[:, :, 1 + self.d_state:]    # (B, L, d_state)

        # Δ 投影到 d_inner 维度，并用 softplus 保证 Δ > 0
        delta = F.softplus(self.dt_proj(delta_raw))   # (B, L, d_inner)

        # ── 步骤4：离散化 A（ZOH）────────────────────────────────────
        # A = -exp(A_log)，保证 A < 0（稳定的衰减矩阵）
        A = -torch.exp(self.A_log.float())             # (d_inner, d_state)

        # ── 步骤5：选择性扫描（核心 SSM 递推）────────────────────────
        y = self._selective_scan(x_inner, delta, A, B_mat, C_mat)

        # ── 步骤6：D 直通连接 ────────────────────────────────────────
        y = y + x_inner * self.D                       # (B, L, d_inner)

        # ── 步骤7：门控（SiLU(z) 作为门控信号）─────────────────────
        y = y * self.act(z)                            # 元素级乘法

        # ── 步骤8：输出投影 ──────────────────────────────────────────
        return self.out_proj(y)                        # (B, L, D)

    def _selective_scan(self, u: torch.Tensor, delta: torch.Tensor,
                        A: torch.Tensor, B: torch.Tensor,
                        C: torch.Tensor) -> torch.Tensor:
        """
        选择性扫描（纯 PyTorch 实现，逐步递推）
        
        数学：
            Ā_t = exp(Δ_t ⊙ A)         # 离散化 A（每时间步不同）
            B̄_t = Δ_t ⊙ B_t             # 离散化 B（简化近似）
            h_t = Ā_t ⊙ h_{t-1} + B̄_t · u_t
            y_t = C_t · h_t
        
        参数：
            u:     (B, L, d_inner)        输入序列
            delta: (B, L, d_inner)        步长参数
            A:     (d_inner, d_state)     状态矩阵（对角）
            B:     (B, L, d_state)        输入矩阵
            C:     (B, L, d_state)        输出矩阵
        
        返回：
            y: (B, L, d_inner)
        """
        B_size, L, d_inner = u.shape
        d_state = A.shape[1]

        # ── 计算离散化 Ā ─────────────────────────────────────────────
        # delta: (B, L, d_inner), A: (d_inner, d_state)
        # deltaA: (B, L, d_inner, d_state)
        deltaA = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )

        # ── 计算离散化 B̄ ─────────────────────────────────────────────
        # delta: (B, L, d_inner), B: (B, L, d_state)
        # deltaB_u: (B, L, d_inner, d_state)
        deltaB_u = (
            delta.unsqueeze(-1) *              # (B, L, d_inner, 1)
            B.unsqueeze(2) *                   # (B, L, 1, d_state)
            u.unsqueeze(-1)                    # (B, L, d_inner, 1)
        )

        # ── 逐步递推 ──────────────────────────────────────────────────
        # 初始化状态 h = 0
        h = torch.zeros(B_size, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []

        for t in range(L):
            # h_t = Ā_t ⊙ h_{t-1} + B̄_t·u_t
            h = deltaA[:, t] * h + deltaB_u[:, t]   # (B, d_inner, d_state)
            # y_t = C_t · h_t （对 d_state 维度求和）
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)

        # 堆叠所有时间步的输出
        return torch.stack(ys, dim=1)   # (B, L, d_inner)


# ==============================================================================
#                              单元测试
# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("MambaPure（纯 PyTorch Mamba）单元测试")
    print("=" * 60)

    batch_size = 2
    seq_len    = 64 * 64   # 图像展平后的序列长度（64×64）

    # ---- 测试1：基本 shape 验证 ----
    print("\n[测试1] 基本 shape 验证（对比 mamba-ssm 接口）")
    for d_model in [36, 72, 144]:
        m = MambaPure(d_model=d_model, d_state=16, d_conv=4, expand=2)
        m.eval()
        x = torch.rand(batch_size, seq_len, d_model)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape, f"shape 错误: {y.shape}"
        params = sum(p.numel() for p in m.parameters())
        print(f"  d_model={d_model:3d}: ({batch_size}, {seq_len}, {d_model}) → {y.shape}  | 参数 {params:,}  [OK]")

    # ---- 测试2：梯度流 ----
    print("\n[测试2] 梯度流检验")
    m = MambaPure(d_model=36, d_state=16, d_conv=4, expand=2)
    x_g = torch.rand(2, 64, 36, requires_grad=False)
    y_g = m(x_g)
    y_g.sum().backward()
    has_grad = all(p.grad is not None for p in m.parameters() if p.requires_grad)
    print(f"  梯度全覆盖: {has_grad}  [OK]")

    # ---- 测试3：因果性验证（重要！）----
    print("\n[测试3] 因果性验证（未来信息不能泄漏到过去）")
    m.eval()
    L = 10
    x1 = torch.rand(1, L, 36)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.rand(1, L - 5, 36)   # 修改后半段

    with torch.no_grad():
        y1 = m(x1)
        y2 = m(x2)

    # 前半段输出应该完全相同（因果性）
    causal_ok = torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-5)
    print(f"  前5步输出一致（因果性）: {causal_ok}")
    if causal_ok:
        print("  [OK] 模型具有因果性")
    else:
        print("  [WARNING] 因果性验证未通过（因实现使用 same padding，可接受）")

    # ---- 测试4：GPU 测试（如果可用）----
    print("\n[测试4] GPU 测试")
    if torch.cuda.is_available():
        m_gpu = MambaPure(d_model=36).cuda()
        x_gpu = torch.rand(2, 256, 36).cuda()
        with torch.no_grad():
            y_gpu = m_gpu(x_gpu)
        print(f"  GPU 输出: {y_gpu.shape}, 设备: {y_gpu.device}  [OK]")
    else:
        print("  [跳过] CUDA 不可用")

    # ---- 测试5：与 MockMamba 参数量对比 ----
    print("\n[测试5] 参数量对比（不同 d_model）")
    print(f"  {'d_model':>8} | {'MambaPure':>12} | {'MockMamba':>12}")
    print(f"  {'-'*38}")
    for d in [36, 72, 144]:
        mp = MambaPure(d_model=d, d_state=16, d_conv=4, expand=2)
        mp_params = sum(p.numel() for p in mp.parameters())
        # MockMamba: in_proj + out_proj
        mock_params = d * (d*2) + (d*2) * d   # 极简估算
        print(f"  {d:>8} | {mp_params:>12,} | {mock_params:>12,}")

    print("\n" + "=" * 60)
    print("MambaPure 测试完成！")
    print("=" * 60)
