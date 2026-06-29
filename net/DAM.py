"""
退化感知模块（Degradation-Aware Module, DAM）

功能：
    从退化图像中自动预测：
    1. 退化类型：[p_rain, p_fog, p_lowlight]（Softmax 概率，3分类）
    2. 退化程度：s_level ∈ [0, 1]（Sigmoid 连续估计）

    输出的4维条件向量 d = [p_rain, p_fog, p_lowlight, s_level] 供 FiLM 生成器使用。

设计原则：
    - 极轻量：目标参数量 < 0.05M
    - 使用深度可分离卷积降低参数
    - 多尺度感受野：3层卷积逐步扩大感受野（kernel: 3, 5, 3）
    - 全局平均池化聚合全局退化信息

在 DA-MambaNet 中的位置：
    输入图像 x → DAM → 条件向量 d → FiLM 生成器 → (γ, β) → 调制 Mamba 输出

作者：DA-MambaNet 项目
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积：深度卷积 + 逐点卷积
    
    相比标准卷积减少约 8~9 倍参数，保持相似表达能力。
    
    参数：
        in_ch:  输入通道数
        out_ch: 输出通道数
        kernel: 卷积核大小（仅深度卷积使用，逐点卷积固定为1）
        stride: 步长
        padding: 填充
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        # 深度卷积：每个通道独立处理空间信息（groups=in_ch）
        self.depthwise = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=kernel, stride=stride, padding=padding,
            groups=in_ch, bias=False  # groups=in_ch 即为深度卷积
        )
        # 逐点卷积：1×1 跨通道混合信息
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        # 批归一化 + ReLU 激活
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：深度卷积 → 逐点卷积 → BN → ReLU
        
        参数：
            x: 输入特征图 (B, in_ch, H, W)
        返回：
            输出特征图 (B, out_ch, H/stride, W/stride)
        """
        x = self.depthwise(x)    # 空间信息混合（通道独立）
        x = self.pointwise(x)    # 跨通道信息混合
        x = self.bn(x)           # 批归一化，稳定训练
        return self.act(x)       # ReLU 激活


class DegradationAwareModule(nn.Module):
    """
    退化感知模块（DAM）
    
    网络结构：
        输入 (B, 3, H, W)
        │
        ├─ DWSConv(3→16, k=3, stride=2)   → (B, 16, H/2, W/2)  # 初步提取
        ├─ DWSConv(16→32, k=5, stride=2)  → (B, 32, H/4, W/4)  # 扩大感受野
        ├─ DWSConv(32→64, k=3, stride=2)  → (B, 64, H/8, W/8)  # 深层特征
        │
        └─ GlobalAvgPool → (B, 64)         # 聚合全局退化信息
           │
           ├─ 分类头：FC(64→32) → ReLU → FC(32→num_classes) → Softmax
           │  → [p_rain, p_fog, p_lowlight]（退化类型概率）
           │
           └─ 程度估计头：FC(64→32) → ReLU → FC(32→1) → Sigmoid
              → s_level ∈ [0,1]（退化严重程度）
    
    最终输出拼接：[p_rain, p_fog, p_lowlight, s_level] ∈ R^(num_classes + 1)
    
    参数：
        num_classes: 退化类型数量，默认3（雨/雾/低光）
        mid_ch:      中间层通道数，默认32（减小可进一步轻量化）
    
    注意事项：
        - 当 num_classes=1（单任务场景，如只做低光）时，分类头退化为单类概率，
          此时条件向量为 [p_lowlight, s_level]，FiLM 生成器输入维度需对应修改
        - 训练时可选加入交叉熵损失做监督，也可纯靠下游任务损失隐式引导
    """

    def __init__(self, num_classes: int = 3, mid_ch: int = 32):
        super().__init__()

        self.num_classes = num_classes

        # ===== 特征提取主干（三层深度可分离卷积） =====
        # 每次步长为2，进行空间降采样，快速缩小特征图至 H/8 × W/8
        self.backbone = nn.Sequential(
            # 层1：浅层退化线索（亮度变化、颜色偏差）
            DepthwiseSeparableConv(3, 16, kernel=3, stride=2, padding=1),
            # 层2：中层退化模式（纹理、雾气散射分布）
            DepthwiseSeparableConv(16, 32, kernel=5, stride=2, padding=2),  # k=5 感受野更大
            # 层3：深层退化语义（整体光照水平、退化类型判断）
            DepthwiseSeparableConv(32, 64, kernel=3, stride=2, padding=1),
        )

        # ===== 全局聚合 =====
        # 将任意分辨率的特征图压缩为固定长度的 64 维向量
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 输出 (B, 64, 1, 1)

        # ===== 退化类型分类头 =====
        # 输出：各退化类型的概率分布（经过 Softmax 归一化）
        self.cls_head = nn.Sequential(
            nn.Linear(64, mid_ch),       # 64 → 32
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, num_classes),  # 32 → num_classes
            # 注意：Softmax 在 forward 中动态调用（推理）或由 CrossEntropyLoss 外部处理（训练）
        )

        # ===== 退化程度估计头 =====
        # 输出：0~1 的连续值，0=无退化，1=严重退化
        self.severity_head = nn.Sequential(
            nn.Linear(64, mid_ch),   # 64 → 32
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, 1),    # 32 → 1
            nn.Sigmoid(),            # 输出压缩到 [0, 1]
        )

        # ===== 权重初始化 =====
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化策略：
        - 卷积层：Kaiming 正态初始化（适合 ReLU 激活）
        - 线性层：Xavier 均匀初始化（适合通用全连接）
        - 偏置：全零初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：退化图像 → 条件向量
        
        参数：
            x: 退化输入图像 (B, 3, H, W)，数值范围 [0, 1]
        
        返回：
            d: 退化条件向量 (B, num_classes + 1)
               - d[:, :num_classes]：退化类型概率（Softmax，和为1）
               - d[:, -1:]：退化程度估计（Sigmoid，范围[0,1]）
        
        示例（num_classes=3）：
            d = [0.8, 0.1, 0.1, 0.7]
                 雨↑   雾   低光  严重程度0.7
        """
        # 主干特征提取（B, 3, H, W → B, 64, H/8, W/8）
        feat = self.backbone(x)

        # 全局平均池化聚合 → (B, 64, 1, 1) → flatten → (B, 64)
        feat = self.global_pool(feat)
        feat = feat.flatten(1)  # 等价于 feat.view(B, -1)

        # 分类头：预测退化类型概率分布
        logits = self.cls_head(feat)                        # (B, num_classes)
        cls_probs = F.softmax(logits, dim=-1)              # 归一化为概率（推理时用）

        # 程度估计头：预测退化严重程度
        severity = self.severity_head(feat)                 # (B, 1)，已经过 Sigmoid

        # 拼接为最终条件向量
        d = torch.cat([cls_probs, severity], dim=-1)        # (B, num_classes + 1)

        return d

    def get_logits(self, x: torch.Tensor):
        """
        获取未经 Softmax 的分类 logits（训练时用于 CrossEntropyLoss 计算）
        
        CrossEntropyLoss 内部会做 LogSoftmax，因此训练时不能传入已 Softmax 的概率。
        
        参数：
            x: 退化输入图像 (B, 3, H, W)
        返回：
            logits: 未归一化分类分数 (B, num_classes)
            severity: 程度估计 (B, 1)
        """
        feat = self.backbone(x)
        feat = self.global_pool(feat).flatten(1)
        logits = self.cls_head(feat)       # 未经 Softmax 的原始分数
        severity = self.severity_head(feat)
        return logits, severity


# ==============================================================================
#                              单元测试
# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("DAM（退化感知模块）单元测试")
    print("=" * 60)

    # ---- 测试参数 ----
    batch_size = 2
    H, W = 256, 256
    num_classes = 3  # 雨 / 雾 / 低光

    # ---- 构造模型 ----
    model = DegradationAwareModule(num_classes=num_classes, mid_ch=32)
    model.eval()

    # ---- 测试1：基本 forward（推理阶段） ----
    print("\n[测试1] 基本前向传播（推理模式）")
    x = torch.rand(batch_size, 3, H, W)      # 模拟退化图像
    with torch.no_grad():
        d = model(x)
    print(f"  输入 shape:  {x.shape}")
    print(f"  输出 shape:  {d.shape}")        # 期望 (2, 4)
    print(f"  d[:, :3] 类型概率（和）: {d[:, :3].sum(dim=-1)}")  # 期望近似 tensor([1., 1.])
    print(f"  d[:, 3:] 程度范围: min={d[:, -1].min():.4f}, max={d[:, -1].max():.4f}")
    assert d.shape == (batch_size, num_classes + 1), "输出维度错误！"
    assert torch.allclose(d[:, :num_classes].sum(dim=-1), torch.ones(batch_size), atol=1e-5), "概率和不为1！"
    assert (d[:, -1] >= 0).all() and (d[:, -1] <= 1).all(), "程度估计超出[0,1]范围！"
    print("  [OK] 通过")

    # ---- 测试2：get_logits（训练阶段） ----
    print("\n[测试2] get_logits（训练阶段，用于 CrossEntropyLoss）")
    model.train()
    logits, severity = model.get_logits(x)
    print(f"  logits shape:   {logits.shape}")    # 期望 (2, 3)
    print(f"  severity shape: {severity.shape}")  # 期望 (2, 1)
    assert logits.shape == (batch_size, num_classes)
    assert severity.shape == (batch_size, 1)
    print("  [OK] 通过")

    # ---- 测试3：参数量统计 ----
    print("\n[测试3] 参数量统计")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量：    {total_params:,}（{total_params/1e6:.4f}M）")
    print(f"  可训练参数：  {trainable_params:,}")
    if total_params < 50_000:
        print("  [OK] 参数量 < 0.05M，满足轻量化要求")
    else:
        print(f"  [WARNING] 参数量 {total_params/1e6:.3f}M 超出 0.05M 目标，考虑减小 mid_ch")

    # ---- 测试4：不同输入分辨率 ----
    print("\n[测试4] 不同输入分辨率（自适应池化测试）")
    for h, w in [(128, 128), (256, 256), (400, 600)]:
        x_test = torch.rand(1, 3, h, w)
        with torch.no_grad():
            d_test = model(x_test)
        print(f"  输入 {h}x{w} → 输出 {d_test.shape}  [OK]")

    # ---- 测试5：梯度流检验 ----
    print("\n[测试5] 梯度流检验")
    model.train()
    x_grad = torch.rand(2, 3, 256, 256, requires_grad=False)
    d_grad = model(x_grad)
    loss = d_grad.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  所有参数都有梯度: {has_grad}")
    assert has_grad, "部分参数没有梯度！"
    print("  [OK] 通过")

    print("\n" + "=" * 60)
    print("所有测试通过！DAM 模块就绪。")
    print("=" * 60)
