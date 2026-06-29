#!/bin/bash
# ==============================================================================
# DA-MambaNet Ubuntu 服务器环境配置脚本
#
# 服务器实际环境（已确认）：
#   - GPU:          RTX 3090（24GB）
#   - 驱动版本:    535.309.01
#   - CUDA 版本:   12.2（nvidia-smi 显示）
#   - nvcc 状态:   未安装（需要先安装 cuda-toolkit）
#   - PyTorch 目标: torch==2.1.2+cu121（cu121 兼容 CUDA 12.x）
#
# 使用方法：
#   chmod +x setup_mamba_env.sh
#   bash setup_mamba_env.sh
#
# 注意：mamba-ssm 需要 nvcc 编译 CUDA 扩展，脚本会自动安装 cuda-toolkit
# ==============================================================================

set -e  # 任何命令失败则立即退出

echo "========================================"
echo "DA-MambaNet 环境配置脚本（CUDA 12.2）"
echo "========================================"

# ---------- 步骤0：安装 nvcc（CUDA Toolkit） ----------
echo ""
echo "[步骤0] 检查并安装 CUDA Toolkit（nvcc）..."
if command -v nvcc &> /dev/null; then
    echo "  nvcc 已安装：$(nvcc --version | grep 'release')"
else
    echo "  nvcc 未找到，开始安装 cuda-toolkit..."
    echo "  （注意：此步骤需要 sudo 权限，请确保当前用户有权限）"
    
    # 方案一：通过 apt 安装（推荐，简单快速）
    # CUDA 12.2 对应的 toolkit 版本
    sudo apt-get update -qq
    sudo apt-get install -y cuda-toolkit-12-2
    
    # 添加 nvcc 到 PATH
    export PATH=/usr/local/cuda-12.2/bin:$PATH
    echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    echo "  [OK] cuda-toolkit 安装完成"
    echo "  nvcc 版本：$(nvcc --version | grep 'release')"
fi

# 确认 nvidia-smi 信息
echo ""
echo "  GPU 信息："
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---------- 步骤1：创建 conda 环境 ----------
echo ""
echo "[步骤1] 创建 conda 环境 mamba_ir（Python 3.10）..."
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^mamba_ir "; then
    echo "  [提示] mamba_ir 环境已存在，跳过创建"
    echo "  如需重建：conda env remove -n mamba_ir -y"
else
    conda create -n mamba_ir python=3.10 -y
    echo "  [OK] conda 环境 mamba_ir 创建成功"
fi

conda activate mamba_ir

# ---------- 步骤2：安装 PyTorch 2.1.2（cu121，兼容 CUDA 12.x） ----------
echo ""
echo "[步骤2] 安装 PyTorch 2.1.2+cu121..."
echo "  → CUDA 12.2 使用 cu121 wheel（完全兼容）"
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
echo "  [OK] PyTorch 安装完成"

# 快速验证 PyTorch + CUDA
python -c "
import torch
print(f'  PyTorch: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
"

# ---------- 步骤3：安装 mamba-ssm（需要 nvcc 编译） ----------
echo ""
echo "[步骤3] 安装 mamba-ssm（编译 CUDA 扩展，约需 5~15 分钟）..."

echo "  3.1 安装 causal-conv1d >= 1.2.0..."
pip install causal-conv1d>=1.2.0

echo "  3.2 安装 mamba-ssm..."
# CUDA_HOME 确保编译时找到正确的 CUDA 路径
export CUDA_HOME=/usr/local/cuda-12.2
pip install mamba-ssm

echo "  [OK] mamba-ssm 安装完成"

# ---------- 步骤4：安装项目其余依赖 ----------
echo ""
echo "[步骤4] 安装项目其余依赖..."
pip install \
    einops>=0.6.1 \
    tqdm \
    Pillow \
    opencv-python \
    lpips \
    timm \
    scikit-image \
    huggingface_hub \
    safetensors
echo "  [OK] 项目依赖安装完成"

# ---------- 步骤5：完整验证 ----------
echo ""
echo "[步骤5] 验证 Mamba 安装..."
python - <<'PYEOF'
import torch
print(f"PyTorch 版本:  {torch.__version__}")
print(f"CUDA 可用:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 设备:      {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU 显存:      {mem:.1f} GB")

try:
    from mamba_ssm import Mamba
    # 测试三种通道数（对应 DA-MambaNet 的三个尺度）
    for ch in [36, 72, 144]:
        m = Mamba(d_model=ch, d_state=16, d_conv=4, expand=2).cuda()
        x = torch.randn(1, 64*64, ch).cuda()
        y = m(x)
        print(f"  Mamba(d_model={ch}): {x.shape} → {y.shape}  [OK]")
    print("\n[SUCCESS] mamba-ssm 安装验证完全通过！")
except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)
PYEOF

# ---------- 完成 ----------
echo ""
echo "========================================"
echo "[完成] mamba_ir 环境配置完毕！"
echo ""
echo "后续训练命令示例："
echo "  conda activate mamba_ir"
echo "  cd /path/to/new_poject_code"
echo "  python train.py --model DA_MambaNet --dataset mixed"
echo "========================================"= Mamba(d_model=36, d_state=16, d_conv=4, expand=2).cuda()
    x = torch.randn(1, 64, 36).cuda()  # (batch, seq_len, d_model)
    y = model(x)
    print(f"\n  Mamba 测试通过！输入: {x.shape} → 输出: {y.shape}")
    print("  [OK] mamba-ssm 安装验证成功")
except ImportError as e:
    print(f"  [ERROR] mamba-ssm 导入失败: {e}")
    exit(1)
except Exception as e:
    print(f"  [ERROR] Mamba 运行失败: {e}")
    exit(1)
EOF

# ---------- 完成 ----------
echo ""
echo "========================================"
echo "[完成] mamba_ir 环境配置完毕！"
echo ""
echo "后续使用方法："
echo "  conda activate mamba_ir"
echo "  python train.py --model DA_MambaNet ..."
echo "========================================"
