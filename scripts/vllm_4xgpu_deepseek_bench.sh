#!/usr/bin/env bash
# ==============================================================================
# Production Launch & Tuning Script: 4xGPU PCIe DeepSeek MoE Serving (vLLM & SGLang)
# ==============================================================================

set -euo pipefail

MODEL_PATH="${1:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
GPU_COUNT=4

echo "======================================================================"
echo " 🚀 Launching High-Performance Serving on 4x GPUs (PCIe Topology)"
echo " Model: ${MODEL_PATH}"
echo "======================================================================"

# 1. Hardware & NCCL Environment Tuning for PCIe (without NVLink)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV
# Disable Infiniband if on single-node PCIe
export NCCL_IB_DISABLE=1
# Enable P2P via PCIe switch if supported
export NCCL_P2P_DISABLE=0
# Tune ring buffer size for PCIe (4MB or 8MB)
export NCCL_BUFFSIZE=4194304

# 2. Check PCIe P2P topology
echo "--- GPU Interconnect Topology ---"
if command -v nvidia-smi &> /dev/null; then
  nvidia-smi topo -m
else
  echo "nvidia-smi not available (running in simulation/dry-run mode)"
fi

echo ""
echo "--- vLLM Tuned Command (Chunked Prefill + Prefix Caching + MLA) ---"
cat << 'EOF'
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  --enable-chunked-prefill true \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --swap-space 16 \
  --port 8000
EOF

echo ""
echo "--- SGLang Tuned Command (RadixAttention + Multi-GPU) ---"
cat << 'EOF'
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --tp-size 4 \
  --mem-fraction-static 0.90 \
  --context-length 8192 \
  --chunked-prefill-size 2048 \
  --enable-radix-attention \
  --port 30000
EOF
echo "======================================================================"
