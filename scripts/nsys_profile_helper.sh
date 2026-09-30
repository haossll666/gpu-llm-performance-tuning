#!/usr/bin/env bash
# ==============================================================================
# Nsight Systems Profiling Automation Helper for vLLM & SGLang
# Captures CUDA Kernels, NCCL Comm, OS Runtime, CPU/GPU Bubbles, and Memory Transfers
# ==============================================================================

set -euo pipefail

OUTPUT_DIR="./profiles"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_NAME="${OUTPUT_DIR}/llm_profile_${TIMESTAMP}"

echo "======================================================================"
echo " 🔍 Nsight Systems LLM Profiler Helper"
echo " Report destination: ${REPORT_NAME}.nsys-rep"
echo "======================================================================"

# Profiling configuration:
# --trace: cuda, nvtx (for engine markers), osrt (OS runtime), cudnn, cublas
# --cuda-memory-usage: tracks allocation peaks (vital for KV Cache debugging)
# --delay: warm up the engine first (e.g. 15s) so we capture steady-state decode
# --duration: capture 30 seconds of high-concurrency requests
NSYS_CMD=(
  nsys profile
  --trace=cuda,nvtx,osrt,cublas
  --cuda-memory-usage=true
  --cuda-graph-trace=node
  --delay=15
  --duration=30
  --sample=process-tree
  --stats=true
  --force-overwrite=true
  --output="${REPORT_NAME}"
)

echo "Profile command template:"
echo "${NSYS_CMD[*]} vllm serve /models/DeepSeek-V2-Lite --tensor-parallel-size 4 --port 8000"
echo ""
echo "💡 Profiling Checklist:"
echo " 1. Start server with above command in Terminal 1."
echo " 2. In Terminal 2, run a concurrency stress bench (e.g. 32 concurrent requests)."
echo " 3. After 45s, open ${REPORT_NAME}.nsys-rep in NVIDIA Nsight Systems GUI."
echo " 4. Key timelines to inspect:"
echo "    - 'ncclKernel_AllToAll' or 'ncclKernel_AllReduce' duration on PCIe."
echo "    - Gap/bubbles between CPU launch and GPU kernel execution."
echo "    - FlashAttention vs FlashDecoding kernel saturation."
echo "======================================================================"
