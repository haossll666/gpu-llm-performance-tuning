#!/usr/bin/env python3
"""
Hardware Roofline & PCIe Communication Bound Analyzer for LLM Serving
Analyzes:
  - Theoretical Arithmetic Intensity (FLOPs / Byte) for Prefill vs Decode
  - GPU Ridge Point / Machine Balance (Peak TFLOPS / Peak Memory Bandwidth)
  - 4xGPU PCIe Topology All-to-All vs All-Reduce Communication Bottleneck for DeepSeek MoE

Usage:
  python3 scripts/roofline_analyzer.py --gpu rtx4090 --batch-size 8 --prompt-len 2048 --out-len 512
  python3 scripts/roofline_analyzer.py --gpu h100-sxm --topology pcie4 --moe-experts 64 --top-k 8
"""

import argparse
import sys

GPU_SPECS = {
    "rtx4090": {
        "name": "NVIDIA GeForce RTX 4090",
        "fp16_tflops": 165.2,    # Tensor Core Dense FP16 TFLOPS
        "fp8_tflops": 330.0,
        "mem_bw_gb_s": 1008.0,   # 1008 GB/s GDDR6X
        "vram_gb": 24,
        "pcie_gen": "PCIe 4.0 x16 (31.5 GB/s bidirectional)",
        "pcie_bw_gb_s": 31.5,
    },
    "l40s": {
        "name": "NVIDIA L40S",
        "fp16_tflops": 366.0,
        "fp8_tflops": 733.0,
        "mem_bw_gb_s": 864.0,    # 864 GB/s GDDR6 with ECC
        "vram_gb": 48,
        "pcie_gen": "PCIe 4.0 x16 (31.5 GB/s bidirectional)",
        "pcie_bw_gb_s": 31.5,
    },
    "a100-sxm": {
        "name": "NVIDIA A100 SXM4 80GB",
        "fp16_tflops": 312.0,
        "fp8_tflops": 312.0,     # A100 has no native FP8
        "mem_bw_gb_s": 2039.0,   # ~2 TB/s HBM2e
        "vram_gb": 80,
        "pcie_gen": "NVLink 3 (600 GB/s bidirectional)",
        "pcie_bw_gb_s": 600.0,
    },
    "h100-sxm": {
        "name": "NVIDIA H100 SXM5 80GB",
        "fp16_tflops": 989.0,
        "fp8_tflops": 1978.0,
        "mem_bw_gb_s": 3350.0,   # 3.35 TB/s HBM3
        "vram_gb": 80,
        "pcie_gen": "NVLink 4 (900 GB/s bidirectional)",
        "pcie_bw_gb_s": 900.0,
    }
}

def analyze_roofline(gpu_key, precision="fp16", batch_size=8, prompt_len=2048, hidden_size=7168, num_layers=61):
    gpu = GPU_SPECS[gpu_key]
    peak_tflops = gpu["fp8_tflops"] if precision == "fp8" else gpu["fp16_tflops"]
    peak_mem_bw = gpu["mem_bw_gb_s"]

    # Machine Balance (Ridge Point / Knee Point) in FLOPs/Byte
    # If Arithmetic Intensity > Ridge Point => Compute Bound
    # If Arithmetic Intensity < Ridge Point => Memory Bandwidth Bound
    ridge_point = (peak_tflops * 1e12) / (peak_mem_bw * 1e9)

    # Decode Phase Arithmetic Intensity:
    # In decode, for each token generated:
    # Computations per token ~ 2 * Parameter Count FLOPs
    # Memory read ~ Parameter Count * bytes_per_param (reading weights from HBM once)
    bytes_per_param = 1 if precision == "fp8" else 2
    # Arithmetic Intensity for batch_size B:
    # Total FLOPs = 2 * P * B
    # Weight Memory Read = P * bytes_per_param
    # KV Cache read = 2 * layers * kv_heads * head_dim * seq_len * bytes
    # Dominant term for weights: I_decode ~ (2 * B) / bytes_per_param
    ai_decode = (2.0 * batch_size) / bytes_per_param

    # Prefill Phase Arithmetic Intensity:
    # GEMM with M=prompt_len, K=hidden_size, N=intermediate_size
    # FLOPs = 2 * prompt_len * hidden_size * intermediate_size
    # Memory = bytes_per_param * (hidden_size * intermediate_size + prompt_len * hidden_size)
    # When prompt_len is large (e.g. 2048), GEMM tiles reuse weights in SRAM/L2!
    # Empirical AI for Prefill is significantly higher:
    ai_prefill = min(200.0, (2.0 * prompt_len) / bytes_per_param)

    # Determine bounds
    decode_bound = "Memory-Bandwidth Bound (访存受限)" if ai_decode < ridge_point else "Compute Bound (算力受限)"
    prefill_bound = "Compute Bound (算力受限)" if ai_prefill >= ridge_point else "Memory-Bandwidth Bound (访存受限)"

    # Theoretical TPOT (Time Per Output Token) in ms
    # Decode is memory-bound: Latency ~ Weight_size / Bandwidth
    # Assuming DeepSeek 21B active params (or custom model ~30GB weights)
    active_weights_gb = 24.0 * bytes_per_param # simplified representative active params per token
    theoretical_decode_latency_s = active_weights_gb / peak_mem_bw
    theoretical_tpot_ms = theoretical_decode_latency_s * 1000.0

    return {
        "gpu": gpu,
        "ridge_point": ridge_point,
        "ai_decode": ai_decode,
        "ai_prefill": ai_prefill,
        "decode_bound": decode_bound,
        "prefill_bound": prefill_bound,
        "theoretical_tpot_ms": theoretical_tpot_ms,
    }

def analyze_4xgpu_pcie_moe(num_gpus=4, pcie_bw_gb_s=31.5, num_experts=64, top_k=8, hidden_size=7168, batch_size=16, dtype_bytes=2):
    """
    Analyzes DeepSeek MoE Expert Parallelism (EP=4) All-to-All communication bottleneck on 4xGPU PCIe without NVLink.
    Communication volume per MoE layer in EP:
      1. Dispatch: Each GPU sends tokens assigned to remote experts.
         Average remote tokens per GPU = batch_size * top_k * ((num_gpus - 1) / num_gpus)
         Dispatch Volume = remote_tokens * hidden_size * dtype_bytes
      2. Combine: Remote experts send back outputs.
         Combine Volume = remote_tokens * hidden_size * dtype_bytes
      Total Comm per MoE layer = 2 * Dispatch Volume
    """
    remote_ratio = (num_gpus - 1) / num_gpus
    remote_tokens_per_gpu = batch_size * top_k * remote_ratio
    comm_bytes_per_layer = 2 * remote_tokens_per_gpu * hidden_size * dtype_bytes
    comm_mb_per_layer = comm_bytes_per_layer / (1024 * 1024)

    # 60 MoE layers (DeepSeek-V3 has 61 layers total, 1 dense + 60 MoE)
    moe_layers = 60
    total_comm_mb_per_step = comm_mb_per_layer * moe_layers

    # Effective PCIe bandwidth with NCCL ring/tree overhead (~70% efficiency on PCIe)
    effective_bw_gb_s = pcie_bw_gb_s * 0.70
    comm_time_per_step_ms = (total_comm_mb_per_step / (effective_bw_gb_s * 1024)) * 1000.0

    # Compare with TP=4 All-Reduce communication:
    # In TP=4, every Attention and MLP layer needs All-Reduce!
    # All-Reduce volume per layer = 2 * ((num_gpus - 1) / num_gpus) * batch_size * hidden_size * dtype_bytes
    # DeepSeek MLA has compressed projection, but MLP TP requires 2 All-Reduces per layer * 61 layers = 122 All-Reduces!
    tp_allreduce_bytes_per_layer = 2 * remote_ratio * batch_size * hidden_size * dtype_bytes
    tp_total_mb = (tp_allreduce_bytes_per_layer * 122) / (1024 * 1024)
    tp_comm_time_ms = (tp_total_mb / (effective_bw_gb_s * 1024)) * 1000.0

    return {
        "remote_tokens_per_gpu": remote_tokens_per_gpu,
        "ep_comm_mb_per_step": total_comm_mb_per_step,
        "ep_comm_time_ms": comm_time_per_step_ms,
        "tp_total_mb": tp_total_mb,
        "tp_comm_time_ms": tp_comm_time_ms,
    }

def main():
    parser = argparse.ArgumentParser(description="LLM Roofline & 4xGPU PCIe MoE Comm Analyzer")
    parser.add_argument("--gpu", choices=list(GPU_SPECS.keys()), default="rtx4090")
    parser.add_argument("--precision", choices=["fp16", "fp8"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=2048)
    args = parser.parse_args()

    res = analyze_roofline(args.gpu, args.precision, args.batch_size, args.prompt_len)
    moe = analyze_4xgpu_pcie_moe(batch_size=args.batch_size)

    gpu = res["gpu"]
    print("=" * 76)
    print(f" LLM Hardware Roofline & 4xGPU PCIe Bottleneck Analysis")
    print(f" Target Device: {gpu['name']}")
    print(f" Precision: {args.precision.upper()} | Batch Size: {args.batch_size} | Prompt Len: {args.prompt_len}")
    print("=" * 76)
    print(f"\n[1] Hardware Specs & Machine Balance:")
    print(f"  • Peak Compute:       {gpu['fp16_tflops']:.1f} TFLOPS (FP16) / {gpu['fp8_tflops']:.1f} TFLOPS (FP8)")
    print(f"  • Peak Memory BW:     {gpu['mem_bw_gb_s']:.0f} GB/s")
    print(f"  • Interconnect:       {gpu['pcie_gen']}")
    print(f"  • Machine Balance (Ridge Point): {res['ridge_point']:.2f} FLOPs/Byte")
    print(f"    (若算力密度 > {res['ridge_point']:.1f}，进入算力饱和区；低于此值则处于访存瓶颈区)")

    print(f"\n[2] Serving Phase Bound Verdict:")
    print(f"  • Prefill 阶段算力密度: ~{res['ai_prefill']:.1f} FLOPs/Byte -> 判定: {res['prefill_bound']}")
    print(f"  • Decode 阶段算力密度:  ~{res['ai_decode']:.1f} FLOPs/Byte -> 判定: {res['decode_bound']}")
    print(f"  • 结论: Decode 生成速度完全取决于 HBM/GDDR 带宽，而非 TFLOPS 峰值！")

    print(f"\n[3] 4xGPU PCIe (无 NVLink) DeepSeek MoE 并行通信代价对比:")
    print(f"  • EP=4 (Expert Parallel) 60层 MoE All-to-All 通信总量: {moe['ep_comm_mb_per_step']:.2f} MB / 步")
    print(f"    预计通信纯开销: ~{moe['ep_comm_time_ms']:.2f} ms / 步 (仅传输激活值)")
    print(f"  • TP=4 (Tensor Parallel) 122次 All-Reduce 通信总量:     {moe['tp_total_mb']:.2f} MB / 步")
    print(f"    预计通信纯开销: ~{moe['tp_comm_time_ms']:.2f} ms / 步 (高频通信气泡)")
    print(f"  💡 生产建议: 在无 NVLink 的 PCIe 拓扑下，EP=4 通信次数远少于 TP=4，结合 Chunked Prefill 效益更佳。")
    print("=" * 76)

if __name__ == "__main__":
    main()
