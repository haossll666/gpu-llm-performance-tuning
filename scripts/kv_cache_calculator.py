#!/usr/bin/env python3
"""
KV Cache Memory & Concurrency Calculator for LLM Serving
Supports:
  - Standard MHA (e.g., LLaMA-1/2)
  - GQA (Grouped-Query Attention, e.g., LLaMA-3, Qwen-2.5)
  - MLA (Multi-Head Latent Attention, e.g., DeepSeek-V2, DeepSeek-V3, DeepSeek-R1)

Usage:
  python3 scripts/kv_cache_calculator.py --arch deepseek-v3 --seq-len 4096 --batch-size 16 --kv-dtype fp8
  python3 scripts/kv_cache_calculator.py --arch llama3-70b --seq-len 8192 --gpu-vram 80 --model-weights 38
"""

import argparse
import sys

# Presets for popular architectures
MODEL_PRESETS = {
    "deepseek-v3": {
        "name": "DeepSeek-V3 / R1 (MLA)",
        "num_layers": 61,
        "arch_type": "mla",
        "kv_lora_rank": 512,       # c_t^{KV}
        "qk_rope_head_dim": 64,    # k_t^R
        "num_heads": 128,
        "head_dim": 128,
        "default_weights_gb": 685 * 0.5, # FP8 weight footprint ~340GB on 8/16 GPUs
    },
    "deepseek-v2-lite": {
        "name": "DeepSeek-V2-Lite (MLA)",
        "num_layers": 27,
        "arch_type": "mla",
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "num_heads": 16,
        "head_dim": 128,
        "default_weights_gb": 32, # ~16B params in FP16 or 8B in FP8
    },
    "llama3-8b": {
        "name": "Meta LLaMA-3-8B (GQA)",
        "num_layers": 32,
        "arch_type": "gqa",
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_weights_gb": 16,
    },
    "llama3-70b": {
        "name": "Meta LLaMA-3-70B (GQA)",
        "num_layers": 80,
        "arch_type": "gqa",
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_weights_gb": 140, # FP16 (~70GB in FP8 / INT4)
    },
    "qwen2.5-72b": {
        "name": "Qwen-2.5-72B (GQA)",
        "num_layers": 80,
        "arch_type": "gqa",
        "num_kv_heads": 8,
        "head_dim": 128,
        "default_weights_gb": 144,
    }
}

DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
}

def calculate_kv_cache_bytes_per_token(model_config, kv_dtype="fp16"):
    dtype_byte = DTYPE_BYTES[kv_dtype.lower()]
    arch = model_config.get("arch_type", "gqa")
    layers = model_config["num_layers"]

    if arch == "mla":
        # DeepSeek Multi-Head Latent Attention
        # Instead of storing standard K and V tensors:
        # KV Cache compresses into latent vector c_t^{KV} of dimension kv_lora_rank (512)
        # plus decoupled RoPE key vector k_t^R of dimension qk_rope_head_dim (64).
        # Total cached vector size per token per layer = kv_lora_rank + qk_rope_head_dim = 576 elements.
        elements_per_token_per_layer = model_config["kv_lora_rank"] + model_config["qk_rope_head_dim"]
        bytes_per_token = layers * elements_per_token_per_layer * dtype_byte
        return bytes_per_token
    elif arch == "gqa":
        # Grouped-Query Attention
        # 2 (Key + Value) * layers * num_kv_heads * head_dim * bytes_per_elem
        kv_heads = model_config["num_kv_heads"]
        head_dim = model_config["head_dim"]
        elements_per_token = 2 * layers * kv_heads * head_dim
        return elements_per_token * dtype_byte
    else:
        # Standard MHA
        num_heads = model_config["num_heads"]
        head_dim = model_config["head_dim"]
        elements_per_token = 2 * layers * num_heads * head_dim
        return elements_per_token * dtype_byte

def main():
    parser = argparse.ArgumentParser(description="KV Cache Memory & Concurrency Profiler for LLM Serving")
    parser.add_argument("--arch", choices=list(MODEL_PRESETS.keys()), default="deepseek-v3", help="Target model architecture preset")
    parser.add_argument("--seq-len", type=int, default=4096, help="Average or maximum sequence length (tokens)")
    parser.add_argument("--batch-size", type=int, default=16, help="Concurrent request batch size")
    parser.add_argument("--kv-dtype", choices=list(DTYPE_BYTES.keys()), default="fp8", help="KV cache storage data type")
    parser.add_argument("--gpu-vram", type=float, default=24.0, help="VRAM per GPU in GB (e.g. 24 for 4090, 80 for A100/H100)")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs in the server (e.g. 4x GPU)")
    parser.add_argument("--weights-vram-total", type=float, default=None, help="Total weight VRAM footprint across all GPUs in GB")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90, help="vLLM/SGLang gpu_memory_utilization factor (e.g. 0.90)")

    args = parser.parse_args()
    cfg = MODEL_PRESETS[args.arch]

    bytes_per_token = calculate_kv_cache_bytes_per_token(cfg, args.kv_dtype)
    bytes_per_seq = bytes_per_token * args.seq_len
    mb_per_seq = bytes_per_seq / (1024 * 1024)
    gb_per_batch = (bytes_per_seq * args.batch_size) / (1024 * 1024 * 1024)

    total_cluster_vram = args.gpu_vram * args.num_gpus
    weights_total = args.weights_vram_total if args.weights_vram_total is not None else cfg["default_weights_gb"]
    usable_cluster_vram = (total_cluster_vram * args.gpu_mem_util) - weights_total

    print("=" * 72)
    print(f" LLM KV Cache & Concurrency Profiler")
    print(f" Model Preset: {cfg['name']}")
    print(f" Cluster Setup: {args.num_gpus}x GPUs @ {args.gpu_vram:.0f} GB VRAM (Total {total_cluster_vram:.0f} GB)")
    print(f" Precision: KV Cache format = {args.kv_dtype.upper()} ({DTYPE_BYTES[args.kv_dtype]} bytes/element)")
    print("=" * 72)

    print(f"\n[1] Per-Token & Per-Sequence Footprint:")
    print(f"  • KV Cache per token:       {bytes_per_token / 1024:.2f} KB / token")
    print(f"  • Single request ({args.seq_len} tokens): {mb_per_seq:.2f} MB")
    print(f"  • Batch of {args.batch_size} ({args.seq_len} tokens): {gb_per_batch:.2f} GB")

    print(f"\n[2] Cluster Capacity & Max Concurrency Projection:")
    print(f"  • Weight memory footprint:  {weights_total:.1f} GB (across {args.num_gpus} GPUs)")
    print(f"  • Usable KV Cache pool:     {max(0.0, usable_cluster_vram):.1f} GB")

    if usable_cluster_vram <= 0:
        print(f"  ⚠️ WARNING: Model weights exceed or fill all available VRAM!")
        print(f"     You must apply quantization (e.g. AWQ/FP8/GGUF) or scale GPU count.")
    else:
        max_tokens_cluster = (usable_cluster_vram * 1024 * 1024 * 1024) / bytes_per_token
        max_concurrency = int(max_tokens_cluster / args.seq_len)
        print(f"  • Max cached token capacity:{int(max_tokens_cluster):,} tokens")
        print(f"  • Max concurrent requests:  {max_concurrency} streams (@ {args.seq_len} seq_len)")

    print("\n[3] DeepSeek MLA Comparison Advantage:")
    if cfg["arch_type"] == "mla":
        # Calculate standard GQA counterpart
        gqa_equivalent_bytes = 2 * cfg["num_layers"] * 8 * 128 * DTYPE_BYTES[args.kv_dtype]
        compression_ratio = gqa_equivalent_bytes / bytes_per_token
        print(f"  • MLA vs 8-head GQA size:   ~{compression_ratio:.2f}x smaller memory footprint!")
        print(f"  • Benefit: Dramatically increases serving concurrency and PagedAttention pool.")
    print("=" * 72)

if __name__ == "__main__":
    main()
