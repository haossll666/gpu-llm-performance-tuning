# 05. 工具链实战指南 (Toolchains Guide)

## 一、 vLLM & SGLang 生产参数对照表

| 核心特性 | vLLM 关键参数 | SGLang 关键参数 | 调优核心意义 |
| :--- | :--- | :--- | :--- |
| **张量并行 / 专家并行** | `--tensor-parallel-size 4` | `--tp-size 4` | 单机 4 卡显存聚合与算力切分 |
| **分块预填充 (Chunked)** | `--enable-chunked-prefill true` | `--chunked-prefill-size 2048` | 防止长 Prompt 独占 GPU 打断存量 Decode |
| **前缀缓存 (Prefix)** | `--enable-prefix-caching` | `--enable-radix-attention` | 系统提示词与多轮对话 KV 零重算复用 |
| **KV 精度压缩** | `--kv-cache-dtype fp8` | `--kv-cache-dtype fp8` | 显存空间释放 50%，提升 2 倍以上并发上限 |
| **显存预留配比** | `--gpu-memory-utilization 0.92` | `--mem-fraction-static 0.90` | 预留系统与动态 CUDA 内核分配空间 |

---

## 二、 NVIDIA Nsight Systems 抓包实战流程

```bash
# 1. 运行自动化抓包助手脚本
bash scripts/nsys_profile_helper.sh

# 2. 或手动执行底层抓包（关注 CUDA 内存与 NVTX 标记）
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --delay=10 --duration=20 \
  -o ./profiles/trace_vllm \
  vllm serve /models/deepseek-v2-lite --tensor-parallel-size 4
```

### 分析三板斧：
1. **看 CUDA Stream 连续性**：是否出现大片空白（CPU Host 调度瓶颈或锁争抢）。
2. **看 NCCL 通信耗时**：查找 `ncclKernel_AllReduce` 或 `ncclKernel_AllToAll`，查看是否与 Compute Kernel 重叠（Overlap）。
3. **看 Attention 算子版本**：核验是否为 `flash_attn_bwd` / `flash_fwd_kernel`，排查旧版未融合算子。
