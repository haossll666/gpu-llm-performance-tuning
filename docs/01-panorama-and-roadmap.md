# 01. 全景地图与建议学习路线 (System Panorama & Learning Roadmap)

## 一、 系统架构全景链路 (End-to-End System Pipeline)

大语言模型（LLM）推理服务绝非简单的单次矩阵乘法，而是一个从应用层流式网络协议向下直钻至 GPU 硅片微架构的复杂分布式系统。

```
[User Request (HTTP/gRPC/SSE)]
        │
        ▼
[01. Latency & SLA] ─── TTFT (排队+首字Prefill) / TPOT (逐字Decode) / ITL (抖动)
        │
        ▼
[02. Serving Engine] ── vLLM / SGLang: 动态调度、内存管理与分布式编排
        │
   ┌────┴───────────────────────────┬───────────────────────────┐
   ▼                                ▼                           ▼
[03. Workload]                 [04. KV Cache]              [05. Model Architecture]
  • 并发请求数 (Concurrency)      • PagedAttention (分块分页)   • Dense vs MoE (混合专家)
  • Prompt/Output Token 分布      • RadixAttention (前缀复用)   • Attention: MHA/GQA/MLA
  • 泊松到达流 (Arrival Rate)    • 显存池与驱逐保护机制          • 激活参数 vs 总参数
   │                                │                           │
   └───────────────┬────────────────┘                           │
                   ▼                                            │
        [06. Iteration Scheduler]                               │
          • 连续批处理 (Continuous Batching)                     │
          • 分块预填充 (Chunked Prefill)                        │
                   │                                            │
                   └─────────────────────┬──────────────────────┘
                                         ▼
                             [07. Multi-GPU Interconnect]
                               • 4×GPU 拓扑: PCIe Gen4/Gen5 vs NVLink
                               • 并行策略: 张量并行 (TP) vs 专家并行 (EP)
                               • NCCL 原语: All-Reduce vs All-to-All
                                         │
                                         ▼
                               [08. CUDA Kernel & HW]
                                 • FlashAttention-2/3 / FlashDecoding
                                 • W8A8 / FP8 GEMM / Triton MoE
                                 • SM 调度器 / Tensor Core / HBM 访存墙
                                         │
                                         ▼
                            [09. Observability & Profiling]
                              • Prometheus + Grafana (DCGM / 引擎指标)
                              • Nsight Systems (时序流水线与通信气泡)
                              • Nsight Compute (Kernel 级算力与访存 Roofline)
```

---

## 二、 建议 10 步进阶学习路线

1. **Prefill vs Decode (计算密集 vs 访存密集)**：
   - Prefill 阶段处理全量 Prompt，激活大量矩阵乘法，属于 **Compute-Bound（算力受限）**。
   - Decode 阶段自回归逐字生成，每次仅加载 1 个 Token 并扫描全部模型权重和历史 KV Cache，属于 **Memory-Bandwidth Bound（显存带宽受限）**。
2. **TTFT / TPOT / ITL / tok/s (延迟与吞吐黄金指标)**：
   - TTFT（Time to First Token）主导交互首屏体验；TPOT（Time Per Output Token）与 ITL（Inter-Token Latency）决定阅读流畅感；Throughput 决定单位运营成本。
3. **KV Cache (PagedAttention 与 MLA 革命)**：
   - 彻底搞懂传统 MHA 显存浪费痛点、PagedAttention 的虚拟内存映射机制，以及 DeepSeek MLA 如何通过低秩投影将显存占用压缩至 1/5 甚至 1/9。
4. **HBM / Tensor Core / Roofline 物理极限定律**：
   - 掌握 Roofline 模型。推导硬件 Machine Balance 拐点，明确为什么在 Decode 阶段一味堆算力（TFLOPS）无法提升单流速度，唯有 HBM 带宽（GB/s）是物理瓶颈。
5. **TP vs EP vs DP (单机多卡分布式并行)**：
   - 4×GPU 下的并行切分博弈：为何 Dense 模型首选 TP，而 MoE 模型在大 Batch 或 PCIe 环境下优先考虑 EP 甚至是混合并行。
6. **NCCL / PCIe / NVLink 通信墙**：
   - 掌握 NCCL 环形与树形通信机制。剖析无 NVLink 时，单向 31.5 GB/s 的 PCIe 4.0 如何被频繁的 All-Reduce 阻塞打爆，以及如何调优 NCCL 环境变量。
7. **Attention / MoE 底层算子加速**：
   - 理解 FlashAttention 的 IO 感知（SRAM 分块加载避免频繁写 HBM）、FlashDecoding 在长上下文下的并行归约，以及 Triton 编写的 MoE Gating/Dispatch/Combine 算子。
8. **vLLM / SGLang 生产级实战与调参**：
   - 掌握 `--enable-chunked-prefill`（消除长 prompt 对流式 decode 的打断）、`--enable-prefix-caching`（Radix 树多轮缓存）、`--gpu-memory-utilization`。
9. **Prometheus / Grafana 全景监控观测**：
   - 结合 NVIDIA DCGM-Exporter 与推理引擎内置 metrics，搭建首字延迟、逐字延迟、队列积压、显存利用率及 PCIe 吞吐的监控大盘。
10. **Nsight Systems / Compute 深度抓包与瓶颈定位**：
    - 实战使用 `nsys` 捕获 CUDA Kernel 执行时序流，精准定位通信气泡、CPU Launch 开销、Host-to-Device 拷贝与算子倾斜。
