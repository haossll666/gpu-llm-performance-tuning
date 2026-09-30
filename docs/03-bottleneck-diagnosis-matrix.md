# 03. GPU LLM 性能瓶颈排查矩阵与实战决策树

## 一、 核心排查矩阵 (Bottleneck Diagnosis Matrix)

| 异常现象 (Symptom) | 典型可观测指标 | 潜在根因 (Root Cause) | 推荐调优方案 (Remediation) |
| :--- | :--- | :--- | :--- |
| **首字延迟 (TTFT) 极高，但后字 (TPOT) 正常** | `vllm:time_to_first_token` 飘高，`vllm:num_requests_waiting > 15` | 等待队列排队严重，或新进长 Prompt 霸占 GPU 进行全量 Prefill | 开启 `--enable-chunked-prefill true`，限制 `--max-num-batched-tokens 2048`；增加节点做负载均衡 |
| **4×GPU PCIe 下吞吐上不去，SM% 低，PCIe 满** | `DCGM_FI_DEV_PCIE_TX_THROUGHPUT > 25GB/s`，SM% < 15% | 在无 NVLink 的 PCIe 拓扑下使用了 TP=4，高频 All-Reduce 阻塞 | DeepSeek MoE 改用 EP=4（专家并行），设置 `NCCL_BUFFSIZE=4194304`，避免跨 Socket NUMA |
| **长文本多轮对话高并发下吞吐周期性腰斩** | `vllm:num_preemptions` 频繁计数，显存水位 > 95% | 显存池耗尽触发请求驱逐与重算（Thrashing） | 开启 `--enable-prefix-caching`（Radix 前缀复用），开启 `--kv-cache-dtype fp8`，约束 `max-model-len` |
| **Decode 阶段延迟固定，SM% 仅 3%~5%** | SM% 极低，HBM 带宽利用率（MBU）接近上限 | 处于单 Batch/小 Batch 物理访存墙区间 | 引入 **FP8 / W8A8 权重压缩** 减少访存量，或部署 **投机解码（Speculative Decoding）** 增加算力密度 |
| **Nsight 发现 Step 之间存在明显空白空泡** | CUDA Stream 间隙大，CPU 占用率高 | Python 调度层或框架 CPU-GPU 同步阻塞（Launch Overhead） | 启用 CUDA Graph（`--enforce-eager false`），调大调度 Batching 粒度 |
