# 04. DeepSeek MoE 在 4×GPU (PCIe) 落地与性能调优指南

## 一、 为什么 4×GPU PCIe 拓扑部署 DeepSeek MoE 极具挑战？

在高性能算力中心（如配备 NVLink 4 900 GB/s 的 H100 SXM 节点），GPU 间的通信带宽几乎与内存带宽同量级。然而在工业级私有化部署或中小团队场景中，**4×RTX 4090 / L40S PCIe** 是极为普遍的高性价比硬件底座：
- **物理带宽悬殊**：PCIe 4.0 x16 双向理论带宽仅为 **31.5 GB/s**（实际有效传输约 22~25 GB/s），仅为 NVLink 的 **1/30**！
- **拓扑不均匀性**：4 张 PCIe 卡通常挂载在不同的 CPU Socket / Root Complex 下（通过 `nvidia-smi topo -m` 查看，常常显示为 `SYS` 或 `PHB`），跨 NUMA 传输时延大幅劣化。
- **MoE 动态路由通信量放大**：DeepSeek MoE 拥有大量路由专家（如 DeepSeek-V2/V3 中的 64/256 专家，Top-8 激活），若并行策略设计不当，GPU 间将出现大量通信阻塞气泡。

---

## 二、 4×GPU 并行策略博弈：TP=4 还是 EP=4？

### 1. TP=4 (张量并行 / Tensor Parallelism)
- **机制**：将每一层的 Attention 投影矩阵与 MLP 权重纵横切分到 4 张 GPU 上。
- **通信行为**：
  - 每个注意力层和 MLP 层计算完毕后，必须执行全局 **All-Reduce** 同步。
  - 对于一个 60 层的模型，单步生成将产生 **120+ 次 All-Reduce 阻塞**！
  - 在 PCIe 拓扑下，每次小量数据的频繁 All-Reduce 将触发极其昂贵的同步时延开销，**通信气泡占比常超过 50%**，导致 GPU SM 利用率极低。

### 2. EP=4 (专家并行 / Expert Parallelism)
- **机制**：Attention 层与共享专家（Shared Expert）在 4 张卡上全量复制（或者小范围切分）；路由专家分散在 4 张卡上（例如每卡承载 16 个专家）。
- **通信行为**：
  - 仅在 MoE 层通过 **All-to-All** 分发 Token 激活值（Dispatch），计算后再次通过 **All-to-All** 汇聚输出（Combine）。
  - 通信量仅与 `Batch Size × Top-K 激活专家数 × 隐藏层维度` 相关，而**无需在卡间传输庞大的模型权重**。
- **PCIe 决策建议**：
  - 当并发批次适中（$Batch < 32$）时，**EP=4 的总通信字节量显著低于 TP=4**，且减少了全局同步屏障次数。
  - 若显存足够存放复制的 Attention 权重，优先考虑 EP=4；若单卡无法装下单份 Attention，可采用混合并行（TP=2 + EP=2）。

---

## 三、 DeepSeek MLA (Multi-Head Latent Attention) 显存革命

传统 LLaMA 架构（MHA/GQA）的 KV Cache 显存占用随上下文长度线性爆炸：
$$\text{Memory}_{\text{GQA}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{dtype\_bytes}$$

DeepSeek 引入了 **MLA 创新机制**：
1. **低秩压缩**：在 Key 和 Value 投影前，将隐藏状态投影至低秩潜在向量 $c_t^{KV}$（维度通常为 512）。
2. **解耦 RoPE Key**：仅额外保留包含旋转位置编码的专用向量 $k_t^R$（维度为 64）。
3. **缓存体积剧减**：
   $$\text{Elements}_{\text{MLA}} = 512 + 64 = 576 \text{ 元素 / Token / 层}$$
   相比于典型 8-head GQA（$2 \times 8 \times 128 = 2048$ 元素），**显存直接压缩至原本的 28%（约 3.56 倍紧凑）**！
4. **工业红利**：在 4×GPU 仅有 24GB/48GB 显存的受限环境下，MLA 允许将最大并发序列提升 3~4 倍而不发生显存击穿。

---

## 四、 4×GPU 生产级启动调优参数范式

### vLLM 生产级启动模板
```bash
# 1. 设置 PCIe 拓扑最佳 NCCL 参数
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=1          # 单机内关闭 IB 探测
export NCCL_P2P_DISABLE=0         # 强制启用 PCIe P2P
export NCCL_BUFFSIZE=4194304      # 设置 4MB 环形缓冲区，平滑 PCIe 突发传输

# 2. 启动服务
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
```

### 关键调优旗标解析：
- `--enable-chunked-prefill true`：**救命级调优项**。将长 Prompt（如 4096 tokens）切分成 512/2048 的微批次，允许在同一个 Iteration 内部穿插其他并发流的 Decode 计算，将 Decode 阶段的 ITL 抖动降低 80% 以上。
- `--enable-prefix-caching`：在多轮对话与系统提示词（System Prompt）重合场景下，复用已计算的 Radix KV 缓存，**TTFT 从数百毫秒直接降至个位数毫秒**。
- `--kv-cache-dtype fp8`：将 MLA KV Cache 存储精度由 BF16 进一步降为 FP8，显存容量再翻倍，单机 4 卡即可支撑数百高并发长文本流。
