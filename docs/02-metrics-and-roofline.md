# 02. LLM 推理关键指标与 Roofline 极限分析

## 一、 服务核心指标与 SLA 规范

| 指标全称 | 简称 | 阶段归属 | 核心影响因素 | 推荐生产 SLA 阈值 |
| :--- | :--- | :--- | :--- | :--- |
| **Time to First Token** | **TTFT** | 排队 + Prefill | 网络握手、等待队列排队、Prompt 长度、矩阵乘法算力（TFLOPS） | **P50 < 400ms, P95 < 1.0s** |
| **Time Per Output Token** | **TPOT** | Decode 自回归 | GPU 显存读取带宽（HBM GB/s）、模型激活参数体积、KV Cache 大小 | **20ms ~ 35ms / tok** (28~50 tok/s) |
| **Inter-Token Latency** | **ITL** | Decode 流式输出 | 突发大 Prompt 插入打断当前迭代（Jitter 抖动） | **P99 抖动 < 50ms** |
| **Model FLOPs Utilization** | **MFU** | 全局硬件利用率 | 计算密集程度与 Tensor Core 饱和度 | Prefill 阶段 35%~55%，Decode 阶段 < 5% |
| **Model Bandwidth Utilization** | **MBU** | 访存总线效率 | 每次前向计算实际载入权重字节 / 理论峰值显存带宽 | Decode 阶段应争取达到 **65% ~ 80%** |

---

## 二、 Roofline 模型在 LLM 推理中的本质映射

Roofline 模型定义了处理器的物理性能边界：
$$\text{Attainable Performance (TFLOPS)} = \min\left(\text{Peak TFLOPS}, \text{Operational Intensity} \times \text{Peak Bandwidth}\right)$$

### 1. 机器平衡点 (Machine Balance / Knee Point)
$$I^* = \frac{\text{Peak TFLOPS}}{\text{Peak Bandwidth (TB/s)}}$$
- **RTX 4090**：$165.2 \text{ TFLOPS} / 1.008 \text{ TB/s} \approx 163.9 \text{ FLOPs/Byte}$
- **NVIDIA L40S**：$366.0 \text{ TFLOPS} / 0.864 \text{ TB/s} \approx 423.6 \text{ FLOPs/Byte}$
- **NVIDIA H100 SXM**：$989.0 \text{ TFLOPS} / 3.350 \text{ TB/s} \approx 295.2 \text{ FLOPs/Byte}$

### 2. Prefill 与 Decode 阶段的物理区隔
- **Prefill 阶段**：Prompt 长度 $T$ 较大，矩阵计算属于大尺度 GEMM，算力密度 $I_{\text{prefill}} \approx \frac{2 \cdot T}{bytes} \ge 200 \text{ FLOPs/Byte}$。
  - **判定**：算力受限（Compute-Bound），性能直接受 GPU TFLOPS 算力制约。
- **Decode 阶段**：自回归生成每步仅处理 1 个 Token，对于 Batch Size 为 $B$ 的场景，算力密度 $I_{\text{decode}} \approx \frac{2 \cdot B}{bytes}$。当并发流 $B=8$ 时，算力密度仅为 $8 \text{ FLOPs/Byte}$，远远低于硬件平衡点 163.9！
  - **判定**：访存极度受限（Memory-Bandwidth Bound）。GPU 的 Tensor Core 处于大幅空闲等待状态，生成时延完全由显存读取权重的物理速度决定。
