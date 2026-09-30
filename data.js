// ==============================================================================
// Unified Data Layer for GPU LLM Performance Tuning
// ==============================================================================

export const PIPELINE_NODES = [
  {
    id: "latency",
    title: "Latency (TTFT / TPOT)",
    subtitle: "TTFT / TPOT / ITL 延迟指标",
    category: "metrics",
    summary: "TTFT 主要反映排队+Prefill；TPOT/ITL 更接近 Decode 持续生成速度。学习重点：“慢”必须先被量化。",
    whatIsIt: "用户体验的直接映射。拆分为 TTFT (Time to First Token，从请求发出到收到第一个 Token)、TPOT (Time Per Output Token，后续每个 Token 的平均耗时) 和 ITL (Inter-Token Latency，Token 间流式抖动)。",
    keyMetrics: "• TTFT SLA: 优于 500ms ~ 1.2s\n• TPOT: 20 ~ 40ms/tok (约 25~50 tok/s，符合人类极佳阅读速度)\n• ITL P99 抖动: 需 < 60ms",
    symptoms: "• TTFT 极高但 TPOT 正常：等待队列过长（Capacity 耗尽）或 Prefill 缺乏分块被打断。\n• TPOT 偶尔突发卡顿数十倍：长 Prompt 请求插队触发未分块的 Prefill 计算，打断当前 Decode 流。",
    moeNotes: "在 4×GPU PCIe 拓扑部署 DeepSeek MoE 时，首字延迟会叠加 All-to-All 路由同步等待，必须启用 Chunked Prefill 消除抖动。"
  },
  {
    id: "engine",
    title: "Engine (vLLM / SGLang)",
    subtitle: "高并发推理服务框架",
    category: "serving",
    summary: "生产级大模型推理引擎的核心在于消除显存碎片与榨干 GPU 计算时序流水线。",
    whatIsIt: "vLLM 与 SGLang 是目前开源界最主流的高性能 LLM 部署框架。其关键职责是实现 Continuous Batching（迭代级批处理）、统一显存池调度以及高效的多卡并行编排。",
    keyMetrics: "• Running Requests: 并发生成中的流数\n• Waiting Queue: 待执行排队数\n• GPU Memory Utilization: 显存分配水位系数（默认 0.90）",
    symptoms: "• Waiting Queue 快速堆积：并发请求超出显存容纳上限，新请求无法分配初始 KV Block。\n• 吞吐低于预期：未开启 Prefix Caching 或 Batching 调度粒度过粗。",
    moeNotes: "DeepSeek 系列首推 SGLang 与 vLLM 最新版本，已原生集成针对 MLA 的压缩内核与 DeepSeek MoE 的高性能 Triton 算子。"
  },
  {
    id: "workload",
    title: "Workload (Concurrency)",
    subtitle: "流量与负载分布特征",
    category: "traffic",
    summary: "负载特征决定架构瓶颈：Prompt 长度主导算力，Output 长度主导带宽与显存池寿命。",
    whatIsIt: "生产环境中的真实请求模型，由并发度（Concurrency）、请求到达率分布（泊松流）以及 Prompt Token 与 Generate Token 的比例（P/D Ratio）共同构成。",
    keyMetrics: "• Concurrency (并发流数量)\n• Prompt Length (长文本 vs 短问答)\n• Generation Length (短摘要 vs 长代码生成)",
    symptoms: "• 突发长文本冲击导致服务雪崩：数个 16K Prompt 瞬间吞噬全部空闲显存块，引发存量请求驱逐。\n• 低并发下 GPU 利用率极低：Batch Size 不足导致 GPU 无法饱和。",
    moeNotes: "Agent 与多轮对话场景通常具有『长 Prompt、短 Output』特征，极度依赖 Prefix Caching 消除重复 Prefill 开销。"
  },
  {
    id: "scheduler",
    title: "Scheduler (Batching)",
    subtitle: "连续批处理与请求编排",
    category: "serving",
    summary: "Iteration-level 调度打破了传统静态 Batching 必须等待全部序列生成完毕的木桶效应。",
    whatIsIt: "推理引擎的调度器负责在每一个生成步（Iteration）动态决定：哪些新请求进入 Prefill、哪些已有请求继续 Decode、显存不足时驱逐哪一个请求。",
    keyMetrics: "• Batched Tokens per Step: 单步吞吐 Token 总数\n• Preemption Count: 显存不足导致的请求被驱逐/换出次数",
    symptoms: "• Preemption 激增：发生请求重算（Re-computation）或换入换出（Swap），系统端到端吞吐暴跌 70% 以上。\n• Decode 停顿：新来大请求霸占当前迭代的计算资源。",
    moeNotes: "务必启用 `--enable-chunked-prefill`，将大 Prompt 切碎成微块（如 512 或 2048），与存量 Decode 混合调度。"
  },
  {
    id: "kvcache",
    title: "KV Cache (Memory / Reuse)",
    subtitle: "显存管理与前缀缓存",
    category: "memory",
    summary: "KV Cache 是大模型推理时显存消耗的最大变量。从 PagedAttention 到 MLA 是显存优化的核心跃迁。",
    whatIsIt: "自回归生成时缓存前文的 Key 和 Value 矩阵，避免每个 Token 重复计算全历史注意力。采用分页虚拟内存机制（PagedAttention）与基数树前缀缓存（RadixAttention）。",
    keyMetrics: "• KV Cache Hit Rate (前缀复用命中率)\n• Block Allocation % (显存块分配率)\n• KV Cache Bytes per Token (每 Token 显存物理消耗)",
    symptoms: "• 命中率近乎为 0：System Prompt 未对齐或提示词头部存在动态时间戳等干扰项。\n• OOM (Out of Memory)：并发突增时未限制 `max_model_len` 或显存预留不足。",
    moeNotes: "DeepSeek MLA 仅需存储 576 元素/Token/层，显存占用比标准 GQA 节省 3.5 倍以上，极大释放并发承载力。"
  },
  {
    id: "model",
    title: "Model (Attention / MoE)",
    subtitle: "DeepSeek 混合专家微架构",
    category: "model",
    summary: "稀疏激活（Sparse MoE）使得大参数量模型在推理时仅计算激活专家，兼顾效果与算力经济性。",
    whatIsIt: "DeepSeek-V2/V3/R1 采用细粒度路由专家（如 64 个路由专家 + 1 个共享专家，每次动态激活 Top-8）。总参数量虽大，但单 Token 激活参数仅约 21B。",
    keyMetrics: "• Total Params (总权重显存)\n• Active Params per Token (实际参与计算参数量)\n• Expert Load Imbalance (专家激活倾斜率)",
    symptoms: "• 专家负载严重倾斜：某些热门专家被频繁命中，承载该专家的 GPU 成为木桶短板，其他 GPU 发生空等。\n• 显存不足无法加载完整模型：需配合 FP8 权重加载。",
    moeNotes: "MoE 模型在单机多卡上的分布式切分，直接决定了卡间是传输权重还是传输激活值。"
  },
  {
    id: "gpu",
    title: "GPU (SM / HBM)",
    subtitle: "硅片微架构与 Roofline 极限",
    category: "hardware",
    summary: "Decode 阶段永远受制于 HBM 显存带宽，算力利用率（MFU）通常极低，这是访存墙的物理本质。",
    whatIsIt: "GPU 核心硬件由流式多处理器（SM）、张量核心（Tensor Core）与高带宽显存（HBM/GDDR）组成。Roofline 模型精确指明了性能受限于计算还是带宽。",
    keyMetrics: "• SM Utilization % (流处理器算力利用率)\n• Memory Bandwidth Utilization % (MBU, 显存带宽利用率)\n• Machine Balance (机器算力带宽平衡比)",
    symptoms: "• SM 利用率仅 5% 但延迟无法降低：处于单 Batch 或小 Batch 的 Decode 阶段，受限于显存带宽读取权重速度。\n• 显卡过热降频：DCGM 报告 Thermal Throttling。",
    moeNotes: "RTX 4090 (GDDR6X 1TB/s) 与 H100 (HBM3 3.35TB/s) 的核心差距不仅在于 TFLOPS，更在于 3.3 倍的访存带宽差距。"
  },
  {
    id: "multigpu",
    title: "Multi-GPU (NCCL / PCIe)",
    subtitle: "单机 4 卡总线通信与拓扑",
    category: "hardware",
    summary: "在没有 NVLink 的 PCIe 拓扑下，通信带宽是最大的暗坑。EP 与 TP 的选择至关重要。",
    whatIsIt: "4 张显卡通过 PCIe 插槽或 NVLink 互联。NCCL 库负责多卡间的高性能集合通信原语（All-Reduce、All-to-All、All-Gather）。",
    keyMetrics: "• PCIe Tx/Rx Throughput (GB/s)\n• NCCL Bus Bandwidth (有效总线利用率)\n• Topology: PIX (单桥) / SYS (跨 CPU Socket)",
    symptoms: "• Nsight 抓包显示大量通信气泡：TP=4 下频繁的 120+ 次 All-Reduce 阻塞在 PCIe 上。\n• 拓扑跨 NUMA：卡间通信走 QPI/UPI 总线导致时延成倍增加。",
    moeNotes: "4×GPU PCIe 拓扑部署 DeepSeek MoE 推荐优先采用 EP=4（仅在 MoE 层 All-to-All 传输激活值），大幅减轻通信压力。"
  },
  {
    id: "kernel",
    title: "CUDA Kernel (Attention / GEMM)",
    subtitle: "算子极致加速与算力饱和",
    category: "kernel",
    summary: "从 FlashAttention 到 Triton MoE 内核，软件与硬件对齐（IO-Aware）才能消除多余显存读写。",
    whatIsIt: "底层执行矩阵乘法（GEMM）与注意力机制的 CUDA/Triton 代码。FlashAttention 避免了将完整的 $N \times N$ 注意力矩阵写入 HBM，直接在片上 SRAM 完成 Softmax 归约。",
    keyMetrics: "• Kernel Duration (微秒级执行耗时)\n• SRAM Shared Memory 占用率\n• Warp Occupancy (活跃线程束占比)",
    symptoms: "• 算子频繁发生 Register Spill（寄存器溢出到局部内存）：严重降低执行性能。\n• 小 Batch 下 GEMM 无法填满 SM：Tensor Core 效率大打折扣。",
    moeNotes: "MLA 注意力算子需要专用的解码加速内核（如 FlashMLA 或 FlashDecoding），消除解耦 RoPE 带来的额外开销。"
  }
];

export const DIAGNOSTIC_CASES = [
  {
    id: "case-1",
    badge: "PCIe 拓扑 / MoE 并行",
    title: "4×RTX 4090 PCIe 部署 DeepSeek MoE 吞吐严重受限",
    description: "你在 4 张 RTX 4090（PCIe 4.0 x16，无 NVLink）上使用 vLLM 部署 DeepSeek MoE。启动参数设置了 `--tensor-parallel-size 4`。压测时发现 GPU SM 利用率普遍低于 15%，DCGM 显示 PCIe 带宽长期跑在 25GB/s 以上，Token 生成极为迟缓。为什么？如何根治？",
    options: [
      { id: "A", text: "增加并发批次（Batch Size=128），让 GPU 算力强行跑满", correct: false, reason: "错误。在 PCIe 通信瓶颈下，继续盲目增大 Batch 会进一步恶化通信传输体积与队列时延。" },
      { id: "B", text: "改用 EP=4（专家并行）或 TP=2+EP=2，降低高频 All-Reduce 频次，并调优 NCCL_BUFFSIZE", correct: true, reason: "正确！TP=4 每层都需要全局 All-Reduce 同步（单步 120+ 次），在 31.5GB/s 的 PCIe 上会引发极高通信气泡。EP 仅在 MoE 层做 All-to-All 传输激活值，通信开销大幅降低。" },
      { id: "C", text: "关闭所有的 P2P 通信，让数据全部经过 CPU 内存中转", correct: false, reason: "严重错误。走 Host 内存中转通信时延成倍增加，性能将彻底崩溃。" }
    ],
    remediationCommand: `# 调优环境变量与推荐启动方式
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_BUFFSIZE=4194304

# 采用分块预填充与专家并行策略
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat \\
  --tensor-parallel-size 4 \\
  --enable-chunked-prefill true \\
  --max-num-batched-tokens 2048 \\
  --kv-cache-dtype fp8`
  },
  {
    id: "case-2",
    badge: "TTFT 抖动 / Chunked Prefill",
    title: "突发并发下首字延迟 (TTFT) 从 300ms 暴增至 6s",
    description: "线上服务监控显示，在上午流量高峰期，老用户的生成速度（TPOT）依然稳定在 30ms/tok，但新发起请求的首字延迟（TTFT）P99 飙升到了 6 秒以上，用户频繁反馈界面无响应。GPU 显存利用率为 85%，并未发生 OOM。",
    options: [
      { id: "A", text: "新进来的大 Prompt 触发了长时间全量 Prefill 独占，且等待队列发生积压；未开启 Chunked Prefill", correct: true, reason: "正确！传统 Continuous Batching 中，若未开启分块预填充，一个长 Prompt 会独占 GPU 执行完整 Prefill，后续请求只能排队等待，直接导致 TTFT 几何级飙升。" },
      { id: "B", text: "GPU 驱动崩溃，正在尝试降频运行", correct: false, reason: "错误。TPOT 保持在 30ms 说明 Decode 计算正常，属于调度与排队队列问题。" },
      { id: "C", text: "模型参数加载损坏，需重启机器重新下载权重", correct: false, reason: "错误。存量请求正常生成，服务完全健康，纯属调度并发参数不合理。" }
    ],
    remediationCommand: `# 开启 Chunked Prefill 根治长短请求争抢
vllm serve /models/deepseek \\
  --enable-chunked-prefill true \\
  --max-num-batched-tokens 2048 \\
  --max-num-seqs 128`
  },
  {
    id: "case-3",
    badge: "显存管理 / KV Cache 驱逐",
    title: "多轮客服长会话频繁触发 Preemption，系统吞吐腰斩",
    description: "多轮长对话业务上线后，随着平均上下文长度增长到 6K tokens，vLLM 日志中频繁出现大量的 Preempting request 警告，Grafana 大盘显示吞吐量在高峰期暴跌 60%，TTFT 与 TPOT 出现周期性剧烈震荡。",
    options: [
      { id: "A", text: "显存池耗尽导致请求被驱逐并发生重算；应开启 Prefix Caching 并精细调高 gpu-memory-utilization", correct: true, reason: "正确！长上下文消耗大量 KV Cache 块，显存池耗尽时调度器只能选择驱逐（Preempt）请求并丢弃已算 KV。启用前缀缓存（Prefix Caching）可大幅复用历史多轮对话上下文，消除重复计算。" },
      { id: "B", text: "这是正常现象，说明调度器在保护系统不发生 OOM，无需任何干预", correct: false, reason: "错误。频繁 Preemption 会导致灾难性的算力重算浪费，必须通过前缀复用和上下文长度上限进行控制。" },
      { id: "C", text: "将精度强制转换为 FP32 以增强显存分配稳定性", correct: false, reason: "错误。转换为 FP32 会让显存需求翻倍，使得显存瞬间彻底耗尽崩溃。" }
    ],
    remediationCommand: `# 启用前缀缓存并压缩 KV Cache 精度为 FP8
vllm serve /models/deepseek \\
  --enable-prefix-caching \\
  --kv-cache-dtype fp8 \\
  --gpu-memory-utilization 0.94 \\
  --max-model-len 8192`
  },
  {
    id: "case-4",
    badge: "Roofline 访存墙 / 单 Batch 优化",
    title: "Decode 阶段 GPU 算力利用率仅 4%，如何提升速度？",
    description: "研发人员测试单流（Batch Size=1）推理，发现 RTX 4090 的 SM 利用率只有 3.8%，期望通过更强算力（换更高 TFLOPS 的显卡）来让单个用户的生成速度提升 3 倍。这个技术思路对吗？为什么？",
    options: [
      { id: "A", text: "完全正确，单流速度取决于 GPU 核心的 TFLOPS 峰值", correct: false, reason: "错误。单流生成处于极度访存受限（Memory-Bound）区间，算力密度不足 2 FLOPs/Byte，TFLOPS 再高也是空转。" },
      { id: "B", text: "思路错误。Decode 自回归每次必须将几十 GB 权重扫过 HBM，单流速度由显存带宽决定；欲提升单流速度应使用投机解码（Speculative Decoding）或量化", correct: true, reason: "正确！单流速度受限于 HBM 读取权重带宽。要突破物理限制，要么采用投机解码（小模型先草拟多个 Token 一并验证），要么采用 W8A8/FP8 压缩权重体积减半传输量。" },
      { id: "C", text: "应当关闭 Tensor Core，改用纯 CUDA Core 计算", correct: false, reason: "错误。关闭 Tensor Core 只会雪上加霜。" }
    ],
    remediationCommand: `# 方案一：采用 FP8 权重（减少 50% 访存读取量）
# 方案二：配置投机采样（Speculative Decoding）
vllm serve /models/deepseek-v2-lite \\
  --speculative-model /models/deepseek-tiny-draft \\
  --num-speculative-tokens 5 \\
  --quantization fp8`
  },
  {
    id: "case-5",
    badge: "Nsight Profiling / 通信排查",
    title: "Nsight Systems 抓包发现 ncclKernel 占用 65% 时间",
    description: "在 4 卡机器上进行推理 profile，导出的 `.nsys-rep` 时序图中，发现每个 step 都有漫长的 `ncclKernel_AllReduce_RING_LL` 耗时，期间 GPU Compute 流完全静止（出现大片空泡）。如何定位该故障？",
    options: [
      { id: "A", text: "通过 nvidia-smi topo -m 排查 PCIe 拓扑是否跨了 CPU Root Complex（SYS），并检查 PCIe 是否协商在 Gen4 x16", correct: true, reason: "正确！当物理卡插在不同的 CPU 插槽上或 PCIe 降级为 Gen1/Gen2 时，跨卡带宽会从 31GB/s 暴跌至几 GB/s。此时任何集合通信都将形成致命时钟气泡。" },
      { id: "B", text: "立刻重新编译操作系统内核与 Python 运行库", correct: false, reason: "错误。毫无针对性，应先用硬件拓扑工具和 NCCL 测试套件排查物理链路协商速率。" },
      { id: "C", text: "删除 Nsight 工具，因为 Profiler 本身引入了测量扰动", correct: false, reason: "错误。虽然 Profiling 有微小开销，但 65% 的通信占比是硬件链路或拓扑严重失配的明确证据。" }
    ],
    remediationCommand: `# 1. 查看物理拓扑与卡间亲和度
nvidia-smi topo -m

# 2. 检查 PCIe 链路协商速度（确保为 16GT/s 宽 x16）
lspci -vvv -d 10de: | grep -E "LnkCap|LnkSta"

# 3. 运行 NCCL 性能基准测试验证卡间带宽
nccl-tests/build/all_reduce_perf -b 8M -e 128M -f 2 -g 4`
  }
];

export const ROADMAP_STEPS = [
  {
    step: 1,
    title: "① Prefill vs Decode",
    summary: "计算密集 (Compute-Bound) vs 访存密集 (Memory-Bound) 的本质物理区隔",
    concept: "Prefill 处理全量提示词，生成首字，激活大尺度 GEMM 运算；Decode 每次仅处理 1 个 Token，自回归迭代加载几十 GB 权重只做向量-矩阵乘法（GEMV）。",
    hwMetrics: "• Prefill: SM 利用率可达 70%~90%，Tensor Core 满载。\n• Decode: SM 利用率常 < 10%，HBM 带宽利用率（MBU）跑满 70% 以上。",
    tuningTakeaway: "单流 Decode 的生成速度只取决于显存带宽（HBM GB/s）除以权重体积；一味堆砌 TFLOPS 算力无法提升单流速度。"
  },
  {
    step: 2,
    title: "② TTFT / TPOT / tok/s",
    summary: "端到端交互延迟量化与黄金 SLA 指标定义",
    concept: "TTFT = 排队时延 + Prefill 时延；TPOT = 单 Token 生成耗时；ITL (Inter-Token Latency) 反映流式打字机效果的均匀性。",
    hwMetrics: "• 优秀体验指标：TTFT P95 < 800ms，TPOT < 30ms/tok (约 35 tok/s)，ITL P99 抖动 < 40ms。",
    tuningTakeaway: "慢必须先量化：用户感觉卡，必须先通过 Prometheus 拆解是卡在队列排队、卡在长 Prefill，还是卡在 Decode 显存带宽争抢。"
  },
  {
    step: 3,
    title: "③ KV Cache (MLA & Paged)",
    summary: "PagedAttention 分页机制与 DeepSeek MLA 低秩压缩革命",
    concept: "传统 MHA/GQA 随着上下文加长线性吃爆显存；DeepSeek MLA 引入潜在向量压缩，单 Token 仅需缓存 576 元素，显存压缩至 GQA 的 28%。",
    hwMetrics: "• 显存公式：MLA 字节/token/层 = (512 + 64) * dtype_bytes = 576 字节 (FP8) 或 1152 字节 (FP16)。",
    tuningTakeaway: "4 卡 24GB 环境下，MLA 允许承载 4 倍以上的上下文并发，从根源上消除 OOM 和 Preemption 驱逐。"
  },
  {
    step: 4,
    title: "④ HBM / Tensor Core",
    summary: "GPU 硅片微架构、Roofline 模型与机器算力平衡比",
    concept: "Machine Balance 拐点公式：Ridge Point = Peak TFLOPS / Peak Bandwidth。Decode 算力密度仅 2~8 FLOPs/Byte，远远落入访存受限区。",
    hwMetrics: "• RTX 4090 平衡拐点: 164 FLOPs/Byte；H100 SXM: 295 FLOPs/Byte。",
    tuningTakeaway: "通过连续批处理（Continuous Batching）增大并发 Batch Size，让多个流共享单次权重载入，将算力密度推高至拐点！"
  },
  {
    step: 5,
    title: "⑤ TP / EP / DP 并行",
    summary: "单机 4 卡分布式并行策略：Dense 模型 vs MoE 模型的并行博弈",
    concept: "Dense 模型通常采用张量并行（TP=4）；而 MoE 模型在无 NVLink 的 PCIe 拓扑下，专家并行（EP=4）通信量远小于频繁 All-Reduce 的 TP=4。",
    hwMetrics: "• TP=4 每层需 All-Reduce 同步；EP=4 仅在 MoE 层做 All-to-All 传输 Token 激活值。",
    tuningTakeaway: "PCIe 拓扑下优先采用 EP 专家并行或 TP=2+EP=2 混合并行，规避高频同步气泡。"
  },
  {
    step: 6,
    title: "⑥ NCCL / PCIe 瓶颈",
    summary: "PCIe 31.5 GB/s 带宽墙、NUMA 拓扑与 NCCL 环形缓冲区调优",
    concept: "PCIe Gen4 x16 双向仅 31.5 GB/s（对比 NVLink 900 GB/s）。跨 CPU Socket（SYS 拓扑）通信还会受制于 UPI/QPI 总线时延劣化。",
    hwMetrics: "• 运行 `nvidia-smi topo -m` 查看拓扑关系（PIX vs SYS）；检查 `DCGM_FI_DEV_PCIE_TX_THROUGHPUT`。",
    tuningTakeaway: "设置 `NCCL_BUFFSIZE=4194304` 扩大环形缓冲；设置 `NCCL_P2P_DISABLE=0` 启用 PCIe 直通交换。"
  },
  {
    step: 7,
    title: "⑦ Attention / MoE kernels",
    summary: "FlashAttention-2/3、FlashDecoding 与 Triton 算子内核加速",
    concept: "FlashAttention 利用片上 SRAM 做分块 Softmax，避免把 $N \times N$ 注意力矩阵写入 HBM；FlashDecoding 对长上下文进行并行化归约。",
    hwMetrics: "• Kernel Duration 微秒级耗时、SRAM Shared Memory 利用率、Warp 活跃度。",
    tuningTakeaway: "MLA 需要专门融合算子（FlashMLA）；MoE Gating 路由与 Top-K 选通必须由高度优化的 Triton/CUDA 算子实现。"
  },
  {
    step: 8,
    title: "⑧ vLLM / SGLang 生产参数",
    summary: "连续批处理、分块预填充 (Chunked Prefill) 与前缀基数树缓存",
    concept: "Chunked Prefill 消除长短请求争抢；RadixAttention 自动识别跨会话、跨多轮请求的共享 System Prompt 并复用 KV。",
    hwMetrics: "• `vllm:prefix_cache_hit_rate` 前缀命中率提升至 60%+ 时，平均 TTFT 降低 85%。",
    tuningTakeaway: "必开双剑客：`--enable-chunked-prefill true` + `--enable-prefix-caching`。"
  },
  {
    step: 9,
    title: "⑨ Prometheus / Grafana 监控",
    summary: "生产级全栈可观测性：DCGM 硬件指标与推理引擎状态聚合",
    concept: "监控大盘三大黄金看板：1. 延迟分布 (P50/P95/P99 TTFT 与 TPOT)；2. 显存与队列水位；3. PCIe 总线与 GPU 功耗温控。",
    hwMetrics: "• 告警规则：`vllm:gpu_cache_usage_factor > 0.95` 时紧急告警，防止请求被驱逐重算引发雪崩。",
    tuningTakeaway: "配合本文档提供的 Grafana Dashboard JSON 与 Prometheus Rules，开箱即用搭建生产监控看板。"
  },
  {
    step: 10,
    title: "⑩ Nsight Systems 抓包",
    summary: "NVIDIA Nsight 真实时序捕获：定位通信气泡与 Host-Device 阻塞",
    concept: "使用 `nsys profile` 捕获 CUDA Kernel、NCCL 通信与 OS Runtime。排查 Step 之间的时钟空泡与未融合算子。",
    hwMetrics: "• Timeline 中的 `ncclKernel_AllToAll` / `AllReduce` 耗时比例；CUDA Stream 并发利用率。",
    tuningTakeaway: "真实调优必须以时序 Trace 为准，告别拍脑袋猜瓶颈。"
  }
];
