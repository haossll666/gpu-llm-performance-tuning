# GPU LLM Deployment & Performance Tuning

> **从用户请求一路下钻到 CUDA Kernel**：  
> **GPU 部署 LLM → DeepSeek MoE → 4×GPU → KV Cache → vLLM/SGLang → NCCL/PCIe → Prometheus/Grafana → Nsight**

[![Cloudflare Pages](https://img.shields.io/badge/Deploy-Cloudflare%20Pages-F38020?logo=cloudflare)](https://pages.cloudflare.com/)
[![Apple HIG](https://img.shields.io/badge/Design-Apple%20HIG%20iPhone%2016-000000?logo=apple)](https://developer.apple.com/design/human-interface-guidelines)
[![DeepSeek MLA](https://img.shields.io/badge/Arch-DeepSeek%20MoE%20%2B%20MLA-0080FF)](https://github.com/deepseek-ai)
[![Zero Dependency](https://img.shields.io/badge/Stack-Zero--Build%20Vanilla%20Web-brightgreen)](https://developer.mozilla.org/)

---

## 🌟 项目亮点与交互体验

本项目是一个面向工业级生产实战的 **GPU + 大语言模型 (LLM) 推理部署与全链路性能工程知识库与交互式仪表盘**。

- 📱 **Apple HIG 级高保真移动交互**：专为 **iPhone 16**（灵动岛、安全边距 Safe Area、>=44pt 触控、流体过渡动效）定制，支持全景拓扑图下钻与交互式计算。
- ⚡ **聚焦核心高频场景**：重点拆解 **DeepSeek MoE（细粒度路由 + MLA 低秩压缩）** 在 **4×GPU (无 NVLink 的 PCIe 拓扑)** 硬件底座上的物理性能边界与调优避坑。
- 🧮 **双轨交互计算器**：动态测算不同模型架构（MLA vs GQA）的单 Token / 完整序列显存开销与理论最大并发承载；基于 Roofline 推导硬件平衡点（Machine Balance）。
- 🧪 **5 大工业级故障诊断实验室**：内置真实的 4×GPU 生产事故案例，支持即时选择、判定并输出根因诊断与一线修复脚本。
- 📦 **遵循 Ponytail 极简原则**：零构建步骤（Zero-build step），原生现代标准 ES6+ / CSS Variables，双击或任意静态服务器即可秒级启动。

---

## 🚀 极速上手 (Quick Start)

### 1. 本地启动交互式学习仪表盘
无需 `npm install` 或复杂的构建流程：

```bash
# 方式 A：Python 一键启动
python3 -m http.server 8080

# 方式 B：使用任意浏览器直接双击打开 index.html 即可！
```

访问 `http://localhost:8080`，即可体验专为 iPhone 16 优化的高保真移动端交互视图。

### 2. 独立运行核心测算与分析工具
项目在 `scripts/` 下提供了无外部三方依赖的标准 Python 3 工具：

```bash
# 1. 测算 DeepSeek-V3 (MLA) 在 4 卡 24GB 显存下的 KV Cache 与并发容量
python3 scripts/kv_cache_calculator.py --arch deepseek-v3 --seq-len 4096 --batch-size 16 --kv-dtype fp8

# 2. 测算 RTX 4090 / L40S 的 Roofline 机器平衡点与 4×GPU PCIe All-to-All 通信时延
python3 scripts/roofline_analyzer.py --gpu rtx4090 --batch-size 8 --prompt-len 2048

# 3. 运行自动化单元测试套件
python3 tests/test_algorithms.py
node tests/test_mobile_viewport.mjs
```

---

## 🗺️ 系统知识体系目录 (Documentation)

- [01. 全景地图与建议学习路线](docs/01-panorama-and-roadmap.md)
- [02. LLM 推理关键指标与 Roofline 极限分析](docs/02-metrics-and-roofline.md)
- [03. 性能瓶颈排查矩阵与实战决策树](docs/03-bottleneck-diagnosis-matrix.md)
- [04. DeepSeek MoE 在 4×GPU (PCIe) 落地与性能调优指南](docs/04-deepseek-moe-4xgpu-pcie.md)
- [05. vLLM / SGLang / Prometheus / Nsight 工具链指南](docs/05-toolchains-guide.md)

---

## ☁️ Cloudflare Pages 部署说明

本项目已适配 Cloudflare Pages 部署，绑定至 `haossll666` 账号：

```bash
# 部署至 Cloudflare Pages 生产环境
npx wrangler pages deploy . --project-name=gpu-llm-performance-tuning
```
