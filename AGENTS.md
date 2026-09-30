# AGENTS.md — GPU LLM Performance Tuning Repository Guide

## 1. 仓库定位与核心主题
本项目专注于 **GPU + 大语言模型 (LLM) 推理部署与全链路性能调优**。
核心全景链路：
`用户请求 (Latency) → 推理引擎 (vLLM/SGLang) → KV Cache 管理 → GPU 显存/算力 (SM/HBM) → 4×GPU 多卡拓扑 (NCCL/PCIe/NVLink) → 底层算子 (CUDA Kernel/Attention/MoE) → 观测调优 (Prometheus/Grafana/Nsight)`。

重点聚焦工业级高频场景：**DeepSeek MoE 架构在 4×GPU (含无 NVLink 的 PCIe 拓扑) 下的推理优化与瓶颈定位**。

---

## 2. 目录规范
- `index.html`, `style.css`, `app.js`: 零依赖纯前端交互式学习仪表盘与诊断实验室。支持直接浏览器双击预览或通过 Cloudflare Pages / Python http.server 启动。
- `docs/`: 结构化沉淀系统全景、指标体系、诊断决策树、引擎配置与实战实验。
- `scripts/`: 可直接运行的计算器与自动化调优工具（如 KV Cache 内存计算器、Roofline 模型推导、Nsight Systems 抓包脚本）。
- `configs/`: 生产级监控规则模板（Prometheus alert 规则、Grafana Dashboard 配置等）。

---

## 3. 开发与运维规范
- 遵循 **Ponytail 极简原则**：交互式界面采用无构建步骤（Zero-build）的原生现代 Web 技术（ES6+、CSS Variables、Vanilla JS），杜绝臃肿的依赖黑洞。
- Python 脚本仅依赖标准库（`argparse`, `math`, `sys`），即开即用。
- 遵循 Local-Mac 规范：本仓库归属 `haossll666` 账号体系，部署目标为 Cloudflare Pages。
