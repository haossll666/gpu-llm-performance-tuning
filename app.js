// ==============================================================================
// App Logic & Interactive Handlers (Zero-dependency Vanilla ES6)
// ==============================================================================

import { PIPELINE_NODES, DIAGNOSTIC_CASES, ROADMAP_STEPS } from './data.js';

let currentNodeId = 'latency';

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  renderTopologyNodes();
  selectNode(currentNodeId);
  initKVCacheCalculator();
  initToolchainGenerator();
  renderDiagnosticCases();
  initRooflineVisualizer();
  initMoESimulator();
  initRoadmapModal();
});

// 1. Navigation Tab Switching
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetView = btn.dataset.tab;
      
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.querySelectorAll('.view-section').forEach(view => {
        view.classList.remove('active');
      });

      const activeView = document.getElementById(targetView);
      if (activeView) {
        activeView.classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  });
}

// 2. Render Topology Nodes
function renderTopologyNodes() {
  const container = document.getElementById('topology-nodes');
  if (!container) return;

  container.innerHTML = PIPELINE_NODES.map(node => `
    <div class="flow-node ${node.id === currentNodeId ? 'selected' : ''}" data-id="${node.id}" tabindex="0">
      <div class="node-title">${node.title.split(' ')[0]}</div>
      <div class="node-sub">${node.title.split(' ')[1] || ''}</div>
    </div>
  `).join('');

  container.querySelectorAll('.flow-node').forEach(elem => {
    elem.addEventListener('click', () => {
      selectNode(elem.dataset.id);
    });
  });
}

// 3. Select and Inspect Node
export function selectNode(nodeId) {
  currentNodeId = nodeId;
  const node = PIPELINE_NODES.find(n => n.id === nodeId);
  if (!node) return;

  // Update node selection visual state
  document.querySelectorAll('.flow-node').forEach(elem => {
    elem.classList.toggle('selected', elem.dataset.id === nodeId);
  });

  // Render Inspector
  const inspector = document.getElementById('node-inspector');
  if (!inspector) return;

  inspector.innerHTML = `
    <div class="inspector-header">
      <div>
        <div class="inspector-title">${node.title}</div>
        <div style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); margin-top: 2px;">${node.subtitle}</div>
      </div>
      <span class="badge-4xgpu">${node.category.toUpperCase()}</span>
    </div>
    
    <div class="inspector-summary">
      <strong>核心洞见：</strong>${node.summary}
    </div>

    <div class="inspector-grid">
      <div class="inspect-box box-what">
        <h4>💡 它是什么 (Architecture & Concept)</h4>
        <p>${node.whatIsIt}</p>
      </div>

      <div class="inspect-box box-metrics">
        <h4>📊 关键指标 (Metrics & SLAs)</h4>
        <pre>${node.keyMetrics}</pre>
      </div>

      <div class="inspect-box box-symptoms">
        <h4>⚠️ 异常现象与根因 (Symptoms & Causes)</h4>
        <p>${node.symptoms}</p>
      </div>

      <div class="inspect-box box-moe">
        <h4>🚀 4×GPU DeepSeek 落地要点</h4>
        <p>${node.moeNotes}</p>
      </div>
    </div>
  `;
}

// Make selectNode accessible for pills
window.selectNode = selectNode;

// 4. Interactive KV Cache & Concurrency Calculator
function initKVCacheCalculator() {
  const archSelect = document.getElementById('calc-arch');
  const seqInput = document.getElementById('calc-seq');
  const batchInput = document.getElementById('calc-batch');
  const dtypeSelect = document.getElementById('calc-dtype');
  const gpusInput = document.getElementById('calc-gpus');
  const vramInput = document.getElementById('calc-vram');

  if (!archSelect) return;

  function update() {
    const arch = archSelect.value;
    const seqLen = parseInt(seqInput.value);
    const batchSize = parseInt(batchInput.value);
    const dtype = dtypeSelect.value;
    const numGpus = parseInt(gpusInput.value);
    const vramPerGpu = parseFloat(vramInput.value);

    // Update labels
    document.getElementById('val-seq').textContent = `${seqLen} tokens`;
    document.getElementById('val-batch').textContent = `${batchSize} 流`;
    document.getElementById('val-gpus').textContent = `${numGpus} 张`;
    document.getElementById('val-vram').textContent = `${vramPerGpu} GB`;

    const bytesPerElem = dtype === 'fp8' ? 1 : 2;
    let bytesPerToken = 0;
    let weightsGb = 0;

    if (arch === 'deepseek-v3-mla') {
      // 61 layers, 576 elements per token per layer
      bytesPerToken = 61 * (512 + 64) * bytesPerElem;
      weightsGb = 340; // FP8 quantized weight across cluster
    } else if (arch === 'deepseek-v2-lite') {
      bytesPerToken = 27 * (512 + 64) * bytesPerElem;
      weightsGb = 32;
    } else if (arch === 'llama3-70b-gqa') {
      // 80 layers, 2 * 8 kv_heads * 128 head_dim = 2048 elements
      bytesPerToken = 80 * 2048 * bytesPerElem;
      weightsGb = 70; // 70B FP8 weights
    } else {
      // Standard MHA
      bytesPerToken = 32 * (2 * 32 * 128) * bytesPerElem;
      weightsGb = 16;
    }

    const kbPerToken = (bytesPerToken / 1024).toFixed(2);
    const mbPerSeq = ((bytesPerToken * seqLen) / (1024 * 1024)).toFixed(1);
    
    const totalVram = numGpus * vramPerGpu;
    const usableKvPool = Math.max(0, (totalVram * 0.90) - (weightsGb / numGpus * numGpus)); // assuming cluster fits weights
    
    let maxConcurrency = 0;
    if (usableKvPool > 0) {
      const maxTokens = (usableKvPool * 1024 * 1024 * 1024) / bytesPerToken;
      maxConcurrency = Math.floor(maxTokens / seqLen);
    }

    document.getElementById('res-tok-size').textContent = `${kbPerToken} KB`;
    document.getElementById('res-seq-size').textContent = `${mbPerSeq} MB`;
    document.getElementById('res-kv-pool').textContent = `${usableKvPool.toFixed(1)} GB`;
    document.getElementById('res-concurrency').textContent = `${maxConcurrency} Streams`;
  }

  [archSelect, seqInput, batchInput, dtypeSelect, gpusInput, vramInput].forEach(el => {
    el.addEventListener('input', update);
  });

  update();
}

// 5. Interactive Toolchain Command Builder
function initToolchainGenerator() {
  const engineSelect = document.getElementById('tool-engine');
  const chunkedToggle = document.getElementById('tool-chunked');
  const prefixToggle = document.getElementById('tool-prefix');
  const fp8Toggle = document.getElementById('tool-fp8');
  const codeBox = document.getElementById('tool-command-preview');

  if (!engineSelect || !codeBox) return;

  function updateCommand() {
    const isVllm = engineSelect.value === 'vllm';
    const chunked = chunkedToggle.checked;
    const prefix = prefixToggle.checked;
    const fp8 = fp8Toggle.checked;

    if (isVllm) {
      codeBox.textContent = `vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat \\
  --tensor-parallel-size 4 \\
  --gpu-memory-utilization 0.92 \\
  --max-model-len 8192 \\${chunked ? '\n  --enable-chunked-prefill true \\\n  --max-num-batched-tokens 2048 \\' : ''}${prefix ? '\n  --enable-prefix-caching \\' : ''}${fp8 ? '\n  --kv-cache-dtype fp8 \\' : ''}
  --port 8000`;
    } else {
      codeBox.textContent = `python3 -m sglang.launch_server \\
  --model-path deepseek-ai/DeepSeek-V2-Lite-Chat \\
  --tp-size 4 \\
  --mem-fraction-static 0.90 \\
  --context-length 8192 \\${chunked ? '\n  --chunked-prefill-size 2048 \\' : ''}${prefix ? '\n  --enable-radix-attention \\' : ''}${fp8 ? '\n  --kv-cache-dtype fp8 \\' : ''}
  --port 30000`;
    }
  }

  [engineSelect, chunkedToggle, prefixToggle, fp8Toggle].forEach(el => {
    el.addEventListener('change', updateCommand);
  });

  updateCommand();
}

// 6. Render Diagnostic Lab Cases & Evaluation
function renderDiagnosticCases() {
  const container = document.getElementById('cases-container');
  if (!container) return;

  container.innerHTML = DIAGNOSTIC_CASES.map(c => `
    <div class="case-card" id="card-${c.id}">
      <span class="case-badge">${c.badge}</span>
      <h3 class="case-title">${c.title}</h3>
      <p class="case-desc">${c.description}</p>
      
      <div class="case-options">
        ${c.options.map(opt => `
          <button class="option-btn" data-case="${c.id}" data-opt="${opt.id}">
            <span class="option-prefix">[${opt.id}]</span>
            <span>${opt.text}</span>
          </button>
        `).join('')}
      </div>

      <div class="feedback-box" id="feedback-${c.id}"></div>
    </div>
  `).join('');

  container.querySelectorAll('.option-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const caseId = btn.dataset.case;
      const optId = btn.dataset.opt;
      handleCaseAnswer(caseId, optId);
    });
  });
}

function handleCaseAnswer(caseId, optId) {
  const caseItem = DIAGNOSTIC_CASES.find(c => c.id === caseId);
  if (!caseItem) return;

  const card = document.getElementById(`card-${caseId}`);
  const feedback = document.getElementById(`feedback-${caseId}`);
  const option = caseItem.options.find(o => o.id === optId);

  // Reset styles
  card.querySelectorAll('.option-btn').forEach(b => {
    b.classList.remove('selected-correct', 'selected-incorrect');
  });

  const clickedBtn = card.querySelector(`[data-opt="${optId}"]`);
  if (option.correct) {
    clickedBtn.classList.add('selected-correct');
    feedback.innerHTML = `
      <div style="color: var(--accent-emerald); font-weight: 700; margin-bottom: 6px;">🎉 诊断正确！</div>
      <p>${option.reason}</p>
      <div style="margin-top: 10px; font-weight: 600; color: var(--accent-cyan);">推荐修复命令 / 调优范式：</div>
      <pre class="code-block">${caseItem.remediationCommand}</pre>
    `;
    feedback.style.borderLeftColor = 'var(--accent-emerald)';
  } else {
    clickedBtn.classList.add('selected-incorrect');
    feedback.innerHTML = `
      <div style="color: var(--accent-rose); font-weight: 700; margin-bottom: 6px;">❌ 诊断偏差</div>
      <p>${option.reason}</p>
      <div style="margin-top: 8px; color: var(--text-muted); font-size: 12px;">提示：思考该场景下的物理瓶颈是算力（TFLOPS）、显存带宽（HBM）、通信总线（PCIe）还是调度冲突？</div>
    `;
    feedback.style.borderLeftColor = 'var(--accent-rose)';
  }

  feedback.classList.add('show');
}

// 7. Interactive SVG Roofline Visualizer
const ROOFLINE_GPUS = {
  rtx4090: { name: 'RTX 4090', fp16: 165.2, fp8: 330.0, bw: 1.008, pcie: 31.5 },
  l40s: { name: 'L40S', fp16: 366.0, fp8: 733.0, bw: 0.864, pcie: 31.5 },
  'a100-sxm': { name: 'A100 SXM', fp16: 312.0, fp8: 312.0, bw: 2.039, pcie: 600 },
  'h100-sxm': { name: 'H100 SXM', fp16: 989.0, fp8: 1978.0, bw: 3.350, pcie: 900 }
};

function initRooflineVisualizer() {
  const gpuSelect = document.getElementById('roofline-gpu');
  const precSelect = document.getElementById('roofline-prec');
  if (!gpuSelect) return;

  function render() {
    const gpuKey = gpuSelect.value;
    const prec = precSelect.value;
    const gpu = ROOFLINE_GPUS[gpuKey];
    const peakTflops = prec === 'fp8' ? gpu.fp8 : gpu.fp16;
    const bw = gpu.bw;
    const ridge = peakTflops / bw;

    const svg = document.getElementById('roofline-svg');
    if (!svg) return;

    const plotW = 290, plotH = 130, padL = 55, padT = 20, padB = 150;

    function toX(val) {
      const logV = Math.max(0, Math.min(3, Math.log10(Math.max(1, val))));
      return padL + (logV / 3) * plotW;
    }
    function toY(val) {
      const logV = Math.max(0, Math.min(3.35, Math.log10(Math.max(1, val))));
      return padB - (logV / 3.35) * plotH;
    }

    const ridgeX = toX(ridge);
    const peakY = toY(peakTflops);
    const startX = toX(1);
    const startY = toY(1 * bw);
    const endX = toX(1000);

    const decodeIntensity = prec === 'fp8' ? 16 : 8;
    const decodeTflops = Math.min(peakTflops, decodeIntensity * bw);
    const decX = toX(decodeIntensity);
    const decY = toY(decodeTflops);

    const prefillIntensity = 200;
    const prefillTflops = Math.min(peakTflops, prefillIntensity * bw);
    const prefX = toX(prefillIntensity);
    const prefY = toY(prefillTflops);

    svg.innerHTML = `
      <!-- Axes -->
      <line x1="${padL}" y1="${padB}" x2="${padL + plotW}" y2="${padB}" stroke="#334155" stroke-width="1.5" />
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padB}" stroke="#334155" stroke-width="1.5" />
      
      <!-- Y Labels -->
      <text x="${padL - 8}" y="${padT + 10}" fill="#94a3b8" font-size="9" text-anchor="end" font-family="monospace">${peakTflops.toFixed(0)}T</text>
      <text x="${padL - 8}" y="${padB}" fill="#94a3b8" font-size="9" text-anchor="end" font-family="monospace">1T</text>
      <text x="16" y="${padT + 65}" fill="#64748b" font-size="9" text-anchor="middle" transform="rotate(-90, 16, ${padT + 65})">TFLOPS</text>

      <!-- X Labels -->
      <text x="${padL}" y="${padB + 16}" fill="#94a3b8" font-size="9" text-anchor="middle" font-family="monospace">1</text>
      <text x="${toX(10)}" y="${padB + 16}" fill="#94a3b8" font-size="9" text-anchor="middle" font-family="monospace">10</text>
      <text x="${toX(100)}" y="${padB + 16}" fill="#94a3b8" font-size="9" text-anchor="middle" font-family="monospace">100</text>
      <text x="${padL + plotW}" y="${padB + 16}" fill="#94a3b8" font-size="9" text-anchor="middle" font-family="monospace">1000</text>
      <text x="${padL + plotW / 2}" y="${padB + 28}" fill="#64748b" font-size="9" text-anchor="middle">算力密度 (FLOPs/Byte)</text>

      <!-- Ridge Line -->
      <line x1="${ridgeX}" y1="${peakY}" x2="${ridgeX}" y2="${padB}" stroke="#818cf8" stroke-dasharray="3,3" stroke-width="1" />
      <text x="${ridgeX}" y="${padB - 6}" fill="#818cf8" font-size="8" font-family="monospace" text-anchor="middle">拐点 ${ridge.toFixed(0)}</text>

      <!-- Roofline Ceilings -->
      <line x1="${startX}" y1="${startY}" x2="${ridgeX}" y2="${peakY}" stroke="#38bdf8" stroke-width="2.5" />
      <line x1="${ridgeX}" y1="${peakY}" x2="${endX}" y2="${peakY}" stroke="#38bdf8" stroke-width="2.5" />

      <!-- Decode Dot -->
      <circle cx="${decX}" cy="${decY}" r="5" fill="#f43f5e" />
      <text x="${decX + 6}" y="${decY - 6}" fill="#fecdd3" font-size="9" font-weight="700">Decode (访存墙)</text>

      <!-- Prefill Dot -->
      <circle cx="${prefX}" cy="${prefY}" r="5" fill="#34d399" />
      <text x="${prefX - 10}" y="${prefY - 8}" fill="#a7f3d0" font-size="9" font-weight="700">Prefill (算力区)</text>
    `;

    const statusBox = document.getElementById('roofline-status-box');
    if (statusBox) {
      statusBox.innerHTML = `
        <div><strong>${gpu.name} (${prec.toUpperCase()}) 分析结论：</strong></div>
        <div>• 机器平衡点 (Machine Balance): <strong>${ridge.toFixed(1)} FLOPs/Byte</strong></div>
        <div>• <span style="color: #f43f5e; font-weight: 700;">Decode 阶段</span>：算力密度仅 ~${decodeIntensity} FLOPs/Byte，<strong>仅发挥约 ${((decodeTflops / peakTflops) * 100).toFixed(1)}% 峰值算力</strong>，瓶颈在显存带宽！</div>
        <div>• <span style="color: #34d399; font-weight: 700;">Prefill 阶段</span>：算力密度 ~200 FLOPs/Byte，充分饱和 Tensor Core 算力。</div>
      `;
    }
  }

  gpuSelect.addEventListener('change', render);
  precSelect.addEventListener('change', render);
  render();
}

// 8. 4xGPU MoE All-to-All Token Dispatch Simulator
function initMoESimulator() {
  const batchSlider = document.getElementById('moe-sim-batch');
  const batchVal = document.getElementById('moe-sim-batch-val');
  const topkSelect = document.getElementById('moe-sim-topk');
  const runBtn = document.getElementById('btn-run-moe-sim');
  const outBox = document.getElementById('moe-sim-output');

  if (!batchSlider || !runBtn) return;

  batchSlider.addEventListener('input', () => {
    batchVal.textContent = `${batchSlider.value} 流`;
  });

  runBtn.addEventListener('click', () => {
    const B = parseInt(batchSlider.value);
    const K = parseInt(topkSelect.value);
    const hiddenSize = 7168; // DeepSeek hidden dim
    const numGpus = 4;
    const remoteRatio = (numGpus - 1) / numGpus; // 75% tokens dispatched to remote GPUs

    const totalActiveTokens = B * K;
    const remoteTokensPerGpu = Math.round(totalActiveTokens * remoteRatio / numGpus);
    const bytesPerLayer = 2 * (remoteTokensPerGpu * numGpus) * hiddenSize * 2; // Dispatch + Combine in FP16
    const mbPerLayer = bytesPerLayer / (1024 * 1024);
    const totalMb = mbPerLayer * 60; // 60 MoE layers

    const pcieBwEffective = 31.5 * 0.70; // 22 GB/s real throughput
    const pcieLatencyMs = ((totalMb / (pcieBwEffective * 1024)) * 1000).toFixed(2);

    // Equivalent TP All-Reduce communication time
    const tpTotalMb = (2 * remoteRatio * B * hiddenSize * 2 * 122) / (1024 * 1024);
    const tpLatencyMs = ((tpTotalMb / (pcieBwEffective * 1024)) * 1000).toFixed(2);

    outBox.innerHTML = `
      <div style="color: var(--accent-cyan); font-weight: 700; margin-bottom: 8px;">📊 4×GPU PCIe All-to-All 路由开销测算完成 (Batch=${B}, Top-${K})</div>
      <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 10px;">
        <div style="background: rgba(0,0,0,0.4); padding: 8px 10px; border-radius: var(--radius-sm);">
          <div style="font-size: 11px; color: var(--text-muted);">EP=4 单步 All-to-All 传输量</div>
          <div style="font-size: 16px; font-weight: 700; color: var(--accent-emerald);">${totalMb.toFixed(2)} MB</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">预计耗时: ~${pcieLatencyMs} ms</div>
        </div>
        <div style="background: rgba(0,0,0,0.4); padding: 8px 10px; border-radius: var(--radius-sm);">
          <div style="font-size: 11px; color: var(--text-muted);">TP=4 122次 All-Reduce 传输量</div>
          <div style="font-size: 16px; font-weight: 700; color: var(--accent-rose);">${tpTotalMb.toFixed(2)} MB</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">同步时延: ~${tpLatencyMs} ms (频繁阻塞)</div>
        </div>
      </div>
      <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
        💡 <strong>拓扑建议：</strong> 4 卡 PCIe 无 NVLink 部署下，EP 仅在 MoE 层发生 All-to-All 且只传输 Token 激活向量，时钟气泡显著优于全量切分权重的 TP=4。
      </div>
    `;
  });
}

// 9. Interactive Roadmap Step Detail Modal
function initRoadmapModal() {
  const modal = document.getElementById('roadmap-modal');
  const titleEl = document.getElementById('modal-step-title');
  const contentEl = document.getElementById('modal-step-content');
  const closeBtn = document.getElementById('btn-close-modal');

  if (!modal) return;

  closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });

  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.style.display = 'none';
  });

  // Attach click to all roadmap pills
  document.querySelectorAll('.roadmap-pills .pill-item').forEach((pill, idx) => {
    pill.addEventListener('click', (e) => {
      e.stopPropagation();
      const stepData = ROADMAP_STEPS[idx];
      if (!stepData) return;

      titleEl.textContent = stepData.title;
      contentEl.innerHTML = `
        <div style="background: rgba(56, 189, 248, 0.08); padding: 10px 12px; border-radius: var(--radius-sm); border-left: 3px solid var(--accent-cyan); margin-bottom: 12px; color: #fff;">
          <strong>核心定义：</strong>${stepData.summary}
        </div>
        <div style="margin-bottom: 12px;">
          <h4 style="color: var(--accent-cyan); font-size: 12px; text-transform: uppercase; margin-bottom: 4px;">🧠 深度概念与心智模型</h4>
          <p>${stepData.concept}</p>
        </div>
        <div style="margin-bottom: 12px;">
          <h4 style="color: var(--accent-emerald); font-size: 12px; text-transform: uppercase; margin-bottom: 4px;">📊 观测与硬件计数器</h4>
          <pre style="white-space: pre-line; background: var(--bg-tertiary); padding: 8px 10px; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 11px;">${stepData.hwMetrics}</pre>
        </div>
        <div>
          <h4 style="color: var(--accent-amber); font-size: 12px; text-transform: uppercase; margin-bottom: 4px;">🚀 调优实战落地原则</h4>
          <p style="color: #f8fafc;">${stepData.tuningTakeaway}</p>
        </div>
      `;

      modal.style.display = 'flex';
    });
  });
}
