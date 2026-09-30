// ==============================================================================
// App Logic & Interactive Handlers (Zero-dependency Vanilla ES6)
// ==============================================================================

import { PIPELINE_NODES, DIAGNOSTIC_CASES } from './data.js';

let currentNodeId = 'latency';

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  renderTopologyNodes();
  selectNode(currentNodeId);
  initKVCacheCalculator();
  initToolchainGenerator();
  renderDiagnosticCases();
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
