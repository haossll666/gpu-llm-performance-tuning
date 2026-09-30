#!/usr/bin/env python3
import unittest
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from kv_cache_calculator import MODEL_PRESETS, calculate_kv_cache_bytes_per_token
from roofline_analyzer import GPU_SPECS, analyze_roofline, analyze_4xgpu_pcie_moe

class TestAlgorithms(unittest.TestCase):
    def test_deepseek_mla_compression(self):
        """Verify that DeepSeek MLA compresses KV Cache compared to standard GQA"""
        mla_cfg = MODEL_PRESETS["deepseek-v3"]
        mla_bytes = calculate_kv_cache_bytes_per_token(mla_cfg, "fp8")
        
        # 61 layers * 576 elements * 1 byte = 35136 bytes
        expected_bytes = 61 * (512 + 64) * 1
        self.assertEqual(mla_bytes, expected_bytes)

        # Standard 8-head GQA equivalent (61 layers * 2 * 8 * 128 * 1 byte = 124928 bytes)
        gqa_equivalent = 61 * 2 * 8 * 128 * 1
        self.assertGreater(gqa_equivalent / mla_bytes, 3.5)

    def test_roofline_bounds(self):
        """Verify that Decode is memory-bound and Prefill is compute-bound"""
        res = analyze_roofline("rtx4090", precision="fp16", batch_size=8, prompt_len=2048)
        self.assertIn("Memory-Bandwidth Bound", res["decode_bound"])
        self.assertIn("Compute Bound", res["prefill_bound"])
        self.assertGreater(res["ridge_point"], 100.0)

    def test_moe_4xgpu_pcie_comm(self):
        """Verify 4xGPU PCIe MoE communication calculation"""
        moe = analyze_4xgpu_pcie_moe(batch_size=16)
        self.assertGreater(moe["ep_comm_mb_per_step"], 0)
        self.assertGreater(moe["tp_total_mb"], 0)
        # EP communication latency is quantified
        self.assertTrue(0.1 < moe["ep_comm_time_ms"] < 50.0)

if __name__ == '__main__':
    unittest.main()
