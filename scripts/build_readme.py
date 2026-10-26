import pandas as pd

kv = pd.read_csv("results/lab1_kv_cache_comparison.csv").to_string(index=False)
th = pd.read_csv("results/lab2_thinking_budget.csv").to_string(index=False)
ms = pd.read_csv("results/master_bf16_vs_fp8_comparison.csv").to_string(index=False)
kp = open("results/lab5_cuda_kernel_profile.txt").read().strip()
pm = open("results/prometheus_metrics_snapshot.txt").read().strip()

out = [
    "# NVIDIA-Nemotron-Nano-9B-v2 Inference Optimization on GCP L4 (24GB)",
    "",
    "End-to-end systems benchmark of `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (27 Mamba-2 + 25 MLP + 4 GQA layers) in **BF16 vs. Native Ada FP8 (`sm_89`)** using **vLLM V1 (`v0.32.0`)** on a single GCP NVIDIA L4 GPU (`23,034 MiB` usable HBM).",
    "",
    "## 1. Headline Engineering Gains (BF16 vs. Native Ada FP8)",
    "- **VRAM Weight Compression:** `16.58 GiB` (BF16) -> `9.67 GiB` (FP8), saving **6.91 GiB (-41.7%)**",
    "- **Aligned Mamba/KV Cache Pool (`1,312` tok/block):** `2.94 GiB` (`54` blocks / `70,836` tok) -> `9.73 GiB` (`178` blocks / `234,194` tok) (**3.31x increase**)",
    "- **Single-Stream Decode (`C=1`):** `14.70 tok/s` (`66.90 ms` ITL) -> `23.30 tok/s` (`42.21 ms` ITL) (**+58.5% speedup**)",
    "- **Breaking the 54-Block Queue Wall (`C=32`):** `p99 TTFT` dropped from `18,767 ms` to `4,218 ms` (**-77.5% latency reduction**)",
    "- **Peak Sustained Throughput (`C=64`):** `168.45 tok/s` -> `312.26 tok/s` (**+85.4% throughput**, **$0.249 / 1M tokens** on L4 Spot)",
    "",
    "[plots/bf16_vs_fp8_knee_point.png]",
    "",
    "## 2. Phase 1: Silicon Roofline & Hybrid KV Cache Scaling (8x Smaller)",
    "```text", kv, "```",
    "",
    "## 3. Phase 2: Single-Stream Roofline (89.2% HBM Bandwidth) & Thinking Toggle",
    "```text", th, "```",
    "",
    "## 4. Phases 3 & 4: Master BF16 vs. FP8 Concurrency Sweep",
    "```text", ms, "```",
    "",
    "## 5. Phase 5: CUDA Kernel Profile (EngineCore) & Prometheus Telemetry",
    "- `_C::cutlass_scaled_mm` (`sm_89` FP8 GEMM): **69.30% CUDA time** (`1.159s`, `3,328` calls)",
    "- `vllm::mamba_mixer2` (`_selective_scan_update_kernel`): **13.73% CUDA time** (`229.66ms`, `864` calls)",
    "- `FlashAttention-2` (4 GQA layers): **< 0.49% CUDA time**",
    "",
    "```text", kp, "```",
    "",
    "### Production Prometheus Metrics (`/metrics`)",
    "```text", pm, "```"
]

open("README.md", "w").write("\n".join(out) + "\n")
print("SUCCESS: README.md generated!")
