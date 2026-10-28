import pandas as pd

MODELS = {
    "Qwen2.5-3B":      {"params_b": 3.09,  "attn_layers": 36, "kv_heads": 2, "head_dim": 128},
    "Qwen2.5-7B":      {"params_b": 7.62,  "attn_layers": 28, "kv_heads": 4, "head_dim": 128},
    "Llama-3.1-8B":    {"params_b": 8.03,  "attn_layers": 32, "kv_heads": 8, "head_dim": 128},
    "Nemotron-9B-v2":  {"params_b": 9.00,  "attn_layers": 4,  "kv_heads": 8, "head_dim": 128},
    "Llama-3.3-70B":   {"params_b": 70.60, "attn_layers": 80, "kv_heads": 8, "head_dim": 128},
}

GPUS = {
    "1x_T4_16GB":   {"tp": 1, "vram_gib_per_gpu": 15.00, "bw_gbps_per_gpu": 320.0,  "sm": 75, "pcie_allreduce_us": 0.0},
    "1x_L4_24GB":   {"tp": 1, "vram_gib_per_gpu": 22.49, "bw_gbps_per_gpu": 300.0,  "sm": 89, "pcie_allreduce_us": 0.0},
    "4x_L4_TP4":    {"tp": 4, "vram_gib_per_gpu": 22.49, "bw_gbps_per_gpu": 300.0,  "sm": 89, "pcie_allreduce_us": 15.0},
    "1x_H100_80GB": {"tp": 1, "vram_gib_per_gpu": 79.10, "bw_gbps_per_gpu": 3350.0, "sm": 90, "pcie_allreduce_us": 0.0},
}

def kv_kb_per_token(model_name: str, kv_bytes: int = 2) -> float:
    m = MODELS[model_name]
    total_bytes = 2 * m["attn_layers"] * m["kv_heads"] * m["head_dim"] * kv_bytes
    return total_bytes / 1024.0

def evaluate_deployment(model_name: str, gpu_cluster: str, weight_bytes: float, kv_bytes: int = 2):
    m = MODELS[model_name]
    g = GPUS[gpu_cluster]
    tp = g["tp"]

    if m["kv_heads"] % tp != 0:
        return {"Status": f"INVALID TP: {m['kv_heads']} KV heads not divisible by TP={tp}"}

    total_weight_gb = m["params_b"] * weight_bytes
    weight_gib_per_gpu = (total_weight_gb / 1.07374) / tp
    free_kv_gib_per_gpu = g["vram_gib_per_gpu"] - weight_gib_per_gpu - 1.25
    total_free_kv_gib = free_kv_gib_per_gpu * tp

    kv_kb_tok = kv_kb_per_token(model_name, kv_bytes=kv_bytes)
    max_cached_tokens = int((total_free_kv_gib * 1024 * 1024) // kv_kb_tok) if total_free_kv_gib > 0 else 0

    combined_bw_gbps = g["bw_gbps_per_gpu"] * tp
    ideal_step_ms = (total_weight_gb / combined_bw_gbps) * 1000.0
    allreduce_Syncs = (m["attn_layers"] * 2) if tp > 1 else 0
    comm_overhead_ms = (allreduce_Syncs * g["pcie_allreduce_us"]) / 1000.0
    real_step_ms = ideal_step_ms + comm_overhead_ms

    return {
        "Model": model_name,
        "Cluster": gpu_cluster,
        "W_Bytes": weight_bytes,
        "KV_KB_Tok": round(kv_kb_tok, 1),
        "Weight_GiB_GPU": round(weight_gib_per_gpu, 2),
        "Free_KV_GiB_Total": round(total_free_kv_gib, 2),
        "Max_Tokens_Pool": max_cached_tokens if total_free_kv_gib > 0 else "OOM CRASH",
        "Ideal_B1_Tok_s": round(1000.0 / ideal_step_ms, 1),
        "Net_Eff_B1_Tok_s": round(1000.0 / real_step_ms, 1) if total_free_kv_gib > 0 else 0.0,
    }

if __name__ == "__main__":
    scenarios = [
        ("Qwen2.5-3B",     "1x_T4_16GB",   2.0,  2),
        ("Qwen2.5-3B",     "1x_T4_16GB",   0.55, 2),
        ("Qwen2.5-7B",     "1x_T4_16GB",   2.0,  2),
        ("Qwen2.5-7B",     "1x_T4_16GB",   0.55, 2),
        ("Llama-3.1-8B",   "1x_L4_24GB",   2.0,  2),
        ("Nemotron-9B-v2", "1x_L4_24GB",   2.0,  2),
        ("Nemotron-9B-v2", "1x_L4_24GB",   1.0,  2),
        ("Llama-3.3-70B",  "4x_L4_TP4",    2.0,  2),
        ("Llama-3.3-70B",  "4x_L4_TP4",    1.0,  1),
        ("Llama-3.3-70B",  "1x_H100_80GB", 1.0,  1),
    ]
    df = pd.DataFrame([evaluate_deployment(*s) for s in scenarios])
    print("=== MULTI-MODEL & MULTI-GPU CAPACITY / ROOFLINE MATRIX (3B -> 70B) ===")
    print(df.to_string(index=False))
    df.to_csv("results/multi_gpu_capacity_matrix.csv", index=False)
