import pandas as pd

# Your exact GCP NVIDIA L4 hardware constants
L4_USABLE_VRAM_GIB = 23034 / 1024.0  # 22.49 GiB from your nvidia-smi
L4_BW_GBPS = 300.0                   # 300 GB/s HBM bandwidth
L4_BF16_TFLOPS = 121.0               # 121 TFLOPS dense BF16
L4_FP8_TFLOPS = 242.0                # 242 TFLOPS native Ada FP8

ridge_point_bf16 = (L4_BF16_TFLOPS * 1e12) / (L4_BW_GBPS * 1e9)
bf16_weight_gb = 9.0 * 2  # ~18 GB in BF16 (2 bytes/param)
fp8_weight_gb = 9.0 * 1   # ~9 GB in FP8 (1 byte/param)

print("=== 1. SILICON ROOFLINE CONSTANTS (YOUR GCP NVIDIA L4) ===")
print(f"Usable VRAM (after ECC):       {L4_USABLE_VRAM_GIB:.2f} GiB (23,034 MiB)")
print(f"BF16 Ridge Point:              {ridge_point_bf16:.1f} FLOPs/Byte (Compute-bound above this, Memory-bound below)")
print(f"Theoretical Max Decode (B=1):  {L4_BW_GBPS / bf16_weight_gb:.1f} tokens/sec in BF16 vs. {L4_BW_GBPS / fp8_weight_gb:.1f} tokens/sec in FP8\n")

# KV Cache Math: 2 (K+V) * num_attn_layers * 8 KV_groups * 128 head_dim * 2 bytes (BF16)
def kv_bytes_per_token(attn_layers, kv_groups=8, head_dim=128, bpe=2):
    return 2 * attn_layers * kv_groups * head_dim * bpe

nemotron_kv_bpt = kv_bytes_per_token(attn_layers=4)       # 16,384 bytes (16 KB/token)
standard_kv_bpt = kv_bytes_per_token(attn_layers=32)      # 131,072 bytes (128 KB/token)

rows = []
for seq_len in [2048, 8192, 32768, 65536, 131072]:
    for batch in [1, 8, 32]:
        nem_kv_gib = (nemotron_kv_bpt * seq_len * batch) / (1024**3)
        std_kv_gib = (standard_kv_bpt * seq_len * batch) / (1024**3)
        total_nem_vram = (bf16_weight_gb / 1.07374) + nem_kv_gib + 1.5  # weights + KV + activations/Mamba state
        rows.append({
            "Seq_Len": seq_len,
            "Batch": batch,
            "Nemotron_KV_GiB (4 Attn)": round(nem_kv_gib, 3),
            "Standard_KV_GiB (32 Attn)": round(std_kv_gib, 3),
            "Est_Total_VRAM_GiB": round(total_nem_vram, 2),
            "Fits_on_L4_22.49GiB?": "YES" if total_nem_vram <= L4_USABLE_VRAM_GIB else "OOM CRASH"
        })

df = pd.DataFrame(rows)
print("=== 2. HYBRID MAMBA-TRANSFORMER KV CACHE SCALING TABLE ===")
print(df.to_string(index=False))
df.to_csv("results/lab1_kv_cache_comparison.csv", index=False)
