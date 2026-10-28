import pandas as pd

# 1. Hardware Constants: NVIDIA Tesla T4 (Turing sm_75)
T4_USABLE_VRAM_GIB = 15.00        # 15,360 MiB usable after ECC
T4_BW_GBPS = 320.0                # 320 GB/s GDDR6 bandwidth
T4_FP16_TFLOPS = 65.0             # 65 TFLOPS FP16 Tensor Core

# 2. Model Constants: Qwen/Qwen2.5-3B-Instruct
PARAMS_B = 3.09
FP16_WEIGHT_GB = PARAMS_B * 2     # 6.18 GB (~5.76 GiB)
AWQ_INT4_WEIGHT_GB = PARAMS_B * 0.55  # ~1.70 GB (~1.58 GiB) with 4-bit AWQ + scales

ridge_point = (T4_FP16_TFLOPS * 1e3) / T4_BW_GBPS
fp16_max_tps = T4_BW_GBPS / FP16_WEIGHT_GB
awq_max_tps = T4_BW_GBPS / AWQ_INT4_WEIGHT_GB

print("=== SPRINT 1: TESLA T4 (sm_75) ROOFLINE FOR QWEN2.5-3B-INSTRUCT ===")
print(f"Usable VRAM:                  {T4_USABLE_VRAM_GIB:.2f} GiB")
print(f"FP16 Ridge Point:             {ridge_point:.1f} FLOPs/Byte")
print(f"Max Decode Speed (B=1, FP16): {fp16_max_tps:.1f} tokens/sec (6.18 GB weights)")
print(f"Max Decode Speed (B=1, AWQ4): {awq_max_tps:.1f} tokens/sec (1.70 GB weights)\n")

# 3. KV Cache Math: 2 (K+V) * 36 layers * 2 KV_heads * 128 head_dim * 2 bytes (FP16)
KV_BYTES_PER_TOK = 2 * 36 * 2 * 128 * 2  # 36,864 bytes (36 KB/token)

rows = []
for seq_len in [4096, 16384, 32768]:
    for batch in [1, 8, 16]:
        kv_gib = (KV_BYTES_PER_TOK * seq_len * batch) / (1024**3)
        fp16_total_gib = (FP16_WEIGHT_GB / 1.07374) + 1.20 + kv_gib
        awq_total_gib = (AWQ_INT4_WEIGHT_GB / 1.07374) + 1.20 + kv_gib
        rows.append({
            "Seq_Len": seq_len,
            "Batch": batch,
            "KV_Cache_GiB": round(kv_gib, 2),
            "FP16_Total_GiB": round(fp16_total_gib, 2),
            "FP16_Fits_15GiB?": "YES" if fp16_total_gib <= T4_USABLE_VRAM_GIB else "OOM CRASH",
            "AWQ4_Total_GiB": round(awq_total_gib, 2),
            "AWQ4_Fits_15GiB?": "YES" if awq_total_gib <= T4_USABLE_VRAM_GIB else "OOM CRASH",
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
