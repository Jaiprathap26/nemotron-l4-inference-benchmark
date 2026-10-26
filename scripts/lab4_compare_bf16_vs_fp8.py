import glob, json, os
import pandas as pd
import matplotlib.pyplot as plt

def load_dir(folder, label):
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "conc_*.json"))):
        d = json.load(open(p))
        c = int(os.path.basename(p).split("_")[1].split(".")[0])
        out_tps = d.get("output_throughput", 0.0)
        rows.append({
            "Concurrency": c,
            f"{label}_Out_Tok_s": round(out_tps, 2),
            f"{label}_p99_TTFT_ms": round(d.get("p99_ttft_ms", 0.0), 2),
            f"{label}_p50_ITL_ms": round(d.get("median_itl_ms", 0.0), 2),
            f"{label}_p99_ITL_ms": round(d.get("p99_itl_ms", 0.0), 2),
            f"{label}_Spot_$_1M": round((0.28 / 3600) * (1e6 / max(out_tps, 1e-6)), 4)
        })
    return pd.DataFrame(rows).sort_values("Concurrency")

bf16 = load_dir("results/bf16_sweep", "BF16")
fp8 = load_dir("results/fp8_sweep", "FP8")
m = pd.merge(bf16, fp8, on="Concurrency")

m["Throughput_Gain"] = ((m["FP8_Out_Tok_s"] / m["BF16_Out_Tok_s"] - 1) * 100).round(1).astype(str) + "%"
m["p99_TTFT_Change"] = ((m["FP8_p99_TTFT_ms"] / m["BF16_p99_TTFT_ms"] - 1) * 100).round(1).astype(str) + "%"
m["Cost_Savings"] = ((1 - m["FP8_Spot_$_1M"] / m["BF16_Spot_$_1M"]) * 100).round(1).astype(str) + "%"

print("\n=== MASTER HEAD-TO-HEAD BENCHMARK: BF16 vs. FP8 ON GCP NVIDIA L4 ===")
print(m.to_string(index=False))
m.to_csv("results/master_bf16_vs_fp8_comparison.csv", index=False)

# Plot dual panel: Throughput & p99 TTFT Knee-Point
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(m["Concurrency"], m["BF16_Out_Tok_s"], marker="o", linewidth=2, label="BF16 (54 KV/Mamba blocks)")
ax1.plot(m["Concurrency"], m["FP8_Out_Tok_s"], marker="s", linewidth=2, label="FP8 (178 KV/Mamba blocks)")
ax1.set_title("Sustained Output Throughput (Tokens/sec)")
ax1.set_xlabel("Concurrent Users")
ax1.set_ylabel("Output Tokens / Sec (Higher is Better)")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(m["Concurrency"], m["BF16_p99_TTFT_ms"] / 1000.0, marker="o", linewidth=2, label="BF16 p99 TTFT (s)")
ax2.plot(m["Concurrency"], m["FP8_p99_TTFT_ms"] / 1000.0, marker="s", linewidth=2, label="FP8 p99 TTFT (s)")
ax2.axhline(2.0, color="red", linestyle="--", label="2.0s UX SLA Target")
ax2.set_title("Tail Latency: p99 Time-To-First-Token (Seconds)")
ax2.set_xlabel("Concurrent Users")
ax2.set_ylabel("p99 TTFT in Seconds (Lower is Better)")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle("NVIDIA-Nemotron-Nano-9B-v2 on GCP L4 (24GB): BF16 vs. Native Ada FP8", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/bf16_vs_fp8_knee_point.png", dpi=200)
print("\nSaved plot to plots/bf16_vs_fp8_knee_point.png")
