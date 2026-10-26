import time
import json
import requests
import pandas as pd

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
PROMPT = "A 9B parameter LLM in BF16 (18 GB weights) runs on an NVIDIA L4 GPU with 300 GB/s memory bandwidth. Calculate the theoretical maximum tokens/sec at batch size 1 and explain why."

def benchmark_mode(mode_name, sys_prompt, max_tokens=350):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": PROMPT}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    t0 = time.perf_counter()
    first_token_time = None
    token_times = []
    usage = {}

    with requests.post(URL, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:]
            if data_str.strip() == "[DONE]":
                break
            chunk = json.loads(data_str)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices", [])
            if choices and choices[0].get("delta", {}).get("content"):
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)

    t_end = time.perf_counter()
    comp_tokens = usage.get("completion_tokens", len(token_times))
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else 0.0
    decode_Window = (t_end - first_token_time) if first_token_time else (t_end - t0)
    decode_tps = (comp_tokens - 1) / max(decode_Window, 1e-6)
    itl_ms = (decode_Window * 1000) / max(comp_tokens - 1, 1)
    # Effective HBM bandwidth achieved = 16.58 GiB (17.8 GB) * decode_tps
    achieved_bw_gbps = 17.8 * decode_tps

    return {
        "Mode": mode_name,
        "System_Prompt": sys_prompt,
        "Output_Tokens": comp_tokens,
        "TTFT_ms": round(ttft_ms, 2),
        "ITL_ms_per_tok": round(itl_ms, 2),
        "Decode_Tok_s": round(decode_tps, 2),
        "HBM_BW_Used_GB_s": f"{achieved_bw_gbps:.1f} / 300.0 ({achieved_bw_gbps/3.0:.1f}%)",
        "Total_Latency_s": round(t_end - t0, 2)
    }

rows = [
    benchmark_mode("Reasoning OFF", "/no_think", max_tokens=350),
    benchmark_mode("Reasoning ON", "detailed thinking on", max_tokens=350)
]
df = pd.DataFrame(rows)
print("\n=== LAB 2: BF16 ROOFLINE & THINKING BUDGET BENCHMARK ===")
print(df.to_string(index=False))
df.to_csv("results/lab2_thinking_budget.csv", index=False)
