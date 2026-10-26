import torch
from torch.profiler import profile, ProfilerActivity
from vllm import LLM, SamplingParams

llm = LLM(
    model="nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8",
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=16,
    gpu_memory_utilization=0.90,
    enforce_eager=True  # Disable CUDA graph wrapper so individual kernel names are visible
)

prompts = [
    "Explain the roofline model and arithmetic intensity for GPU matrix multiplication in detail."
] * 8
params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)

# Warmup pass
llm.generate(prompts[:2], params)
torch.cuda.synchronize()

# Profiled pass
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    llm.generate(prompts, params)
    torch.cuda.synchronize()

table_str = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18)
print("\n=== LAB 5: TOP 18 CUDA KERNELS BY GPU EXECUTION TIME (FP8 HYBRID MAMBA-TRANSFORMER) ===")
print(table_str)

with open("results/lab5_cuda_kernel_profile.txt", "w") as f:
    f.write(table_str + "\n")
