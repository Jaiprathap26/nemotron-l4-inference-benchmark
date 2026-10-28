from scripts.multi_gpu_capacity_planner import kv_kb_per_token, evaluate_deployment

def test_kv_cache_formulas_across_architectures():
    assert kv_kb_per_token("Qwen2.5-3B", kv_bytes=2) == 36.0
    assert kv_kb_per_token("Qwen2.5-7B", kv_bytes=2) == 56.0
    assert kv_kb_per_token("Llama-3.1-8B", kv_bytes=2) == 128.0
    assert kv_kb_per_token("Nemotron-9B-v2", kv_bytes=2) == 16.0
    assert kv_kb_per_token("Llama-3.3-70B", kv_bytes=2) == 320.0
    assert kv_kb_per_token("Llama-3.3-70B", kv_bytes=1) == 160.0

def test_qwen7b_fp16_ooms_on_t4_but_awq4_fits():
    fp16 = evaluate_deployment("Qwen2.5-7B", "1x_T4_16GB", weight_bytes=2.0)
    awq4 = evaluate_deployment("Qwen2.5-7B", "1x_T4_16GB", weight_bytes=0.55)
    assert fp16["Max_Tokens_Pool"] == "OOM CRASH"
    assert isinstance(awq4["Max_Tokens_Pool"], int) and awq4["Max_Tokens_Pool"] > 150000

def test_llama70b_fp8_fits_on_4x_l4_and_h100():
    bf16_l4 = evaluate_deployment("Llama-3.3-70B", "4x_L4_TP4", weight_bytes=2.0)
    fp8_l4 = evaluate_deployment("Llama-3.3-70B", "4x_L4_TP4", weight_bytes=1.0, kv_bytes=1)
    fp8_h100 = evaluate_deployment("Llama-3.3-70B", "1x_H100_80GB", weight_bytes=1.0, kv_bytes=1)
    assert bf16_l4["Max_Tokens_Pool"] == "OOM CRASH"
    assert fp8_l4["Max_Tokens_Pool"] > fp8_h100["Max_Tokens_Pool"]
    assert fp8_h100["Net_Eff_B1_Tok_s"] > fp8_l4["Net_Eff_B1_Tok_s"]
