import pandas as pd

def analyze_prefix_cache(sys_prompt_tokens: int, user_query_tokens: int, block_size: int):
    total_input = sys_prompt_tokens + user_query_tokens
    cached_blocks = sys_prompt_tokens // block_size
    cached_tokens = cached_blocks * block_size
    unaligned_tail_tokens = sys_prompt_tokens - cached_tokens
    computed_tokens_on_hit = unaligned_tail_tokens + user_query_tokens
    compute_saved_pct = (cached_tokens / total_input) * 100.0

    return {
        "Block_Size": block_size,
        "Sys_Prompt_Tok": sys_prompt_tokens,
        "User_Query_Tok": user_query_tokens,
        "Cached_Blocks": cached_blocks,
        "Cached_Tokens": cached_tokens,
        "Unaligned_Tail_Tok": unaligned_tail_tokens,
        "Tokens_Computed_Hit": computed_tokens_on_hit,
        "Prefill_FLOPs_Saved_%": round(compute_saved_pct, 2),
    }

def simulate_chunked_prefill_itl(long_prefill_tokens: int, chunk_size: int, prefill_rate_tok_ms: float = 25.0, decode_step_ms: float = 25.0):
    unchunked_itl_stall_ms = (long_prefill_tokens / prefill_rate_tok_ms) + decode_step_ms
    num_chunks = (long_prefill_tokens + chunk_size - 1) // chunk_size
    chunked_itl_ms = (chunk_size / prefill_rate_tok_ms) + (decode_step_ms * 0.25)
    return {
        "Incoming_Prefill_Tok": long_prefill_tokens,
        "Chunk_Size": chunk_size,
        "Num_Chunks": num_chunks,
        "Unchunked_Max_ITL_Stall_ms": round(unchunked_itl_stall_ms, 1),
        "Chunked_Max_ITL_ms": round(chunked_itl_ms, 1),
        "ITL_Jitter_Reduction_%": round((1.0 - chunked_itl_ms / unchunked_itl_stall_ms) * 100.0, 1),
    }

if __name__ == "__main__":
    rows = [
        analyze_prefix_cache(4096, 104, block_size=16),    # Standard Transformer (16-tok block)
        analyze_prefix_cache(4096, 104, block_size=1312),  # Nemotron Hybrid (1312-tok block, unaligned)
        analyze_prefix_cache(3936, 104, block_size=1312),  # Nemotron Hybrid (aligned to 3 * 1312!)
    ]
    print("=== 1. PREFIX CACHING BLOCK ALIGNMENT: STANDARD (16) vs. NEMOTRON (1312) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    chunks = [
        simulate_chunked_prefill_itl(16384, chunk_size=2048),
        simulate_chunked_prefill_itl(16384, chunk_size=512),
    ]
    print("\n=== 2. CHUNKED PREFILL DECODE ITL PROTECTION ===")
    print(pd.DataFrame(chunks).to_string(index=False))