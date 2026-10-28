from scripts.prefix_cache_and_chunked_prefill_sim import analyze_prefix_cache, simulate_chunked_prefill_itl

def test_nemotron_1312_block_alignment_vs_16_block():
    std = analyze_prefix_cache(4096, 104, block_size=16)
    nem_unaligned = analyze_prefix_cache(4096, 104, block_size=1312)
    nem_aligned = analyze_prefix_cache(3936, 104, block_size=1312)

    assert std["Unaligned_Tail_Tok"] == 0
    assert std["Tokens_Computed_Hit"] == 104
    assert nem_unaligned["Cached_Blocks"] == 3
    assert nem_unaligned["Cached_Tokens"] == 3936
    assert nem_unaligned["Unaligned_Tail_Tok"] == 160
    assert nem_unaligned["Tokens_Computed_Hit"] == 264
    assert nem_aligned["Unaligned_Tail_Tok"] == 0

def test_chunked_prefill_slashes_itl_stall():
    res = simulate_chunked_prefill_itl(16384, chunk_size=512)
    assert res["Chunked_Max_ITL_ms"] < 30.0
    assert res["Unchunked_Max_ITL_Stall_ms"] > 650.0
    assert res["ITL_Jitter_Reduction_%"] > 95.0
