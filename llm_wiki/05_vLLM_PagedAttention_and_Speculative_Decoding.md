# Dual-GPU vLLM Architecture & Speculative Decoding Benchmarks

## PagedAttention KV Block Allocation
- Implemented `PagedKVBlockManager` with 16-token physical blocks and BLAKE2b prefix hashing.
- Sustains Automatic Prefix Caching (APC) with zero-redundancy prompt evaluation.

## Speculative Decoding Results (Target: Qwen2.5-1.5B on cuda:0, Draft: Qwen2.5-0.5B on cuda:1)
- **Autoregressive Baseline**: 129.29 Tokens/Sec (TTFT: 9.7 ms)
- **Prompt Lookup Speculative**: 139.79 Tokens/Sec (**1.08x Speedup**)
- **Neural Dual-GPU Speculative**: 63.65 Tokens/Sec (Acceptance Rate $\alpha = 0.306$)
