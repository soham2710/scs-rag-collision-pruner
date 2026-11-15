# SCS-v2 Zero-Leakage Multi-GPU Architecture

## Dual-GPU Hardware Topology
- **cuda:0 (NVIDIA RTX 5080 #1)**:
  - BAAI-bge-m3 dense embedder + ColBERT token state generator
  - Target LLM (Qwen2.5-1.5B-Instruct) for generation and verification
- **cuda:1 (NVIDIA RTX 5080 #2)**:
  - BAAI-bge-reranker-v2-m3 sequence classification cross-encoder
  - Draft LLM (Qwen2.5-0.5B-Instruct) for speculative draft generation

## Batched GPU Collision Graph Pruning (CGP)
Rather than sequential Python loops, CGP constructs the $k \times k$ collision matrix directly on GPU tensors and prunes superseded nodes in $< 0.5$ ms per query.
