# Shadowed in Vector Space

**Zero-Label Detection and Resolution of Stale Version Collisions in Retrieval-Augmented Generation**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24243871.svg)](https://doi.org/10.5281/zenodo.24243871)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/soham2710/scs-rag-collision-pruner)](https://github.com/soham2710/scs-rag-collision-pruner/releases)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-soham27%2Fscs--rag--collision--pruner-yellow)](https://huggingface.co/soham27/scs-rag-collision-pruner)

When a corpus contains several versions of the same document, dense retrievers return both the current and the obsolete version, and the generator may answer from the obsolete one. This repo contains the paper, the `SCS*` pruner (PyTorch), the evaluation harnesses, raw result logs, and a drop-in package.

**Paper (PDF):** [Shadowed_in_Vector_Space.pdf](Shadowed_in_Vector_Space.pdf)

## Contributions
1. **Pooling dilution bound.** For mean pooling, a single-token change moves the unit-normalised embedding by at most `Δ²/(2L²)` in cosine distance (symbolic check: `src/formal_math_verification.py`).
2. **MMR observation.** In our setup, MMR re-ranking *raised* the stale-answer rate (31.8% → 47.9%).
3. **SCS\* + CGP.** A training-free, zero-label detector on unpooled token states (rank-tie density × frame equivalence × query-conditioned span divergence) with Collision Graph Pruning (`src/scs_rigorous_benchmark.py`). `scs_pruner_pkg/` and `src/scs_pytorch_tensor.py` are a lightweight embedding-level approximation for drop-in use.

## Results (independent audited reproduction; the paper gives the original A100 numbers and the regex-leakage audit in Appendix A)

| Method | Run 1 fresh / stale (%) | Run 2 fresh / stale (%) |
|---|---|---|
| Vanilla dense | 52.1 / 31.8 | 50.0 / 25.0 |
| MMR | 37.0 / 47.9 | 50.0 / 25.0 |
| Time-aware | 31.8 / 51.6 | – |
| **SCS\*** | **86.5 / 0.0** | **75.0 / 0.0** |

- Run 1: 10 seeds, 1,920 queries (`artifacts/ten_fold_empirical_results.json`).
- Run 2: 12 seeds, 4 scenarios, 48 inferences (`artifacts/stress_test_multiseed_results.json`).
- Hardware: 2× RTX 5080; generator Qwen2.5-1.5B-Instruct.

## Limitations (please read)
- Run 1 reports std = 0.0 across seeds → folds are **not independent**; no significance test is claimed.
- Run 2 uses 4 hand-written scenarios and **synthetic embeddings** with injected stale/fresh similarity; it tests the pruning mechanism, not end-to-end retrieval.
- Only a 1.5B generator and a synthetic corpus were used; public conflict benchmarks and larger models are future work.
- The sub-millisecond latency refers to the pruning step only.

## Quickstart
```bash
pip install ./scs_pruner_pkg
```
```python
from scs_pruner import SCSPruner
docs = [
  {"text": "2024 Refund Policy: 14 days.", "timestamp": 20240101},
  {"text": "2025 Refund Policy: 30 days.", "timestamp": 20250101},
]
print(SCSPruner(tau_collision=0.55).prune(docs))  # keeps the 2025 version
```
LangChain (`LangChainSCSCompressor`) and LlamaIndex (`LlamaIndexSCSPostprocessor`) adapters are in `scs_pruner_pkg/`.

## Layout
```
artifacts/   paper (.pdf, .docx source), figure, raw JSON results
src/         SCS* module, benchmarks, stress harness, math verification
scs_pruner_pkg/   pip-installable package
hf_spaces_demo/   Gradio demo
notebooks/   marimo notebook
llm_wiki/    research notes
```

## Citation
See [CITATION.cff](CITATION.cff).
```bibtex
@misc{sharma2026shadowed,
  title  = {Shadowed in Vector Space: Zero-Label Detection and Resolution of Stale Version Collisions in Retrieval-Augmented Generation},
  author = {Sharma, Soham},
  year   = {2026},
  doi    = {10.5281/zenodo.24243871},
  note   = {v1.0.0, https://github.com/soham2710/scs-rag-collision-pruner}
}
```

## License
MIT
