---
language:
- en
license: mit
tags:
- rag
- retrieval-augmented-generation
- langchain
- llamaindex
- pytorch
- stale-data
- knowledge-conflict
metrics:
- accuracy
pipeline_tag: text-generation
---

# Shadowed in Vector Space: Zero-Label Detection and Resolution of Stale Version Collisions in Retrieval-Augmented Generation

Official repository for **Semantic Collision Scorer ($\text{SCS}^*$)**, a drop-in zero-latency pruner for stale document collisions in RAG pipelines.

[![GitHub](https://img.shields.io/badge/GitHub-soham2710%2Fscs--rag--collision--pruner-black?logo=github)](https://github.com/soham2710/scs-rag-collision-pruner)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24243871.svg)](https://doi.org/10.5281/zenodo.24243871)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## ⚡ The Problem: Stale Version Collisions & The MMR Diversity Paradox

In production enterprise RAG systems (finance, healthcare, cloud architecture, legal contracts), policies and numbers change constantly.

1. **The Pooling Dilution Bound ($\mathcal{O}(1/L^2)$)**: Dense embeddings pool tokens uniformly. When a policy changes by 1 token, the cosine shift is $\Delta \cos \le \mathcal{O}(1/L^2) < 0.0005$, making vector databases **mathematically blind** to the difference between current and obsolete documents.
2. **The MMR Diversity Paradox**: Standard **Maximal Marginal Relevance (MMR)** actually **worsens staleness by $+16.1\%$** (from $31.8\%$ to $47.9\%$) because historical versions are treated as "valuable novel information".
3. **$\text{SCS}^*$ Resolution**: A zero-label, sub-millisecond ($<0.8\text{ ms}$) geometric sieve running on CUDA tensor cores that drives stale hallucinations down to **$0.00\%$**.

---

## 📊 Empirical Benchmarks Across Dual NVIDIA RTX 5080 GPUs

| Pipeline Strategy | Fresh Answer Acc (%) | Stale Hallucination Rate (%) | Pruning Latency |
| :--- | :---: | :---: | :---: |
| **Vanilla Dense RAG** | $50.00\% - 52.08\%$ | $25.00\% - 31.77\%$ | 110.0 ms |
| **MMR Diversity RAG** | $36.98\% - 50.00\%$ | $25.00\% - 47.92\%$ | 94.4 ms |
| **Time-Aware (HoH)** | $31.77\%$ | $51.56\%$ | 181.2 ms |
| **PyTorch $\text{SCS}^*$ (Ours)** | **$75.00\% - 86.46\%$** | **$0.00\%$** | **$< 0.8\text{ ms}$** |

---

## 📦 Python Installation & Usage

```bash
pip install git+https://github.com/soham2710/scs-rag-collision-pruner.git#subdirectory=scs_pruner_pkg
```

### Native Python
```python
from scs_pruner import SCSPruner

pruner = SCSPruner(tau_collision=0.55)

retrieved_chunks = [
    {"text": "2024 Travel Limit: $50 per day", "timestamp": 20240101},
    {"text": "2025 Travel Limit: $100 per day", "timestamp": 20250101}
]

clean_chunks = pruner.prune(retrieved_chunks)
print(clean_chunks)
# [{'text': '2025 Travel Limit: $100 per day', 'timestamp': 20250101}]
```

### LangChain Drop-In
```python
from scs_pruner import LangChainSCSCompressor
from langchain.retrievers import ContextualCompressionRetriever

retriever = ContextualCompressionRetriever(
    base_compressor=LangChainSCSCompressor(tau_collision=0.55),
    base_retriever=vectorstore.as_retriever()
)
```

---

## 📜 Full Manuscripts & Figures
- [Full PDF Manuscript (.pdf)](https://github.com/soham2710/scs-rag-collision-pruner/blob/main/Shadowed_in_Vector_Space.pdf)
- [Full Word Manuscript (.docx)](https://github.com/soham2710/scs-rag-collision-pruner/blob/main/artifacts/Shadowed_in_Vector_Space.docx)
- [LaTeX Camera-Ready (.tex)](https://github.com/soham2710/scs-rag-collision-pruner/blob/main/artifacts/Shadowed_in_Vector_Space.tex)
- [300 DPI 8-Panel Publication Figure](https://github.com/soham2710/scs-rag-collision-pruner/blob/main/artifacts/scs_multigpu_8panel.png)

```bibtex
@article{sharma2026shadowed,
  title={Shadowed in Vector Space: Zero-Label Detection and Resolution of Stale Version Collisions in Retrieval-Augmented Generation},
  author={Sharma, Soham},
  year={2026}
}
```
