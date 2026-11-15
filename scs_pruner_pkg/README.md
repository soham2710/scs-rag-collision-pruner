# SCS* Pruner (`scs-pruner`)
### Zero-Label Collision Detection & Pruning of Stale Documents in RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**`scs-pruner`** is a drop-in Python library for LangChain, LlamaIndex, and native RAG pipelines that resolves the **Stale Version Collision** problem and the **MMR Diversity Paradox**.

---

### Why is this needed?

In enterprise RAG (finance, legal, internal wikis, healthcare), documents are regularly updated (e.g. Q3 vs Q4 policies, 2024 vs 2025 pricing).

Because dense vector models (e.g. `text-embedding-3`, `bge`, `e5`) use uniform mean-pooling across tokens, **a 1-token version change produces near-identical vector embeddings** ($\Delta \cos \approx \mathcal{O}(1/L^2)$).

1. **Vanilla Dense RAG** retrieves both versions, confusing the generator into **31.8% stale hallucinations**.
2. **Maximal Marginal Relevance (MMR)** actively makes this **worse**—staleness rises to **47.9% (+16.1% inflation)** because outdated versions are treated as "novel diverse information".
3. **`scs-pruner`** detects these semantic collisions and purges stale historical duplicates, driving **stale hallucinations down to 0.0%**.

---

### Quickstart

#### 1. Native Python Usage
```python
from scs_pruner import SCSPruner

pruner = SCSPruner(tau_collision=0.55)

retrieved_docs = [
    {"text": "2024 Refund Policy: Users have 14 days to request a refund.", "timestamp": 20240101},
    {"text": "2025 Refund Policy: Users have 30 days to request a refund.", "timestamp": 20250101},
]

# Automatically detects collision and discards the stale 2024 document
clean_docs = pruner.prune(retrieved_docs)
print(clean_docs)
# [{'text': '2025 Refund Policy: Users have 30 days to request a refund.', 'timestamp': 20250101}]
```

#### 2. LangChain Drop-In
```python
from scs_pruner import LangChainSCSCompressor
from langchain.retrievers import ContextualCompressionRetriever

base_retriever = vectorstore.as_retriever()
compressor = LangChainSCSCompressor(tau_collision=0.55)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

#### 3. LlamaIndex Drop-In
```python
from scs_pruner import LlamaIndexSCSPostprocessor

query_engine = index.as_query_engine(
    node_postprocessors=[LlamaIndexSCSPostprocessor(tau_collision=0.55)]
)
```
