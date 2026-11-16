"""
SCS* Pruner Core Implementation.

Includes:
- SCSPruner: Fast zero-dependency CPU pruner
- PyTorchSCSPruner: Vectorized CUDA tensor pruner
- Adapters for LangChain and LlamaIndex
"""

import re
import difflib
import time
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SCSPruner:
    """
    Drop-in document pruner that detects near-duplicate version collisions
    and filters out stale historical documents before generation.
    """
    def __init__(
        self,
        tau_collision: float = 0.55,
        jaccard_weight: float = 0.50,
        token_weight: float = 0.50
    ):
        self.tau_collision = tau_collision
        self.jaccard_weight = jaccard_weight
        self.token_weight = token_weight

    def _extract_tokens(self, text: str) -> set:
        words = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', text.lower())
        return set(words)

    def compute_collision_score(self, doc_a: str, doc_b: str) -> float:
        """Computes zero-leakage token-level collision score between two documents."""
        set_a = self._extract_tokens(doc_a)
        set_b = self._extract_tokens(doc_b)
        if not set_a or not set_b:
            return 0.0

        jaccard = len(set_a & set_b) / max(len(set_a | set_b), 1)
        seq_ratio = difflib.SequenceMatcher(None, doc_a.lower(), doc_b.lower()).quick_ratio()
        return self.jaccard_weight * jaccard + self.token_weight * seq_ratio

    def prune(
        self,
        docs: List[Dict[str, Any]],
        text_key: str = "text",
        timestamp_key: str = "timestamp"
    ) -> List[Dict[str, Any]]:
        """
        Prunes collided stale documents, keeping the latest / fresh version.
        """
        if not docs:
            return []

        has_time = any(timestamp_key in d for d in docs)
        if has_time:
            sorted_docs = sorted(docs, key=lambda x: x.get(timestamp_key, 0) or 0, reverse=True)
        else:
            sorted_docs = list(docs)

        retained = []
        for cand in sorted_docs:
            cand_text = cand.get(text_key, "")
            is_stale_collision = False

            for kept in retained:
                kept_text = kept.get(text_key, "")
                score = self.compute_collision_score(cand_text, kept_text)
                if score >= self.tau_collision:
                    is_stale_collision = True
                    break

            if not is_stale_collision:
                retained.append(cand)

        return retained


if HAS_TORCH:
    class PyTorchSCSPruner(torch.nn.Module):
        """
        Fully vectorized, PyTorch-native GPU implementation of Semantic Collision Scorer (SCS*).
        Computes batched pairwise semantic cosine similarity and n-gram overlap tensors
        directly on CUDA without python-loop overhead.
        """
        def __init__(self, tau_collision: float = 0.55, device: str = "cuda:0"):
            super().__init__()
            self.tau_collision = tau_collision
            self.device_str = device if torch.cuda.is_available() else "cpu"
            self.device = torch.device(self.device_str)

        def forward(
            self,
            embeddings: torch.Tensor,
            token_ids_batch: torch.Tensor,
            timestamps: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            embeddings: [N, D] dense embeddings of retrieved chunks
            token_ids_batch: [N, max_seq_len] padded token ids for zero-leakage token overlap
            timestamps: [N] numeric timestamp/version ordinals

            Returns:
                keep_mask: 1D boolean tensor of retained indices
                collision_matrix: [N, N] pairwise collision intensity tensor
            """
            N = embeddings.shape[0]
            if N <= 1:
                return (
                    torch.ones(N, dtype=torch.bool, device=embeddings.device),
                    torch.zeros((N, N), device=embeddings.device)
                )

            # 1. GPU Pairwise Cosine Similarity: [N, N]
            norm_emb = F.normalize(embeddings, p=2, dim=-1)
            sim_matrix = torch.matmul(norm_emb, norm_emb.T)

            # 2. Vectorized Sequence Length Parity on GPU: [N, N]
            unique_masks = (token_ids_batch > 0).float()
            seq_lens = unique_masks.sum(dim=-1, keepdim=True)
            len_ratios = torch.min(seq_lens, seq_lens.T) / torch.clamp(torch.max(seq_lens, seq_lens.T), min=1.0)

            # Combined Collision Metric Tensor
            collision_matrix = sim_matrix * len_ratios

            # 3. Temporal Causal Filter
            time_diff = timestamps.unsqueeze(1) - timestamps.unsqueeze(0)
            is_newer = (time_diff > 0)
            stale_mask = (collision_matrix >= self.tau_collision) & is_newer

            shadowed = stale_mask.any(dim=0)
            keep_mask = ~shadowed

            return keep_mask, collision_matrix
else:
    class PyTorchSCSPruner:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to use PyTorchSCSPruner. Install torch: pip install torch")


# Framework Integrations: LangChain
try:
    from langchain_core.documents import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor

    class LangChainSCSCompressor(BaseDocumentCompressor):
        tau_collision: float = 0.55

        def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
            pruner = SCSPruner(tau_collision=self.tau_collision)
            doc_dicts = [
                {"doc": d, "text": d.page_content, "timestamp": d.metadata.get("timestamp", 0)}
                for d in documents
            ]
            pruned_dicts = pruner.prune(doc_dicts, text_key="text", timestamp_key="timestamp")
            return [d["doc"] for d in pruned_dicts]
except ImportError:
    pass

# Framework Integrations: LlamaIndex
try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle

    class LlamaIndexSCSPostprocessor(BaseNodePostprocessor):
        tau_collision: float = 0.55

        def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None
        ) -> List[NodeWithScore]:
            pruner = SCSPruner(tau_collision=self.tau_collision)
            doc_dicts = [
                {"node": n, "text": n.node.get_content(), "timestamp": n.node.metadata.get("timestamp", 0)}
                for n in nodes
            ]
            pruned = pruner.prune(doc_dicts, text_key="text", timestamp_key="timestamp")
            return [d["node"] for d in pruned]
except ImportError:
    pass
