"""
SCS* Pruner: Zero-Label Collision Detection & Pruning for Retrieval-Augmented Generation.
Paper: Shadowed in Vector Space: Zero-Label Detection and Resolution of Stale Version Collisions in RAG
"""

import re
import difflib
from typing import List, Dict, Any, Optional

class SCSPruner:
    """
    Drop-in document pruner that detects near-duplicate version collisions
    and filters out stale historical documents before generation.
    """
    def __init__(
        self,
        tau_collision: float = 0.55,
        entity_threshold: float = 0.50,
        jaccard_weight: float = 0.50,
        token_weight: float = 0.50
    ):
        self.tau_collision = tau_collision
        self.entity_threshold = entity_threshold
        self.jaccard_weight = jaccard_weight
        self.token_weight = token_weight

    def _extract_tokens(self, text: str) -> set:
        words = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', text.lower())
        return set(words)

    def compute_collision_score(self, doc_a: str, doc_b: str) -> float:
        """
        Computes zero-leakage token-level collision score between two documents.
        """
        set_a = self._extract_tokens(doc_a)
        set_b = self._extract_tokens(doc_b)
        if not set_a or not set_b:
            return 0.0

        jaccard = len(set_a & set_b) / max(len(set_a | set_b), 1)
        seq_ratio = difflib.SequenceMatcher(None, doc_a.lower(), doc_b.lower()).quick_ratio()
        
        return self.jaccard_weight * jaccard + self.token_weight * seq_ratio

    def prune(self, docs: List[Dict[str, Any]], text_key: str = "text", timestamp_key: str = "timestamp") -> List[Dict[str, Any]]:
        """
        Prunes collided stale documents, keeping the latest / fresh version.
        Each doc should have a text field and optionally timestamp or version metadata.
        """
        if not docs:
            return []

        # Sort docs if timestamp exists (descending: newest first)
        has_time = any(timestamp_key in d for d in docs)
        if has_time:
            sorted_docs = sorted(docs, key=lambda x: x.get(timestamp_key, 0), reverse=True)
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
                    # Near identical structure / entity overlap, but older -> discard
                    is_stale_collision = True
                    break

            if not is_stale_collision:
                retained.append(cand)

        return retained


# LangChain Compressor Adapter
try:
    from langchain_core.documents import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor
    
    class LangChainSCSCompressor(BaseDocumentCompressor):
        tau_collision: float = 0.55
        
        def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
            pruner = SCSPruner(tau_collision=self.tau_collision)
            doc_dicts = [{"doc": d, "text": d.page_content, "timestamp": d.metadata.get("timestamp", 0)} for d in documents]
            pruned_dicts = pruner.prune(doc_dicts, text_key="text", timestamp_key="timestamp")
            return [d["doc"] for d in pruned_dicts]
except ImportError:
    pass

# LlamaIndex Postprocessor Adapter
try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle

    class LlamaIndexSCSPostprocessor(BaseNodePostprocessor):
        tau_collision: float = 0.55

        def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
            pruner = SCSPruner(tau_collision=self.tau_collision)
            node_dicts = [{"node": n, "text": n.node.get_content(), "timestamp": n.node.metadata.get("timestamp", 0)} for n in nodes]
            pruned = pruner.prune(node_dicts, text_key="text", timestamp_key="timestamp")
            return [p["node"] for p in pruned]
except ImportError:
    pass
