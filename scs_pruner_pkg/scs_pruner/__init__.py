"""
SCS* Pruner: Zero-Label Collision Detection & Pruning for Retrieval-Augmented Generation.

Paper: Shadowed in Vector Space: Zero-Label Detection and Resolution of Stale Version Collisions in RAG
Zenodo DOI: 10.5281/zenodo.24243871
"""

from .core import (
    SCSPruner,
    PyTorchSCSPruner,
)
from .audit import SCSAudit

try:
    from .core import LangChainSCSCompressor
except (ImportError, AttributeError):
    LangChainSCSCompressor = None

try:
    from .core import LlamaIndexSCSPostprocessor
except (ImportError, AttributeError):
    LlamaIndexSCSPostprocessor = None

__version__ = "1.0.0"
__all__ = [
    "SCSPruner",
    "PyTorchSCSPruner",
    "SCSAudit",
    "LangChainSCSCompressor",
    "LlamaIndexSCSPostprocessor",
    "__version__"
]
