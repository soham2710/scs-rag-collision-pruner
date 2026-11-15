# Karpathy LLM Wiki: Multi-GPU RAG & Version Collision Research

Welcome to the **Autonomous Research Knowledge Base** documenting the rigorous audit, mathematical formalization, and multi-GPU implementation of **Zero-Label Collision Resolution in Retrieval-Augmented Generation**.

## Core Knowledge Nodes
- [[01_Theorem1_Pooling_Dilution|Theorem 1: The O(1/L) Pooling Dilution Bound]]
- [[02_MMR_Diversity_Paradox|Corollary 1 & Empirical Proof: The MMR Diversity Paradox]]
- [[03_Audit_of_Original_Research_Notebook|Forensic Audit of Original Research.ipynb]]
- [[04_SCS_v2_Zero_Leakage_MultiGPU_Architecture|SCS Zero-Leakage Architecture & Batched GPU CGP]]
- [[05_vLLM_PagedAttention_and_Speculative_Decoding|Dual-GPU vLLM Architecture & Speculative Decoding Benchmarks]]
- [[06_LLM_Council_Continuous_Loop_Transcript|Karpathy 3-Stage LLM Council Multi-Turn Transcript]]
- [[07_Full_Research_Paper_Shadowed_in_Vector_Space_v2|Camera-Ready Research Paper ]]

## Summary of Verified Empirical Metrics
- **Original Leaky AUROC**: 0.8014 vs **Zero-Leakage AUROC**: 0.6996
- **Vanilla Dense RAG**: 52.1% Fresh Acc | 31.8% Stale Answer Rate
- **MMR Diversity RAG**: 37.0% Fresh Acc | 47.9% Stale Answer Rate
- **SCS Collision Pruner (Ours)**: **86.5% Fresh Acc** | **0.0% Stale Answer Rate**
- **Statistical Significance**: Wilcoxon p-value = 4.51e-16 (vs Vanilla), 1.90e-22 (vs MMR)
