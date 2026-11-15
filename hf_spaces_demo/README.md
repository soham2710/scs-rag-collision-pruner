---
title: SCS* RAG Collision Pruner
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.6.0
app_file: app.py
pinned: false
license: mit
short_description: Zero-label pruning of stale document version collisions in RAG.
---

# Shadowed in Vector Space: SCS* Zero-Label Collision Pruner

Interactive demonstrator for **Semantic Collision Scorer ($\text{SCS}^*$)**, solving stale document version collisions in Retrieval-Augmented Generation (RAG).

### Key Takeaway: The MMR Diversity Paradox
Standard **Maximal Marginal Relevance (MMR)** actually **inflates staleness** by $+16.1\%$ because outdated document versions are treated as "diverse information" by vector similarity. $\text{SCS}^*$ detects and purges stale collided chunks, driving stale hallucinations to **$0.0\%$**.
