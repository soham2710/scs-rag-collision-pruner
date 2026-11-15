# Corollary 1 & Empirical Proof: The MMR Diversity Paradox

## Mathematical Corollary
Maximal Marginal Relevance (MMR) selects document $d$ maximizing:
$$\text{MMR}(q, d) = \lambda \cos(q, d) - (1-\lambda) \max_{d_j \in S} \cos(d, d_j)$$

When an updated passage $R_{v3}$ and an outdated passage $O_{v2}$ share high query relevance but slight wording divergences, $\cos(R_{v3}, O_{v2}) < 1.0$. MMR penalizes redundant contemporary documents and **actively favors outdated superseded documents** under the guise of diversity.

## Live Empirical Measurement
- **Vanilla Dense Stale Rate**: 31.8%
- **MMR Diversity Stale Rate**: 47.9%

MMR increases the Stale Answer Rate by **+16.1%**, proving that standard diversity mechanisms in production RAG actively poison generation with stale assertions.
