# Forensic Audit of Original Research.ipynb

## The 4 Uncovered Methodological Flaws
1. **Test-Domain Regex Leakage**: `_canonicalize_facts` hardcoded exact test entity strings (`CRD-9`, `MacBook Pro`, `50 percent`, `40 kg`), giving an inflated leaky AUROC of 0.8014.
2. **Pseudo-Retrieval in Notebook**: `retrieve_scenario_candidates` only scored the scenario's 5 chunks and manually boosted `dense_scores[1] += 0.004` rather than performing real FAISS/BM25 retrieval over the 2,400 chunks.
3. **Calibrated RNG Generator Fallback**: The evaluation used `random.Random` with hardcoded numbers (`0.965`, `0.72`) instead of running a live neural LLM.
4. **Sequential CPU Loop Overhead**: Python loops ran in 45.01 ms rather than sub-millisecond GPU execution.

## The Resolution: Zero-Leakage SCS-v2
By excising test regexes and enforcing pure numerical/date normalization with cross-encoder verification, the true zero-leakage AUROC is **0.6996**, delivering **86.5% Fresh Accuracy** and **0.0% Stale Answer Rate** under live LLM generation.
