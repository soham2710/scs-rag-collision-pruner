#!/usr/bin/env python3
"""
Complete Benchmark:
1. Original Method WITH regexes (historical reference)
2. Original Method WITHOUT regexes (ablation showing degradation)
3. Redesigned SCS* (Pure Neural: Query-attention token salience + unpooled MaxSim divergence + neural cross-encoder, ZERO regexes)
"""

import os
import sys
import time
import math
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

PROJECT_ROOT = Path("/mnt/ai_vault/projects/scs-multi-gpu-research")
sys.path.insert(0, str(PROJECT_ROOT))

from src.original_research_raw import (
    ExperimentConfig,
    build_comprehensive_research_corpus,
)

BGE_M3_PATH = "/run/media/soham/D Drive/models/05_Multimodal_Audio_Video_RAG/BAAI-bge-m3"
RERANKER_PATH = "/run/media/soham/D Drive/models/05_Multimodal_Audio_Video_RAG/BAAI-bge-reranker-v2-m3"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

device_embed = "cuda:0" if torch.cuda.is_available() else "cpu"
device_rerank = "cuda:1" if torch.cuda.device_count() > 1 else device_embed

class SCSComprehensiveSuite:
    def __init__(self):
        print(f"[Init] Loading BGE-M3 on {device_embed}...")
        self.tok_embed = AutoTokenizer.from_pretrained(BGE_M3_PATH)
        self.mod_embed = AutoModel.from_pretrained(BGE_M3_PATH).to(device_embed).eval()
        
        print(f"[Init] Loading BGE-Reranker-v2-m3 on {device_rerank}...")
        self.tok_rerank = AutoTokenizer.from_pretrained(RERANKER_PATH)
        self.mod_rerank = AutoModelForSequenceClassification.from_pretrained(RERANKER_PATH).to(device_rerank).eval()
        
        self.crit_pattern = re.compile(
            r"(\d+|\$\d+|january|february|march|april|may|june|july|august|september|october|november|december|"
            r"202\d|prohibited|forbidden|permitted|allowed|authorized|not|never|strictly|dr\.|mr\.|ms\.|%|mg|days|weeks|years)",
            re.IGNORECASE
        )
        
        self.tau_tie = 0.08
        self.gamma_frame = 0.68
        self.beta_sigmoid = 14.0
        self.top_p_tokens = 16

    @torch.no_grad()
    def encode_corpus(self, texts: List[str], batch_size: int = 64):
        all_globals = []
        all_tok_states = []
        all_tok_strs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tok_embed(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device_embed)
            out = self.mod_embed(**enc)
            last_hidden = out.last_hidden_state
            mask = enc.attention_mask
            
            # Mean-pooled normalized global embeddings
            sum_emb = torch.sum(last_hidden * mask.unsqueeze(-1), dim=1)
            sum_msk = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            global_vecs = F.normalize(sum_emb / sum_msk, p=2, dim=-1)
            all_globals.append(global_vecs.cpu())
            
            for b in range(len(batch)):
                v_len = int(mask[b].sum().item())
                s_i, e_i = 1, max(2, v_len - 1)
                t_vecs = F.normalize(last_hidden[b, s_i:e_i, :], p=2, dim=-1)
                t_ids = enc.input_ids[b, s_i:e_i].tolist()
                t_strs = self.tok_embed.convert_ids_to_tokens(t_ids)
                all_tok_states.append(t_vecs.cpu())
                all_tok_strs.append(t_strs)
                
        return torch.cat(all_globals, dim=0), all_tok_states, all_tok_strs

    def _canonicalize_facts_leaky(self, text: str) -> set:
        clean = re.sub(r"\([^)]*\)", " ", text)
        clean = re.sub(r"\[[^\]]*\]", " ", clean)
        clean = re.sub(r"\b(?:2022|2023|2025|2026\s+policy|2026\s+charter|2026\s+ciso|2026\s+h200|2026\s+it|2026\s+crd-9|2026\s+multi|2026\s+aml|2026\s+kubernetes|2026\s+zero-trust|2026\s+commercial|2026\s+gateway|2026\s+consensus|2026\s+federal|2026\s+orchestration|2026\s+board|2026\s+hepatic|2026)\b", " ", clean, flags=re.IGNORECASE)
        low = clean.lower()
        nums = {n.replace(",", "") for n in re.findall(r"\$?\d+(?:,\d+)?(?:\.\d+)?%?", low)}
        dates = set(re.findall(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,\s*\d{4})?", low))
        polarities = set()
        if re.search(r"\b(prohibited|forbidden|blocked|denied|disallowed|deprecated|terminated)\b", low):
            polarities.add("POLARITY_NEG")
        if re.search(r"\b(permitted|allowed|authorized|granted|active|enabled|current)\b", low):
            polarities.add("POLARITY_POS")
        return nums | dates | polarities

    # 1. Original Method WITH Regexes
    def compute_orig_with_regex(
        self, q_toks, v_i, h_i, toks_i, text_i, score_i, v_j, h_j, toks_j, text_j, score_j, sigma_q=0.1
    ) -> float:
        # Phi_tie
        score_diff = abs(score_i - score_j)
        phi_tie = math.exp(-score_diff / (self.tau_tie * max(sigma_q, 1e-4)))
        
        # Phi_frame
        global_cos = torch.dot(v_i, v_j).item()
        phi_frame = 1.0 / (1.0 + math.exp(-self.beta_sigmoid * (global_cos - self.gamma_frame)))
        
        # Psi_span (with regexes)
        h_i_dev = h_i.to(device_embed)
        h_j_dev = h_j.to(device_embed)
        q_dev = q_toks.to(device_embed)
        
        sim_q_i = torch.matmul(q_dev, h_i_dev.T).max(dim=0).values
        sim_q_j = torch.matmul(q_dev, h_j_dev.T).max(dim=0).values
        
        w_i = torch.tensor([1.75 if self.crit_pattern.search(t.replace("##", "")) else 1.0 for t in toks_i], device=device_embed)
        w_j = torch.tensor([1.75 if self.crit_pattern.search(t.replace("##", "")) else 1.0 for t in toks_j], device=device_embed)
        
        p_i = min(self.top_p_tokens, h_i.size(0))
        p_j = min(self.top_p_tokens, h_j.size(0))
        
        span_i = h_i_dev[torch.topk(sim_q_i * w_i, k=p_i).indices]
        span_j = h_j_dev[torch.topk(sim_q_j * w_j, k=p_j).indices]
        
        cross_sim = torch.matmul(span_i, span_j.T)
        token_span_sim = 0.5 * (cross_sim.max(dim=1).values.mean().item() + cross_sim.max(dim=0).values.mean().item())
        raw_token_div = max(0.0, 1.0 - token_span_sim)
        
        residual_norm = torch.norm(v_j - torch.dot(v_j, v_i) * v_i, p=2).item()
        
        facts_i = self._canonicalize_facts_leaky(text_i)
        facts_j = self._canonicalize_facts_leaky(text_j)
        fact_mismatch = len(facts_i.symmetric_difference(facts_j)) / max(1, len(facts_i.union(facts_j))) if (facts_i and facts_j) else 0.0
        
        psi_span = float(np.clip(
            0.76 * fact_mismatch + 1.40 * raw_token_div * (0.22 + 0.78 * fact_mismatch) + 0.50 * residual_norm * fact_mismatch,
            0.0, 1.0
        ))
        return float(phi_tie * phi_frame * psi_span)

    # 2. Original Method WITHOUT Regexes
    def compute_orig_without_regex(
        self, q_toks, v_i, h_i, toks_i, text_i, score_i, v_j, h_j, toks_j, text_j, score_j, sigma_q=0.1
    ) -> float:
        # Phi_tie
        score_diff = abs(score_i - score_j)
        phi_tie = math.exp(-score_diff / (self.tau_tie * max(sigma_q, 1e-4)))
        
        # Phi_frame
        global_cos = torch.dot(v_i, v_j).item()
        phi_frame = 1.0 / (1.0 + math.exp(-self.beta_sigmoid * (global_cos - self.gamma_frame)))
        
        # Psi_span (NO REGEX: uniform token weights, fact_mismatch = 0)
        h_i_dev = h_i.to(device_embed)
        h_j_dev = h_j.to(device_embed)
        q_dev = q_toks.to(device_embed)
        
        sim_q_i = torch.matmul(q_dev, h_i_dev.T).max(dim=0).values
        sim_q_j = torch.matmul(q_dev, h_j_dev.T).max(dim=0).values
        
        # Uniform weighting (no regex pattern)
        w_i = torch.ones(len(toks_i), device=device_embed)
        w_j = torch.ones(len(toks_j), device=device_embed)
        
        p_i = min(self.top_p_tokens, h_i.size(0))
        p_j = min(self.top_p_tokens, h_j.size(0))
        
        span_i = h_i_dev[torch.topk(sim_q_i * w_i, k=p_i).indices]
        span_j = h_j_dev[torch.topk(sim_q_j * w_j, k=p_j).indices]
        
        cross_sim = torch.matmul(span_i, span_j.T)
        token_span_sim = 0.5 * (cross_sim.max(dim=1).values.mean().item() + cross_sim.max(dim=0).values.mean().item())
        raw_token_div = max(0.0, 1.0 - token_span_sim)
        
        residual_norm = torch.norm(v_j - torch.dot(v_j, v_i) * v_i, p=2).item()
        
        # fact_mismatch is 0 because regex canonicalization is removed
        fact_mismatch = 0.0
        
        psi_span = float(np.clip(
            0.76 * fact_mismatch + 1.40 * raw_token_div * (0.22 + 0.78 * fact_mismatch) + 0.50 * residual_norm * fact_mismatch,
            0.0, 1.0
        ))
        return float(phi_tie * phi_frame * psi_span)

    # 3. Redesigned SCS* (Pure Neural: Query-salience attention + Unpooled MaxSim Divergence + Cross-Encoder Attention)
    def compute_redesigned_scs_star(
        self, q_toks, v_i, h_i, text_i, score_i, v_j, h_j, text_j, score_j, sigma_q=0.1
    ) -> float:
        # Scale-Invariant Rank Tie Density Phi_tie
        score_diff = abs(score_i - score_j)
        phi_tie = math.exp(-score_diff / (self.tau_tie * max(sigma_q, 1e-4)))
        
        # Soft Sigmoidal Frame Equivalence Phi_frame
        global_cos = torch.dot(v_i, v_j).item()
        phi_frame = 1.0 / (1.0 + math.exp(-self.beta_sigmoid * (global_cos - self.gamma_frame)))
        
        if phi_frame < 0.05:
            # Low topical frame equivalence: strictly cannot be a collision
            return 0.0
            
        h_i_dev = h_i.to(device_embed)
        h_j_dev = h_j.to(device_embed)
        q_dev = q_toks.to(device_embed)
        
        # 1. Neural Query Attention Token Salience (Zero Regex!)
        # Tokens that are semantically aligned to the user question are prioritized
        salience_i = torch.matmul(q_dev, h_i_dev.T).max(dim=0).values # [L_i]
        salience_j = torch.matmul(q_dev, h_j_dev.T).max(dim=0).values # [L_j]
        
        # 2. Asymmetric MaxSim Divergence weighted by Query Salience
        # sim_cross[k, l] = cos(h_k^{(i)}, h_l^{(j)})
        sim_cross = torch.matmul(h_i_dev, h_j_dev.T)
        
        # Max semantic support for each token in i when looked up in j
        max_supp_i = sim_cross.max(dim=1).values # [L_i]
        # Unmatched query-salient tokens represent factual substitution
        div_tokens_i = torch.clamp(1.0 - max_supp_i, min=0.0)
        # Weighted divergence
        psi_tok_i = (div_tokens_i * F.softmax(salience_i * 3.0, dim=0)).sum().item()
        
        max_supp_j = sim_cross.max(dim=0).values # [L_j]
        div_tokens_j = torch.clamp(1.0 - max_supp_j, min=0.0)
        psi_tok_j = (div_tokens_j * F.softmax(salience_j * 3.0, dim=0)).sum().item()
        
        psi_neural_diverge = 0.5 * (psi_tok_i + psi_tok_j)
        
        # 3. Neural Cross-Encoder Calibration
        with torch.no_grad():
            rr_inputs = self.tok_rerank(
                [[text_i, text_j]], padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(device_rerank)
            rr_logit = self.mod_rerank(**rr_inputs).logits.item()
            ce_prob = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, rr_logit))))
            
        # Unified Neural Span Factor (Pure Neural, Zero Regex)
        # Multiplies high token divergence by cross-encoder consistency
        psi_neural = float(np.clip(
            2.5 * psi_neural_diverge * (0.35 + 0.65 * ce_prob),
            0.0, 1.0
        ))
        
        # Multiplicative Gating
        scs_star = phi_tie * phi_frame * psi_neural
        return float(scs_star)


def run_rigorous_evaluation():
    print("=" * 84)
    print("RUNNING RIGOROUS RE-RUN & REDESIGN EVALUATION")
    print("=" * 84)
    
    suite = SCSComprehensiveSuite()
    scenarios, chunks = build_comprehensive_research_corpus(num_scenarios=480)
    chunk_texts = [c.text for c in chunks]
    
    print("\n[Encoding] Processing all passages with BGE-M3...")
    dense_corpus, tok_states, tok_strs = suite.encode_corpus(chunk_texts, batch_size=64)
    
    test_scenarios = [s for s in scenarios if s.split == "test"]
    print(f"\nTotal test scenarios: {len(test_scenarios)}")
    
    # Encode test queries
    test_queries = [s.query for s in test_scenarios]
    q_dense, q_toks, _ = suite.encode_corpus(test_queries, batch_size=32)
    
    pairs_labels = []
    scores_orig_reg = []
    scores_orig_noreg = []
    scores_redesigned = []
    
    t0 = time.perf_counter()
    eval_count = 0
    for s_idx, sc in enumerate(test_scenarios):
        # Scenario chunks
        base = s_idx * 5
        q_d = q_dense[s_idx]
        q_t = q_toks[s_idx]
        
        c_indices = [base + k for k in range(5)]
        raw_scores = [torch.dot(q_d, dense_corpus[ci]).item() for ci in c_indices]
        sigma_q = float(np.std(raw_scores)) + 1e-4
        
        # Test candidate pairs for this scenario:
        #   (0, 1) -> Real Version Update Collision (Label 1)
        #   (0, 2) -> Real Amendment Collision (Label 1)
        #   (0, 3) -> Benign Equivalent Duplicate / Non-Collision (Label 0)
        #   (0, 4) -> Hard Negative Topical Distractor (Label 0)
        pairs = [
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 0),
            (0, 4, 0)
        ]
        
        for i_loc, j_loc, label in pairs:
            idx_i = c_indices[i_loc]
            idx_j = c_indices[j_loc]
            v_i = dense_corpus[idx_i]
            v_j = dense_corpus[idx_j]
            h_i = tok_states[idx_i]
            h_j = tok_states[idx_j]
            t_i = chunk_texts[idx_i]
            t_j = chunk_texts[idx_j]
            ts_i = tok_strs[idx_i]
            ts_j = tok_strs[idx_j]
            score_i = raw_scores[i_loc]
            score_j = raw_scores[j_loc]
            
            s1 = suite.compute_orig_with_regex(q_t, v_i, h_i, ts_i, t_i, score_i, v_j, h_j, ts_j, t_j, score_j, sigma_q=sigma_q)
            s2 = suite.compute_orig_without_regex(q_t, v_i, h_i, ts_i, t_i, score_i, v_j, h_j, ts_j, t_j, score_j, sigma_q=sigma_q)
            s3 = suite.compute_redesigned_scs_star(q_t, v_i, h_i, t_i, score_i, v_j, h_j, t_j, score_j, sigma_q=sigma_q)
            
            pairs_labels.append(label)
            scores_orig_reg.append(s1)
            scores_orig_noreg.append(s2)
            scores_redesigned.append(s3)
            eval_count += 1
            
        if (s_idx + 1) % 40 == 0:
            print(f"Evaluated {s_idx + 1}/{len(test_scenarios)} scenarios ({eval_count} pairs)...")
            
    total_time = time.perf_counter() - t0
    print(f"\nEvaluated {eval_count} test pairs in {total_time:.2f}s ({total_time/eval_count*1000:.2f} ms/pair)")
    
    def calc_metrics(y_true, y_scores):
        auc = roc_auc_score(y_true, y_scores)
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        f1_arr = 2 * (prec * rec) / np.maximum(1e-9, prec + rec)
        best_f1 = np.max(f1_arr)
        best_idx = np.argmax(f1_arr)
        return float(auc), float(best_f1), float(prec[best_idx]), float(rec[best_idx])
        
    auc1, f1_1, p1, r1 = calc_metrics(pairs_labels, scores_orig_reg)
    auc2, f1_2, p2, r2 = calc_metrics(pairs_labels, scores_orig_noreg)
    auc3, f1_3, p3, r3 = calc_metrics(pairs_labels, scores_redesigned)
    
    results = {
        "N_test_pairs": eval_count,
        "original_with_regex": {
            "description": "Original paper method with hardcoded regex patterns (leaky baseline)",
            "auroc": auc1,
            "f1": f1_1,
            "precision": p1,
            "recall": r1
        },
        "original_without_regex": {
            "description": "Original method with regexes stripped (ablated baseline)",
            "auroc": auc2,
            "f1": f1_2,
            "precision": p2,
            "recall": r2,
            "degradation_auroc": auc2 - auc1,
            "degradation_f1": f1_2 - f1_1
        },
        "redesigned_scs_star": {
            "description": "Redesigned SCS* (Pure Neural: Query-Salience Token Divergence + Cross-Encoder Calibration, 100% Regex-Free)",
            "auroc": auc3,
            "f1": f1_3,
            "precision": p3,
            "recall": r3,
            "recovery_vs_no_regex_auroc": auc3 - auc2,
            "recovery_vs_no_regex_f1": f1_3 - f1_2
        }
    }
    
    print("\n" + "=" * 84)
    print("FINAL RIGOROUS EMPIRICAL VERIFICATION RESULTS:")
    print("=" * 84)
    print(f"1. Original Baseline (WITH Regexes)   : AUROC = {auc1:.4f} | F1 = {f1_1:.4f}")
    print(f"2. Original Baseline (WITHOUT Regexes): AUROC = {auc2:.4f} | F1 = {f1_2:.4f}  (Degradation: {auc2 - auc1:+.4f} AUROC, {f1_2 - f1_1:+.4f} F1)")
    print(f"3. Redesigned SCS* (Pure Neural)      : AUROC = {auc3:.4f} | F1 = {f1_3:.4f}  (Recovery: {auc3 - auc2:+.4f} AUROC, {f1_3 - f1_2:+.4f} F1)")
    print("=" * 84)
    
    out_path = ARTIFACTS_DIR / "rerun_and_redesign_scs_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved complete results to {out_path}")

if __name__ == "__main__":
    run_rigorous_evaluation()
