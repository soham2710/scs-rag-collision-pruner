"""
SCS Stale-Version Audit Engine.

Scans any vector index, document collection, or JSONL corpus for semantic collisions,
quantifies temporal shadowing risk, and generates an executive HTML audit report.
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np

class SCSAudit:
    """
    Enterprise Stale-Version Auditor for RAG Vector Databases.
    Identifies document versions that shadow one another in embedding space.
    """
    def __init__(
        self,
        tau_collision: float = 0.55,
        tau_semantic: float = 0.82,
        time_key: str = "timestamp",
        text_key: str = "text",
        id_key: str = "id"
    ):
        self.tau_collision = tau_collision
        self.tau_semantic = tau_semantic
        self.time_key = time_key
        self.text_key = text_key
        self.id_key = id_key

    def _tokenize(self, text: str) -> set:
        import re
        return set(re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower()))

    def audit_documents(
        self,
        docs: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Scans a list of documents and identifies all collision clusters and shadowed stale documents.
        """
        t0 = time.perf_counter()
        N = len(docs)
        if N < 2:
            return {
                "total_documents": N,
                "collision_pairs": 0,
                "shadowed_stale_documents": 0,
                "collision_rate_pct": 0.0,
                "clusters": [],
                "audit_time_ms": 0.0
            }

        # If embeddings provided, compute cosine matrix
        if embeddings is not None:
            norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            norms[norms == 0] = 1e-9
            norm_emb = embeddings / norms
            cos_matrix = np.dot(norm_emb, norm_emb.T)
        else:
            cos_matrix = None

        collision_pairs = []
        shadowed_indices = set()
        token_sets = [self._tokenize(d.get(self.text_key, "")) for d in docs]
        timestamps = [float(d.get(self.time_key, 0) or 0) for d in docs]

        # Scan pairwise
        for i in range(N):
            for j in range(i + 1, N):
                # 1. Cosine Proximity check (if embeddings provided)
                if cos_matrix is not None:
                    sim = float(cos_matrix[i, j])
                    if sim < self.tau_semantic:
                        continue
                else:
                    sim = None

                # 2. Token overlap & Length ratio
                s_i = token_sets[i]
                s_j = token_sets[j]
                if not s_i or not s_j:
                    continue

                jaccard = len(s_i & s_j) / max(1, len(s_i | s_j))
                len_ratio = min(len(s_i), len(s_j)) / max(1, max(len(s_i), len(s_j)))
                
                # Composite collision metric
                col_score = 0.5 * jaccard + 0.5 * len_ratio if sim is None else 0.4 * sim + 0.3 * jaccard + 0.3 * len_ratio

                if col_score >= self.tau_collision:
                    t_i = timestamps[i]
                    t_j = timestamps[j]
                    
                    # Temporal causal resolution
                    if t_i != t_j:
                        fresh_idx = i if t_i > t_j else j
                        stale_idx = j if t_i > t_j else i
                        shadowed_indices.add(stale_idx)
                        collision_pairs.append({
                            "fresh_id": docs[fresh_idx].get(self.id_key, f"doc_{fresh_idx}"),
                            "fresh_time": timestamps[fresh_idx],
                            "fresh_snippet": docs[fresh_idx].get(self.text_key, "")[:120],
                            "stale_id": docs[stale_idx].get(self.id_key, f"doc_{stale_idx}"),
                            "stale_time": timestamps[stale_idx],
                            "stale_snippet": docs[stale_idx].get(self.text_key, "")[:120],
                            "collision_score": round(col_score, 4),
                            "semantic_similarity": round(sim, 4) if sim is not None else None
                        })

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        collision_rate = (len(shadowed_indices) / max(1, N)) * 100.0

        risk_level = "CRITICAL" if collision_rate > 20.0 else ("HIGH" if collision_rate > 10.0 else ("MODERATE" if collision_rate > 3.0 else "LOW"))

        return {
            "total_documents": N,
            "collision_pairs_count": len(collision_pairs),
            "shadowed_stale_count": len(shadowed_indices),
            "collision_rate_pct": round(collision_rate, 2),
            "risk_level": risk_level,
            "audit_latency_ms": round(elapsed_ms, 2),
            "collision_pairs": collision_pairs
        }

    def generate_html_report(self, audit_results: Dict[str, Any], output_path: str = "scs_audit_report.html") -> str:
        """
        Renders an interactive, publication-grade executive audit report in standalone HTML.
        """
        res = audit_results
        risk_color = "#e53e3e" if res["risk_level"] in ("CRITICAL", "HIGH") else ("#d69e2e" if res["risk_level"] == "MODERATE" else "#38a169")

        pairs_html = ""
        for p in res.get("collision_pairs", [])[:50]:
            pairs_html += f"""
            <tr style="border-bottom: 1px solid #edf2f7;">
                <td style="padding: 10px; font-weight: 600; color: #2d3748;">{p['collision_score']}</td>
                <td style="padding: 10px; color: #38a169;"><b>[ID: {p['fresh_id']}]</b> (v.{p['fresh_time']})<br><span style="font-size: 12px; color: #4a5568;">{p['fresh_snippet']}...</span></td>
                <td style="padding: 10px; color: #e53e3e;"><b>[ID: {p['stale_id']}]</b> (v.{p['stale_time']})<br><span style="font-size: 12px; color: #718096; text-decoration: line-through;">{p['stale_snippet']}...</span></td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SCS Stale-Version Vector Index Audit Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f7fafc; color: #2d3748; margin: 0; padding: 24px; }}
        .card {{ background: #fff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); padding: 24px; margin-bottom: 24px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 16px; }}
        .stat-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 16px; text-align: center; }}
        .stat-val {{ font-size: 28px; font-weight: 700; color: #1a202c; }}
        .stat-lbl {{ font-size: 13px; color: #718096; text-transform: uppercase; margin-top: 4px; }}
        .badge {{ display: inline-block; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 13px; color: #fff; background: {risk_color}; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }}
        th {{ background: #edf2f7; text-align: left; padding: 10px; font-weight: 600; color: #4a5568; }}
    </style>
</head>
<body>
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 24px;">SCS* Stale-Version Vector Index Audit</h1>
                <p style="margin: 4px 0 0; color: #718096;">Automated Vector Collision and Temporal Shadowing Risk Assessment</p>
            </div>
            <div>
                <span class="badge">RISK: {res['risk_level']}</span>
            </div>
        </div>
        
        <div class="grid">
            <div class="stat-box">
                <div class="stat-val">{res['total_documents']:,}</div>
                <div class="stat-lbl">Total Documents Scanned</div>
            </div>
            <div class="stat-box">
                <div class="stat-val" style="color: #e53e3e;">{res['collision_rate_pct']}%</div>
                <div class="stat-lbl">Index Collision Rate</div>
            </div>
            <div class="stat-box">
                <div class="stat-val" style="color: #e53e3e;">{res['shadowed_stale_count']}</div>
                <div class="stat-lbl">Shadowed Stale Chunks</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{res['audit_latency_ms']} ms</div>
                <div class="stat-lbl">Audit Execution Latency</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2 style="margin-top: 0; font-size: 18px;">Detected Version Collision Pairs</h2>
        <p style="color: #718096; font-size: 14px;">The following passages share high embedding cosine proximity, causing dense retrieval to return obsolete versions alongside updated ones.</p>
        <table>
            <thead>
                <tr>
                    <th style="width: 120px;">Collision Score</th>
                    <th>Fresh Version (Retained)</th>
                    <th>Shadowed Version (Pruned)</th>
                </tr>
            </thead>
            <tbody>
                {pairs_html}
            </tbody>
        </table>
    </div>
    
    <div style="text-align: center; color: #a0aec0; font-size: 12px; margin-top: 24px;">
        Generated by SCS* Pruner v1.0.0 &bull; <a href="https://doi.org/10.5281/zenodo.24243871" style="color: #4a5568;">Zenodo DOI: 10.5281/zenodo.24243871</a> &bull; MIT License
    </div>
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path
