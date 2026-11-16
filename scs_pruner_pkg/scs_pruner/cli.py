"""
Command-line interface for scs-pruner and scs-audit.
"""

import sys
import json
import argparse
from pathlib import Path
from .core import SCSPruner
from .audit import SCSAudit

def audit_main():
    parser = argparse.ArgumentParser(description="Audit vector index / corpus for stale version collisions.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSON / JSONL corpus file")
    parser.add_argument("--output", "-o", type=str, default="scs_audit_report.html", help="Path to output HTML report")
    parser.add_argument("--tau-collision", "-t", type=float, default=0.55, help="Collision threshold (default: 0.55)")
    parser.add_argument("--time-key", type=str, default="timestamp", help="Metadata field containing version/timestamp")
    parser.add_argument("--text-key", type=str, default="text", help="Metadata field containing text content")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file {in_path} not found.")
        sys.exit(1)

    print(f"[SCS-Audit] Loading corpus from {in_path}...")
    docs = []
    if in_path.suffix == ".jsonl":
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    else:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            docs = data if isinstance(data, list) else data.get("documents", [])

    print(f"[SCS-Audit] Loaded {len(docs)} documents. Running collision scan...")
    auditor = SCSAudit(
        tau_collision=args.tau_collision,
        time_key=args.time_key,
        text_key=args.text_key
    )
    results = auditor.audit_documents(docs)
    
    print("\n" + "=" * 60)
    print("SCS* VECTOR INDEX AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total Documents Scanned : {results['total_documents']:,}")
    print(f"Detected Collisions     : {results['collision_pairs_count']:,} pairs")
    print(f"Shadowed Stale Chunks   : {results['shadowed_stale_count']:,}")
    print(f"Index Collision Rate    : {results['collision_rate_pct']}%")
    print(f"Shadowing Risk Level    : {results['risk_level']}")
    print(f"Audit Latency           : {results['audit_latency_ms']} ms")
    print("=" * 60)

    out_file = auditor.generate_html_report(results, args.output)
    print(f"[SCS-Audit] Interactive executive report saved to: {out_file}\n")

def main():
    parser = argparse.ArgumentParser(description="SCS* Pruner CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Prune command
    p_prune = subparsers.add_parser("prune", help="Prune stale collisions from retrieved passages")
    p_prune.add_argument("--input", "-i", type=str, required=True, help="Input JSON file with candidate passages")
    p_prune.add_argument("--output", "-o", type=str, default="pruned.json", help="Output JSON file")
    p_prune.add_argument("--tau", "-t", type=float, default=0.55, help="Collision threshold")

    # Audit command
    p_audit = subparsers.add_parser("audit", help="Audit corpus for stale collisions")
    p_audit.add_argument("--input", "-i", type=str, required=True, help="Input JSON / JSONL corpus file")
    p_audit.add_argument("--output", "-o", type=str, default="scs_audit_report.html", help="Path to output HTML report")
    p_audit.add_argument("--tau", "-t", type=float, default=0.55, help="Collision threshold")

    args = parser.parse_args()
    if args.command == "audit":
        sys.argv = [sys.argv[0], "--input", args.input, "--output", args.output, "--tau-collision", str(args.tau)]
        audit_main()
    elif args.command == "prune":
        with open(args.input, "r", encoding="utf-8") as f:
            docs = json.load(f)
        pruner = SCSPruner(tau_collision=args.tau)
        pruned = pruner.prune(docs)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(pruned, f, indent=2)
        print(f"Pruned {len(docs)} documents down to {len(pruned)} fresh documents. Saved to {args.output}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
