import json

with open("/mnt/ai_vault/projects/scs-multi-gpu-research/artifacts/ten_fold_empirical_results.json") as f:
    run1 = json.load(f)

with open("/mnt/ai_vault/projects/scs-multi-gpu-research/artifacts/stress_test_multiseed_results.json") as f:
    run2 = json.load(f)

print("=== RUN 1: 10-FOLD EMPIRICAL BENCHMARK (Seeds 101-110, 1920 queries, 7680 pairs) ===")
print("Vanilla Dense RAG:")
print(f"  Fresh Acc: {run1['vanilla_dense_rag']['fresh_acc_mean']:.2f}% +- {run1['vanilla_dense_rag']['fresh_acc_std']:.2f}%")
print(f"  Stale Rate: {run1['vanilla_dense_rag']['stale_rate_mean']:.2f}% +- {run1['vanilla_dense_rag']['stale_rate_std']:.2f}%")

print("MMR Diversity RAG:")
print(f"  Fresh Acc: {run1['mmr_diversity_rag']['fresh_acc_mean']:.2f}% +- {run1['mmr_diversity_rag']['fresh_acc_std']:.2f}%")
print(f"  Stale Rate: {run1['mmr_diversity_rag']['stale_rate_mean']:.2f}% +- {run1['mmr_diversity_rag']['stale_rate_std']:.2f}%")

print("Time-Aware HoH RAG:")
print(f"  Fresh Acc: {run1['time_aware_hoh_rag']['fresh_acc_mean']:.2f}% +- {run1['time_aware_hoh_rag']['fresh_acc_std']:.2f}%")
print(f"  Stale Rate: {run1['time_aware_hoh_rag']['stale_rate_mean']:.2f}% +- {run1['time_aware_hoh_rag']['stale_rate_std']:.2f}%")

print("SCS* RAG (Run 1):")
print(f"  Fresh Acc: {run1['scs_v2_rag']['fresh_acc_mean']:.2f}% +- {run1['scs_v2_rag']['fresh_acc_std']:.2f}%")
print(f"  Stale Rate: {run1['scs_v2_rag']['stale_rate_mean']:.2f}% +- {run1['scs_v2_rag']['stale_rate_std']:.2f}%")

print("\n=== RUN 2: PYTORCH NATIVE GPU TENSOR HARNESS (Seeds 201-212, 48 Live Inferences) ===")
print("Vanilla Dense RAG:")
print(f"  Fresh Acc: {run2['vanilla_rag']['fresh_accuracy']:.2f}%")
print(f"  Stale Rate: {run2['vanilla_rag']['stale_hallucination_rate']:.2f}%")

print("MMR Diversity RAG:")
print(f"  Fresh Acc: {run2['mmr_rag']['fresh_accuracy']:.2f}%")
print(f"  Stale Rate: {run2['mmr_rag']['stale_hallucination_rate']:.2f}%")

print("PyTorch SCS* (Run 2):")
print(f"  Fresh Acc: {run2['scs_pytorch_rag']['fresh_accuracy']:.2f}%")
print(f"  Stale Rate: {run2['scs_pytorch_rag']['stale_hallucination_rate']:.2f}%")
print(f"  Collision Prune Success Rate: {run2['scs_pytorch_rag']['collision_prune_success_rate']:.2f}%")
