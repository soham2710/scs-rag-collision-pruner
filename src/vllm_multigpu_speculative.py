"""Multi-GPU vLLM-Architecture PagedAttention, Prefix Caching, and Speculative Decoding Engine.

Designed for dual NVIDIA RTX 5080 (16 GB VRAM each, 32 GB total) + AI Vault models (/mnt/ai_vault/models).
Implements:
1. AI Vault Model Scanner & Dual-GPU VRAM Telemetry
2. vLLM-style BlockAllocator & BLAKE2b Automatic Prefix Caching (APC / Radix)
3. Dual-GPU Target-Draft Speculative Decoding (Target on cuda:0, Draft on cuda:1)
4. Prompt-Lookup / Merkle-Context Speculative Drafting (MCSD) for RAG workloads
5. Native vLLM & llama.cpp AI Vault Model Inventory Integration
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

AI_VAULT_MODELS_DIR = Path("/mnt/ai_vault/models")


def get_gpu_telemetry() -> List[Dict[str, Any]]:
    """Collect real-time CUDA device memory and hardware telemetry across all GPUs."""
    devices: List[Dict[str, Any]] = []
    if not torch.cuda.is_available():
        return devices
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        free_b, total_b = torch.cuda.mem_get_info(idx)
        alloc_b = torch.cuda.memory_allocated(idx)
        devices.append({
            "gpu_id": idx,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
            "total_vram_gb": round(total_b / (1024 ** 3), 2),
            "free_vram_gb": round(free_b / (1024 ** 3), 2),
            "used_vram_gb": round((total_b - free_b) / (1024 ** 3), 2),
            "torch_allocated_gb": round(alloc_b / (1024 ** 3), 3),
        })
    return devices


def scan_ai_vault_models(models_dir: Path = AI_VAULT_MODELS_DIR) -> List[Dict[str, Any]]:
    """Inventory all local model artifacts in /mnt/ai_vault/models with readiness and GPU fit."""
    artifacts: List[Dict[str, Any]] = []
    if not models_dir.exists():
        return artifacts

    seen_real: set[Path] = set()
    for path in sorted(models_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".gguf", ".safetensors", ".bin", ".pth", ".gpart"}:
            continue
        real = path.resolve()
        if real in seen_real:
            continue
        seen_real.add(real)

        size_bytes = path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)
        status = "downloading" if path.suffix == ".gpart" else "ready"

        if "05_Multimodal" in str(path):
            role = "RAG / Multimodal / Cross-Encoder"
        elif path.suffix == ".gguf":
            role = "Local LLM (GGUF / llama.cpp)"
        elif "ComfyUI" in str(path) or path.parent.name in {"clip", "loras", "unet", "vae"}:
            role = "Vision / Image Editing"
        else:
            role = "Model Artifact"

        if size_gb <= 14.5:
            gpu_fit = "Single RTX 5080 (16 GB)"
        elif size_gb <= 29.0:
            gpu_fit = "Dual RTX 5080 Split (32 GB)"
        else:
            gpu_fit = "Dual GPU + 128 GB RAM Offload"

        artifacts.append({
            "name": path.name,
            "family": str(path.relative_to(models_dir)).split("/")[0],
            "relpath": str(path.relative_to(models_dir)),
            "abs_path": str(path),
            "format": path.suffix.lstrip("."),
            "role": role,
            "size_gb": round(size_gb, 3),
            "size_mb": round(size_bytes / (1024 ** 2), 1),
            "status": status,
            "hardware_fit": gpu_fit,
        })
    return artifacts


@dataclass
class PagedKVBlock:
    block_id: int
    prefix_hash: str
    token_ids: Tuple[int, ...]
    ref_count: int = 1


class PagedKVBlockManager:
    """vLLM-style PagedAttention physical KV block table + BLAKE2b Automatic Prefix Caching (APC)."""

    def __init__(self, block_size: int = 16, num_blocks: int = 4096) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks: List[int] = list(range(num_blocks - 1, -1, -1))
        self.prefix_cache: Dict[str, PagedKVBlock] = {}
        self.total_logical_tokens: int = 0
        self.reused_prefix_tokens: int = 0
        self.cache_hits: int = 0
        self.cache_lookups: int = 0

    def allocate_sequence(self, token_ids: List[int]) -> Dict[str, Any]:
        """Map logical prompt tokens to paged physical KV blocks with prefix hash deduplication."""
        n_tokens = len(token_ids)
        self.total_logical_tokens += n_tokens
        n_blocks = math.ceil(max(1, n_tokens) / self.block_size)

        chain_hash = "root"
        block_table: List[int] = []
        hit_blocks = 0
        new_blocks = 0

        for b_idx in range(n_blocks):
            start = b_idx * self.block_size
            chunk = tuple(token_ids[start : start + self.block_size])
            is_full_block = len(chunk) == self.block_size

            payload = f"{chain_hash}:{chunk}".encode("utf-8")
            chain_hash = hashlib.blake2b(payload, digest_size=16).hexdigest()

            if is_full_block:
                self.cache_lookups += 1
                if chain_hash in self.prefix_cache:
                    blk = self.prefix_cache[chain_hash]
                    blk.ref_count += 1
                    block_table.append(blk.block_id)
                    hit_blocks += 1
                    self.cache_hits += 1
                    self.reused_prefix_tokens += self.block_size
                    continue

            if not self.free_blocks:
                raise RuntimeError("PagedKVBlockManager out of physical KV blocks.")
            phys_id = self.free_blocks.pop()
            blk = PagedKVBlock(block_id=phys_id, prefix_hash=chain_hash, token_ids=chunk)
            if is_full_block:
                self.prefix_cache[chain_hash] = blk
            block_table.append(phys_id)
            new_blocks += 1

        return {
            "logical_tokens": n_tokens,
            "total_blocks": len(block_table),
            "apc_hit_blocks": hit_blocks,
            "newly_allocated_blocks": new_blocks,
            "prefix_hit_tokens": hit_blocks * self.block_size,
            "prefix_hit_rate_pct": round((hit_blocks / max(1, len(block_table))) * 100.0, 2),
            "block_table": block_table[:12],
        }

    def stats(self) -> Dict[str, Any]:
        used = self.num_blocks - len(self.free_blocks)
        return {
            "block_size": self.block_size,
            "total_blocks": self.num_blocks,
            "allocated_blocks": used,
            "cached_prefix_blocks": len(self.prefix_cache),
            "total_logical_tokens": self.total_logical_tokens,
            "reused_prefix_tokens": self.reused_prefix_tokens,
            "apc_savings_pct": round(
                (self.reused_prefix_tokens / max(1, self.total_logical_tokens)) * 100.0, 2
            ),
        }


def _clone_or_crop_past_key_values(
    past_key_values: Any,
    keep_len: int,
) -> Any:
    """Truncate HuggingFace DynamicCache / tuple KV-cache to `keep_len` sequence positions."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "layers"):
        # Transformers 5.x DynamicCache stores list of CacheLayer in past_key_values.layers
        cloned = DynamicCache()
        for layer_idx, layer in enumerate(past_key_values.layers):
            if layer.keys is not None and layer.values is not None:
                k = layer.keys[:, :, :keep_len, :].clone()
                v = layer.values[:, :, :keep_len, :].clone()
                cloned.update(k, v, layer_idx)
        return cloned
    elif hasattr(past_key_values, "crop"):
        # Legacy DynamicCache
        cloned = DynamicCache()
        for layer_idx in range(len(past_key_values)):
            k, v = past_key_values[layer_idx]
            cloned.update(k[:, :, :keep_len, :].clone(), v[:, :, :keep_len, :].clone(), layer_idx)
        return cloned
    return tuple(
        (k[:, :, :keep_len, :].clone(), v[:, :, :keep_len, :].clone())
        for k, v in past_key_values
    )


def _crop_cache_inplace(past_key_values: Any, keep_len: int) -> Any:
    """In-place truncate KV-cache to exact accepted sequence length `keep_len`."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        # Transformers 5.x: crop takes negative integer (tokens_to_remove <= 0)
        cur_len = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else None
        if cur_len is not None:
            if cur_len > keep_len:
                tokens_to_remove = -(cur_len - keep_len)
                past_key_values.crop(tokens_to_remove)
            return past_key_values
        else:
            try:
                past_key_values.crop(keep_len)
                return past_key_values
            except Exception:
                pass
    if hasattr(past_key_values, "layers"):
        # DynamicCache fallback
        return past_key_values
    return tuple((k[:, :, :keep_len, :], v[:, :, :keep_len, :]) for k, v, *_ in past_key_values)


@dataclass
class SpeculativeRunMetrics:
    mode: str
    gamma: int
    prompt_tokens: int
    generated_tokens: int
    total_time_ms: float
    ttft_ms: float
    itl_ms: float
    tokens_per_sec: float
    speedup_vs_ar: float
    drafted_tokens: int
    accepted_tokens: int
    acceptance_rate_alpha: float
    mean_accepted_per_step: float
    theoretical_tokens_per_step: float
    verifier_steps: int
    apc_prefix_hit_pct: float
    target_device: str
    draft_device: str
    output_text: str


class MultiGPUvLLMSpeculativeEngine:
    """Dual-GPU vLLM-architecture inference engine with PagedAttention APC + Speculative Decoding.

    Placement across dual RTX 5080 GPUs:
      - Target LLM (`target_model_name`, e.g. Qwen/Qwen2.5-1.5B-Instruct) on `cuda:0`
      - Draft LLM (`draft_model_name`, e.g. Qwen/Qwen2.5-0.5B-Instruct) on `cuda:1`
    Supports both Neural Draft Speculative Decoding and Training-Free Prompt-Lookup / Merkle Drafting.
    """

    def __init__(
        self,
        target_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        draft_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        target_device: Optional[str] = None,
        draft_device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.has_native_vllm = importlib.util.find_spec("vllm") is not None
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.target_device = target_device or ("cuda:0" if n_gpus >= 1 else "cpu")
        self.draft_device = draft_device or ("cuda:1" if n_gpus >= 2 else self.target_device)
        self.dtype = dtype if torch.cuda.is_available() else torch.float32

        self.target_model_name = target_model_name
        self.draft_model_name = draft_model_name
        self.block_manager = PagedKVBlockManager(block_size=16, num_blocks=4096)

        # Prefix KV-state cache keyed by block prefix signature (real tensor APC reuse!)
        self.apc_tensor_cache: Dict[str, Tuple[Tuple[int, ...], Any, torch.Tensor]] = {}

        print(f"[vLLM-Arch] Loading Target LLM '{target_model_name}' -> {self.target_device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name, torch_dtype=self.dtype
        ).to(self.target_device)
        self.target_model.eval()

        print(f"[vLLM-Arch] Loading Draft LLM '{draft_model_name}' -> {self.draft_device} ...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name, torch_dtype=self.dtype
        ).to(self.draft_device)
        self.draft_model.eval()

        self._warmup()

    def _warmup(self) -> None:
        ids = self.tokenizer("Warmup:", return_tensors="pt")["input_ids"]
        with torch.no_grad():
            _ = self.target_model(ids.to(self.target_device), use_cache=True)
            _ = self.draft_model(ids.to(self.draft_device), use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.no_grad()
    def _prefill_with_apc(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        device: str,
        cache_tag: str,
        enable_apc: bool = True,
    ) -> Tuple[torch.Tensor, Any, float, float]:
        """Prefill prompt using Automatic Prefix Caching when full prefix blocks match."""
        t0 = time.perf_counter()
        seq_list = input_ids[0].tolist()
        apc_info = self.block_manager.allocate_sequence(seq_list)

        # Compute exact longest shared full-block prefix key (aligned to 16-token blocks)
        block_size = self.block_manager.block_size
        n_aligned = (len(seq_list) // block_size) * block_size
        best_prefix_len = 0
        best_entry = None
        if enable_apc and n_aligned >= block_size:
            for plen in range(n_aligned, block_size - 1, -block_size):
                sig = f"{cache_tag}:{hashlib.blake2b(bytes(str(seq_list[:plen]), 'utf-8'), digest_size=12).hexdigest()}"
                if sig in self.apc_tensor_cache and plen < len(seq_list):
                    best_prefix_len = plen
                    best_entry = self.apc_tensor_cache[sig]
                    break

        if best_entry is not None:
            _, cached_kv, _ = best_entry
            past_kv = _clone_or_crop_past_key_values(cached_kv, best_prefix_len)
            suffix_ids = input_ids[:, best_prefix_len:].to(device)
            out = model(suffix_ids, past_key_values=past_kv, use_cache=True)
            logits = out.logits[:, -1, :]
            kv = out.past_key_values
        else:
            out = model(input_ids.to(device), use_cache=True)
            logits = out.logits[:, -1, :]
            kv = out.past_key_values
            if enable_apc and n_aligned >= block_size and len(self.apc_tensor_cache) < 32:
                # Cache aligned prefix up to n_aligned for subsequent queries
                sig = f"{cache_tag}:{hashlib.blake2b(bytes(str(seq_list[:n_aligned]), 'utf-8'), digest_size=12).hexdigest()}"
                self.apc_tensor_cache[sig] = (
                    tuple(seq_list[:n_aligned]),
                    _clone_or_crop_past_key_values(kv, n_aligned),
                    logits.detach().clone(),
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        return logits, kv, ttft_ms, apc_info["prefix_hit_rate_pct"]

    @torch.no_grad()
    def generate_autoregressive(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        enable_apc: bool = True,
    ) -> SpeculativeRunMetrics:
        """Standard autoregressive decoding on Target GPU (cuda:0) with KV-cache."""
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = input_ids.shape[1]

        t_start = time.perf_counter()
        next_logits, past_kv, ttft_ms, apc_pct = self._prefill_with_apc(
            self.target_model, input_ids, self.target_device, "target", enable_apc=enable_apc
        )
        generated: List[int] = []

        next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated.append(int(next_tok.item()))

        for _ in range(max_new_tokens - 1):
            if generated[-1] == self.tokenizer.eos_token_id:
                break
            out = self.target_model(next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(int(next_tok.item()))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t_start) * 1000.0
        n_gen = len(generated)
        decode_ms = max(1e-3, total_ms - ttft_ms)
        itl_ms = decode_ms / max(1, n_gen - 1)
        tps = n_gen / (total_ms / 1000.0)

        return SpeculativeRunMetrics(
            mode="autoregressive_target",
            gamma=1,
            prompt_tokens=prompt_len,
            generated_tokens=n_gen,
            total_time_ms=round(total_ms, 2),
            ttft_ms=round(ttft_ms, 2),
            itl_ms=round(itl_ms, 2),
            tokens_per_sec=round(tps, 2),
            speedup_vs_ar=1.0,
            drafted_tokens=n_gen,
            accepted_tokens=n_gen,
            acceptance_rate_alpha=1.0,
            mean_accepted_per_step=1.0,
            theoretical_tokens_per_step=1.0,
            verifier_steps=n_gen,
            apc_prefix_hit_pct=apc_pct,
            target_device=self.target_device,
            draft_device="none",
            output_text=self.tokenizer.decode(generated, skip_special_tokens=True),
        )

    @torch.no_grad()
    def generate_speculative_neural(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        gamma: int = 5,
        enable_apc: bool = True,
        ar_baseline_tps: Optional[float] = None,
    ) -> SpeculativeRunMetrics:
        """Dual-GPU Speculative Decoding: Draft (cuda:1) proposes gamma tokens; Target (cuda:0) verifies in 1 pass."""
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = input_ids.shape[1]

        t_start = time.perf_counter()
        t_logits, t_kv, t_ttft, apc_pct = self._prefill_with_apc(
            self.target_model, input_ids, self.target_device, "target", enable_apc=enable_apc
        )
        d_logits, d_kv, _, _ = self._prefill_with_apc(
            self.draft_model, input_ids, self.draft_device, "draft", enable_apc=enable_apc
        )

        # First token directly from target prefill
        first_tok = int(torch.argmax(t_logits, dim=-1).item())
        generated: List[int] = [first_tok]
        if first_tok == self.tokenizer.eos_token_id:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return SpeculativeRunMetrics(
                mode="dual_gpu_neural_speculative",
                gamma=gamma,
                prompt_tokens=prompt_len,
                generated_tokens=1,
                total_time_ms=round(total_ms, 2),
                ttft_ms=round(t_ttft, 2),
                itl_ms=0.0,
                tokens_per_sec=round(1.0 / (total_ms / 1000.0), 2),
                speedup_vs_ar=1.0,
                drafted_tokens=0,
                accepted_tokens=0,
                acceptance_rate_alpha=1.0,
                mean_accepted_per_step=1.0,
                theoretical_tokens_per_step=1.0,
                verifier_steps=1,
                apc_prefix_hit_pct=apc_pct,
                target_device=self.target_device,
                draft_device=self.draft_device,
                output_text=self.tokenizer.decode(generated, skip_special_tokens=True),
            )

        # Sync draft model with the first accepted token
        d_in = torch.tensor([[first_tok]], device=self.draft_device)
        d_out = self.draft_model(d_in, past_key_values=d_kv, use_cache=True)
        d_kv = d_out.past_key_values
        d_next_tok = torch.argmax(d_out.logits[:, -1, :], dim=-1, keepdim=True)

        # Target KV currently holds `prompt_len` entries; keep pointer to exact verified length
        target_kv_len = prompt_len
        draft_kv_len = prompt_len + 1

        total_drafted = 0
        total_accepted = 0
        verifier_steps = 0
        pending_prefix_for_target = [first_tok]

        while len(generated) < max_new_tokens:
            k_step = min(gamma, max_new_tokens - len(generated))
            if k_step <= 0:
                break

            # 1) Draft model on cuda:1 generates k_step candidate tokens autoregressively
            draft_tokens: List[int] = [int(d_next_tok.item())]
            cur_tok = d_next_tok
            for _ in range(k_step - 1):
                if draft_tokens[-1] == self.tokenizer.eos_token_id:
                    break
                step_out = self.draft_model(cur_tok, past_key_values=d_kv, use_cache=True)
                d_kv = step_out.past_key_values
                cur_tok = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(int(cur_tok.item()))

            actual_k = len(draft_tokens)
            total_drafted += actual_k

            # 2) Target model on cuda:0 verifies [pending_prefix + draft_tokens] in ONE forward pass
            verify_seq = pending_prefix_for_target + draft_tokens
            verify_tensor = torch.tensor([verify_seq], device=self.target_device)
            t_out = self.target_model(verify_tensor, past_key_values=t_kv, use_cache=True)
            t_kv = t_out.past_key_values
            verifier_steps += 1

            # t_out.logits[0, i] predicts the token AFTER verify_seq[i]:
            #   i = 0 .. actual_k-1 -> verifies draft_tokens[0 .. actual_k-1]
            #   i = actual_k        -> bonus token if all actual_k drafted tokens matched!
            n_match = 0
            replacement_tok = None
            for idx in range(actual_k):
                t_pred = int(torch.argmax(t_out.logits[0, idx, :], dim=-1).item())
                if t_pred == draft_tokens[idx]:
                    n_match += 1
                    generated.append(t_pred)
                    if t_pred == self.tokenizer.eos_token_id or len(generated) >= max_new_tokens:
                        break
                else:
                    replacement_tok = t_pred
                    break

            total_accepted += n_match

            if generated[-1] == self.tokenizer.eos_token_id or len(generated) >= max_new_tokens:
                break

            if n_match == actual_k:
                # All drafted tokens accepted! Take target bonus token from last logit
                bonus_tok = int(torch.argmax(t_out.logits[0, actual_k, :], dim=-1).item())
                generated.append(bonus_tok)
                target_kv_len += len(pending_prefix_for_target) + actual_k
                pending_prefix_for_target = [bonus_tok]

                # Advance draft KV across the last drafted token + bonus token
                d_adv = torch.tensor([[draft_tokens[-1], bonus_tok]], device=self.draft_device)
                d_out = self.draft_model(d_adv, past_key_values=d_kv, use_cache=True)
                d_kv = d_out.past_key_values
                d_next_tok = torch.argmax(d_out.logits[:, -1, :], dim=-1, keepdim=True)
                draft_kv_len = prompt_len + len(generated)
            else:
                # Roll back Target KV to keep only verified tokens before replacement_tok
                assert replacement_tok is not None
                generated.append(replacement_tok)
                target_kv_len += len(pending_prefix_for_target) + n_match
                t_kv = _crop_cache_inplace(t_kv, target_kv_len)
                pending_prefix_for_target = [replacement_tok]

                # Roll back Draft KV to verified length and feed replacement_tok
                draft_keep = prompt_len + (len(generated) - 1)
                d_kv = _crop_cache_inplace(d_kv, draft_keep)
                d_rep = torch.tensor([[replacement_tok]], device=self.draft_device)
                d_out = self.draft_model(d_rep, past_key_values=d_kv, use_cache=True)
                d_kv = d_out.past_key_values
                d_next_tok = torch.argmax(d_out.logits[:, -1, :], dim=-1, keepdim=True)
                draft_kv_len = draft_keep + 1

            if generated[-1] == self.tokenizer.eos_token_id:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t_start) * 1000.0
        n_gen = len(generated)
        tps = n_gen / (total_ms / 1000.0)
        alpha = total_accepted / max(1, total_drafted)
        mean_acc_step = n_gen / max(1, verifier_steps)
        if alpha >= 0.999:
            theo_step = float(gamma + 1)
        else:
            theo_step = (1.0 - (alpha ** (gamma + 1))) / max(1e-6, 1.0 - alpha)

        speedup = (tps / ar_baseline_tps) if (ar_baseline_tps and ar_baseline_tps > 0) else 1.0
        decode_ms = max(1e-3, total_ms - t_ttft)

        return SpeculativeRunMetrics(
            mode="dual_gpu_neural_speculative",
            gamma=gamma,
            prompt_tokens=prompt_len,
            generated_tokens=n_gen,
            total_time_ms=round(total_ms, 2),
            ttft_ms=round(t_ttft, 2),
            itl_ms=round(decode_ms / max(1, n_gen - 1), 2),
            tokens_per_sec=round(tps, 2),
            speedup_vs_ar=round(speedup, 3),
            drafted_tokens=total_drafted,
            accepted_tokens=total_accepted,
            acceptance_rate_alpha=round(alpha, 4),
            mean_accepted_per_step=round(mean_acc_step, 3),
            theoretical_tokens_per_step=round(theo_step, 3),
            verifier_steps=verifier_steps,
            apc_prefix_hit_pct=apc_pct,
            target_device=self.target_device,
            draft_device=self.draft_device,
            output_text=self.tokenizer.decode(generated, skip_special_tokens=True),
        )

    @torch.no_grad()
    def generate_prompt_lookup_speculative(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        gamma: int = 5,
        ngram_n: int = 3,
        enable_apc: bool = True,
        ar_baseline_tps: Optional[float] = None,
    ) -> SpeculativeRunMetrics:
        """Training-free Prompt-Lookup / Merkle N-Gram Speculative Decoding (MCSD style) on Target GPU.

        Ideal for grounded RAG generation where answers frequently copy phrases/numbers from retrieved context.
        Zero draft model VRAM overhead!
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_tokens = input_ids[0].tolist()
        prompt_len = len(prompt_tokens)

        # Build prompt n-gram lookup index (longest-suffix match)
        ngram_index: Dict[Tuple[int, ...], List[int]] = {}
        for i in range(max(0, prompt_len - ngram_n)):
            for n in range(ngram_n, 1, -1):
                if i + n + 1 <= prompt_len:
                    key = tuple(prompt_tokens[i : i + n])
                    if key not in ngram_index:
                        ngram_index[key] = prompt_tokens[i + n : i + n + gamma]

        t_start = time.perf_counter()
        t_logits, t_kv, t_ttft, apc_pct = self._prefill_with_apc(
            self.target_model, input_ids, self.target_device, "target", enable_apc=enable_apc
        )

        first_tok = int(torch.argmax(t_logits, dim=-1).item())
        generated: List[int] = [first_tok]
        target_kv_len = prompt_len
        pending_prefix = [first_tok]

        total_drafted = 0
        total_accepted = 0
        verifier_steps = 0

        while len(generated) < max_new_tokens and generated[-1] != self.tokenizer.eos_token_id:
            full_hist = prompt_tokens + generated
            draft_tokens: List[int] = []
            for n in range(ngram_n, 1, -1):
                if len(full_hist) >= n:
                    key = tuple(full_hist[-n:])
                    if key in ngram_index:
                        draft_tokens = ngram_index[key][: min(gamma, max_new_tokens - len(generated))]
                        break

            if not draft_tokens:
                # Single target step
                inp = torch.tensor([pending_prefix], device=self.target_device)
                out = self.target_model(inp, past_key_values=t_kv, use_cache=True)
                t_kv = out.past_key_values
                target_kv_len += len(pending_prefix)
                verifier_steps += 1
                nxt = int(torch.argmax(out.logits[0, -1, :], dim=-1).item())
                generated.append(nxt)
                pending_prefix = [nxt]
                continue

            total_drafted += len(draft_tokens)
            verify_seq = pending_prefix + draft_tokens
            inp = torch.tensor([verify_seq], device=self.target_device)
            out = self.target_model(inp, past_key_values=t_kv, use_cache=True)
            t_kv = out.past_key_values
            verifier_steps += 1

            n_match = 0
            replacement = None
            for idx, d_tok in enumerate(draft_tokens):
                t_pred = int(torch.argmax(out.logits[0, idx, :], dim=-1).item())
                if t_pred == d_tok:
                    n_match += 1
                    generated.append(t_pred)
                    if t_pred == self.tokenizer.eos_token_id or len(generated) >= max_new_tokens:
                        break
                else:
                    replacement = t_pred
                    break

            total_accepted += n_match
            if generated[-1] == self.tokenizer.eos_token_id or len(generated) >= max_new_tokens:
                break

            if n_match == len(draft_tokens):
                bonus = int(torch.argmax(out.logits[0, len(draft_tokens), :], dim=-1).item())
                generated.append(bonus)
                target_kv_len += len(pending_prefix) + len(draft_tokens)
                pending_prefix = [bonus]
            else:
                assert replacement is not None
                generated.append(replacement)
                target_kv_len += len(pending_prefix) + n_match
                t_kv = _crop_cache_inplace(t_kv, target_kv_len)
                pending_prefix = [replacement]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t_start) * 1000.0
        n_gen = len(generated)
        tps = n_gen / (total_ms / 1000.0)
        alpha = total_accepted / max(1, total_drafted) if total_drafted > 0 else 0.0
        mean_acc_step = n_gen / max(1, verifier_steps)
        speedup = (tps / ar_baseline_tps) if (ar_baseline_tps and ar_baseline_tps > 0) else 1.0
        decode_ms = max(1e-3, total_ms - t_ttft)

        return SpeculativeRunMetrics(
            mode="prompt_lookup_merkle_speculative",
            gamma=gamma,
            prompt_tokens=prompt_len,
            generated_tokens=n_gen,
            total_time_ms=round(total_ms, 2),
            ttft_ms=round(t_ttft, 2),
            itl_ms=round(decode_ms / max(1, n_gen - 1), 2),
            tokens_per_sec=round(tps, 2),
            speedup_vs_ar=round(speedup, 3),
            drafted_tokens=total_drafted,
            accepted_tokens=total_accepted,
            acceptance_rate_alpha=round(alpha, 4),
            mean_accepted_per_step=round(mean_acc_step, 3),
            theoretical_tokens_per_step=round(
                (1.0 - (alpha ** (gamma + 1))) / max(1e-6, 1.0 - alpha) if alpha < 0.999 else float(gamma + 1),
                3,
            ),
            verifier_steps=verifier_steps,
            apc_prefix_hit_pct=apc_pct,
            target_device=self.target_device,
            draft_device="cpu_ngram_merkle_lookup",
            output_text=self.tokenizer.decode(generated, skip_special_tokens=True),
        )


def probe_local_gguf_model(
    gguf_path: Path,
    prompt: str = "Summarize why late-interaction token matching beats mean pooling in 2 concise sentences.",
    n_predict: int = 48,
) -> Dict[str, Any]:
    """Run a quick timing/output check against a local AI Vault GGUF model via llama-cli."""
    llama_cli = Path("/home/soham/.local/bin/llama-cli")
    if not llama_cli.exists() or not gguf_path.exists():
        return {"available": False, "model": gguf_path.name}

    t0 = time.perf_counter()
    try:
        res = subprocess.run(
            [
                str(llama_cli),
                "-m",
                str(gguf_path),
                "-p",
                prompt,
                "-n",
                str(n_predict),
                "-t",
                "32",
                "--no-display-prompt",
                "-no-cnv",
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        out_text = (res.stdout or "").strip()
        return {
            "available": res.returncode == 0,
            "model": gguf_path.name,
            "elapsed_ms": round(elapsed_ms, 1),
            "output_preview": out_text[:400],
            "backend": "llama.cpp (32-core Threadripper)",
        }
    except Exception as exc:
        return {"available": False, "model": gguf_path.name, "error": str(exc)}


def run_speculative_benchmark_suite(
    target_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    gammas: Tuple[int, ...] = (2, 4, 6, 8),
    max_new_tokens: int = 72,
) -> Dict[str, Any]:
    """Benchmark Dual-GPU vLLM APC + Neural Speculative + Prompt-Lookup Speculative across gamma values."""
    engine = MultiGPUvLLMSpeculativeEngine(
        target_model_name=target_model,
        draft_model_name=draft_model,
    )

    shared_system_prefix = (
        "<|im_start|>system\n"
        "You are an enterprise compliance verification assistant. "
        "Quote exact policy figures, deadlines, and SLA thresholds directly from the provided context passages. "
        "Do not extrapolate outdated policies.\n<|im_end|>\n"
    )
    rag_context_prompt = (
        shared_system_prefix
        + "<|im_start|>user\nContext:\n"
        "[Passage 1]: Effective under the updated 2026 billing policy superseding prior charters, enterprise clients are granted "
        "14 calendar days from invoice delivery to submit a formal billing dispute through the customer portal. "
        "Approved billing credit memos are remitted to the client's primary treasury account within 30 calendar days of final audit sign-off. "
        "Submissions lacking a valid purchase order reference, cryptographic invoice hash, or executive comptroller attestation are automatically rejected.\n"
        "[Passage 2]: Under the active 2026 IT Equipment Policy, accidental damage to a corporate MacBook Pro carries a $250 (two hundred fifty dollars) employee deductible. "
        "Deductible charges are processed automatically through payroll deduction unless damage occurred during documented airline transit.\n\n"
        "Question: Under the 2026 policy, how many days are enterprise clients granted to submit a formal billing dispute, what happens if an attestation is missing, and what is the MacBook Pro deductible?\n"
        "Answer verbatim and concisely:<|im_end|>\n<|im_start|>assistant\n"
    )

    # Second prompt shares the exact same system + Passage 1 prefix blocks -> tests APC prefix hit!
    apc_followup_prompt = (
        shared_system_prefix
        + "<|im_start|>user\nContext:\n"
        "[Passage 1]: Effective under the updated 2026 billing policy superseding prior charters, enterprise clients are granted "
        "14 calendar days from invoice delivery to submit a formal billing dispute through the customer portal. "
        "Approved billing credit memos are remitted to the client's primary treasury account within 30 calendar days of final audit sign-off. "
        "Submissions lacking a valid purchase order reference, cryptographic invoice hash, or executive comptroller attestation are automatically rejected.\n"
        "[Passage 2]: Current 2026 CRD-9 clinical guidelines cap the adult daily oral maintenance regimen at 40 mg administered once daily.\n\n"
        "Question: State both the 2026 billing dispute window and the 2026 CRD-9 adult daily oral maintenance cap verbatim.<|im_end|>\n<|im_start|>assistant\n"
    )

    # Cold AR baseline vs Warm-APC AR baseline
    ar_cold = engine.generate_autoregressive(rag_context_prompt, max_new_tokens=max_new_tokens, enable_apc=False)
    ar_warm_1 = engine.generate_autoregressive(rag_context_prompt, max_new_tokens=max_new_tokens, enable_apc=True)
    ar_warm_2 = engine.generate_autoregressive(apc_followup_prompt, max_new_tokens=max_new_tokens, enable_apc=True)

    runs: List[Dict[str, Any]] = [asdict(ar_cold), asdict(ar_warm_1), asdict(ar_warm_2)]
    runs[0]["label"] = "AR Baseline (Cold Prefill, No APC)"
    runs[1]["label"] = "AR Baseline (APC Populated)"
    runs[2]["label"] = "AR Follow-up (Shared Prefix APC Hit)"

    for g in gammas:
        spec_neural = engine.generate_speculative_neural(
            rag_context_prompt,
            max_new_tokens=max_new_tokens,
            gamma=g,
            enable_apc=True,
            ar_baseline_tps=ar_warm_1.tokens_per_sec,
        )
        d_n = asdict(spec_neural)
        d_n["label"] = f"Dual-GPU Neural Speculative (gamma={g})"
        d_n["exact_match_with_ar"] = spec_neural.output_text == ar_warm_1.output_text
        runs.append(d_n)

        spec_lookup = engine.generate_prompt_lookup_speculative(
            rag_context_prompt,
            max_new_tokens=max_new_tokens,
            gamma=g,
            enable_apc=True,
            ar_baseline_tps=ar_warm_1.tokens_per_sec,
        )
        d_l = asdict(spec_lookup)
        d_l["label"] = f"Merkle Prompt-Lookup Speculative (gamma={g})"
        d_l["exact_match_with_ar"] = spec_lookup.output_text == ar_warm_1.output_text
        runs.append(d_l)

    gpu_telem = get_gpu_telemetry()
    apc_stats = engine.block_manager.stats()

    # Clean up model VRAM after benchmark completes so caller can allocate cleanly
    del engine.target_model
    del engine.draft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "target_model": target_model,
        "draft_model": draft_model,
        "has_native_vllm_pkg": importlib.util.find_spec("vllm") is not None,
        "gpu_telemetry_during_load": gpu_telem,
        "apc_block_manager_stats": apc_stats,
        "runs": runs,
    }
