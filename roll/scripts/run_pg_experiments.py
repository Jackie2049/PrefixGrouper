"""Phase 0: Baseline experiments for PrefixGrouper + ROLL integration.

Three experiments that must pass before any ROLL pipeline changes:

A. Baseline forward with flash_attention_2
B. Monkey-patched attention with prefix_grouper=None
C. PrefixGrouper grouped forward with logits restore

Usage:
    python roll/scripts/run_pg_experiments.py --experiment all     # Run all three
    python roll/scripts/run_pg_experiments.py --experiment baseline
    python roll/scripts/run_pg_experiments.py --experiment monkey_patch
    python roll/scripts/run_pg_experiments.py --experiment prefix_grouper
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# Add roll/ to path for the adapter
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from roll.utils.prefix_grouper import (
    PGBatch,
    PrefixGrouper,
    build_pg_from_micro_batch,
    forward_with_prefix_grouper,
    install_prefix_grouper_attention_patch,
    uninstall_prefix_grouper_attention_patch,
    reset_pg_call_count,
    get_pg_call_count,
)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_synthetic_batch(
    tokenizer,
    batch_size: int = 4,
    group_size: int = 2,
    prompt_len: int = 64,
    response_len: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict:
    """Create a synthetic ROLL-like batch for testing.

    *batch_size* prompts, each with *group_size* completions → total N = batch_size * group_size rows.

    Returns dict with keys: input_ids, attention_mask, prompt_mask, response_mask, position_ids, prefix_group_id.
    All tensors are right-padded.
    """
    set_seed(42)
    N = batch_size * group_size

    # Build input_ids
    # Each row: [prompt_tokens (right-padded), response_tokens (right-padded)]
    # Since we use right-padding, prompt is left-aligned, response follows
    max_seq_len = prompt_len + response_len

    input_ids = torch.full((N, max_seq_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((N, max_seq_len), dtype=torch.long, device=device)
    prompt_mask = torch.zeros((N, max_seq_len), dtype=torch.long, device=device)
    response_mask = torch.zeros((N, max_seq_len), dtype=torch.long, device=device)

    # Each prompt gets its own ID pattern (for group identification)
    prefix_group_id = torch.zeros(N, dtype=torch.long, device=device)

    for g in range(batch_size):
        base = g * group_size
        # Generate prompt tokens (non-pad)
        prompt_tokens = torch.randint(10, 100, (prompt_len,), device=device)
        for j in range(group_size):
            row = base + j
            # Right-padded prompt: fill from left
            input_ids[row, :prompt_len] = prompt_tokens
            attention_mask[row, :prompt_len] = 1
            prompt_mask[row, :prompt_len] = 1
            # Response tokens (different for each completion)
            resp_tokens = torch.randint(10, 100, (response_len,), device=device)
            input_ids[row, prompt_len:prompt_len + response_len] = resp_tokens
            attention_mask[row, prompt_len:prompt_len + response_len] = 1
            response_mask[row, prompt_len:prompt_len + response_len] = 1
            prefix_group_id[row] = g

    # Position IDs
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prompt_mask": prompt_mask,
        "response_mask": response_mask,
        "prefix_group_id": prefix_group_id,
    }


def make_synthetic_batch_v2(
    tokenizer,
    batch_size: int = 4,
    group_size: int = 2,
    min_prompt_len: int = 48,
    max_prompt_len: int = 80,
    min_response_len: int = 16,
    max_response_len: int = 48,
    device: str = "cuda",
) -> Dict:
    """Synthetic batch with varied prompt/response lengths for more realistic testing."""
    set_seed(42)
    N = batch_size * group_size
    max_seq_len = max_prompt_len + max_response_len

    input_ids = torch.full((N, max_seq_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((N, max_seq_len), dtype=torch.long, device=device)
    prompt_mask = torch.zeros((N, max_seq_len), dtype=torch.long, device=device)
    response_mask = torch.zeros((N, max_seq_len), dtype=torch.long, device=device)
    prefix_group_id = torch.zeros(N, dtype=torch.long, device=device)

    for g in range(batch_size):
        base = g * group_size
        p_len = torch.randint(min_prompt_len, max_prompt_len + 1, (1,)).item()
        prompt_tokens = torch.randint(10, 100, (p_len,), device=device)

        for j in range(group_size):
            row = base + j
            r_len = torch.randint(min_response_len, max_response_len + 1, (1,)).item()
            input_ids[row, :p_len] = prompt_tokens
            input_ids[row, p_len:p_len + r_len] = torch.randint(10, 100, (r_len,), device=device)
            attention_mask[row, :p_len + r_len] = 1
            prompt_mask[row, :p_len] = 1
            response_mask[row, p_len:p_len + r_len] = 1
            prefix_group_id[row] = g

    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
    return {
        "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids,
        "prompt_mask": prompt_mask, "response_mask": response_mask, "prefix_group_id": prefix_group_id,
    }


@torch.no_grad()
def compute_log_probs_and_loss(logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor):
    """Compute response log-probs and a simple GRPO-style loss."""
    labels = input_ids[:, 1:].clone()
    labels[response_mask[:, 1:] == 0] = 0  # mask non-response
    log_probs = F.log_softmax(logits.float(), dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_probs_labels = log_probs_labels * response_mask[:, 1:]
    loss = -log_probs_labels[log_probs_labels != 0].mean()
    return log_probs_labels, loss


def run_experiment_baseline(model, tokenizer, batch, dtype=torch.bfloat16, device="cuda"):
    """Experiment A: Standard forward with flash_attention_2."""
    model.eval()
    set_seed(42)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    position_ids = batch["position_ids"]

    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits

    log_probs, loss = compute_log_probs_and_loss(logits, input_ids, batch["response_mask"])
    return {"logits": logits, "log_probs": log_probs, "loss": loss.item()}


def run_experiment_monkey_patch(model, tokenizer, batch, dtype=torch.bfloat16, device="cuda"):
    """Experiment B: Monkey-patched attention with prefix_grouper=None."""
    # Install the patch
    install_prefix_grouper_attention_patch("flash_attention_2")

    model.eval()
    set_seed(42)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    position_ids = batch["position_ids"]

    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits

    log_probs, loss = compute_log_probs_and_loss(logits, input_ids, batch["response_mask"])

    # Clean up
    uninstall_prefix_grouper_attention_patch("flash_attention_2")

    return {"logits": logits, "log_probs": log_probs, "loss": loss.item()}


def run_experiment_prefix_grouper(model, tokenizer, batch, dtype=torch.bfloat16, device="cuda"):
    """Experiment C: PrefixGrouper grouped forward with logits restore."""
    # Install the patch
    install_prefix_grouper_attention_patch("flash_attention_2")

    model.eval()
    set_seed(42)

    # Build PrefixGrouper batch
    pg_batch = build_pg_from_micro_batch(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        prompt_mask=batch["prompt_mask"],
        response_mask=batch["response_mask"],
        prefix_group_id=batch["prefix_group_id"],
        group_size=2,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Forward with PrefixGrouper
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        restored_logits = forward_with_prefix_grouper(
            model=model,
            pg_batch=pg_batch,
            original_input_ids=batch["input_ids"],
        )

    log_probs, loss = compute_log_probs_and_loss(restored_logits, batch["input_ids"], batch["response_mask"])

    # Clean up
    uninstall_prefix_grouper_attention_patch("flash_attention_2")

    return {"logits": restored_logits, "log_probs": log_probs, "loss": loss.item(), "pg_batch": pg_batch}


def run_experiment_pg_count(model, tokenizer, batch, dtype=torch.bfloat16, device="cuda"):
    """Experiment D: Verify prefix_grouper is received by every layer's attention.

    Counts how many times the monkey-patched attention wrapper is called
    with a non-None ``prefix_grouper``. Expected count = num_layers × num_groups.
    """
    install_prefix_grouper_attention_patch("flash_attention_2")

    model.eval()
    set_seed(42)

    # Determine number of decoder layers
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, "num_layers"):
        num_layers = model.config.num_layers
    else:
        num_layers = None

    reset_pg_call_count()

    pg_batch = build_pg_from_micro_batch(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        prompt_mask=batch["prompt_mask"],
        response_mask=batch["response_mask"],
        prefix_group_id=batch["prefix_group_id"],
        group_size=2,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        restored_logits = forward_with_prefix_grouper(
            model=model,
            pg_batch=pg_batch,
            original_input_ids=batch["input_ids"],
        )

    call_count = get_pg_call_count()
    num_groups = pg_batch.num_groups

    print(f"\n{'='*60}")
    print(f"  Experiment D: PrefixGrouper attention call count")
    print(f"{'='*60}")
    print(f"  Model config says {num_layers} layers")
    print(f"  Batched   {num_groups} groups (in one grouped forward)")
    print(f"  Expected  calls = {num_layers} × 1 (one grouped forward) = {num_layers}")
    print(f"  Actual    calls: {call_count}")
    nz = (restored_logits.abs().sum(dim=-1) > 0).sum().item()
    total = restored_logits.size(0) * restored_logits.size(1)
    print(f"  Restored  logits shape: {list(restored_logits.shape)}, non_zero_pos: {nz}/{total}")

    passed = call_count == num_layers
    print(f"\n  Result: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"{'='*60}")

    uninstall_prefix_grouper_attention_patch("flash_attention_2")

    return call_count


def run_experiment_pg_train_equivalence(model, tokenizer, batch, dtype=torch.bfloat16, device="cuda"):
    """Experiment E: Compare baseline vs PG forward+restore with gradients enabled.

    Both paths: model.train(), forward → loss → backward.
    Compares loss and per-parameter gradient norms.
    """
    set_seed(42)
    model.train()

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    position_ids = batch["position_ids"]

    # ─── Baseline path (standard forward) ──────────────────────────────
    set_seed(42)
    model.zero_grad()
    with torch.autocast(device_type=device, dtype=dtype):
        logits_baseline = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits  # [4, 96, V]

    # Compute per-token CE loss on response tokens only
    # logits: [N, S, V] → labels: [N, S-1] (shift by 1 for next-token pred)
    # PG restored logits also has shape [N, S, V], so labels use same dims
    logits_flat = logits_baseline[:, :-1, :].reshape(-1, logits_baseline.size(-1))  # [N*(S-1), V]
    labels_flat = input_ids[:, 1:].reshape(-1)  # [N*(S-1)]
    # mask out pad tokens and non-response positions
    loss_mask = batch["response_mask"][:, 1:].reshape(-1).bool()
    masked_logits = logits_flat[loss_mask]
    masked_labels = labels_flat[loss_mask]
    loss_baseline = torch.nn.functional.cross_entropy(masked_logits.float(), masked_labels, reduction="mean")
    loss_baseline.backward()
    grad_norms_baseline = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]

    # ─── PG path (grouped forward + restore) ──────────────────────────
    install_prefix_grouper_attention_patch("flash_attention_2")

    set_seed(42)
    model.zero_grad()
    pg_batch = build_pg_from_micro_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_mask=batch["prompt_mask"],
        response_mask=batch["response_mask"],
        prefix_group_id=batch["prefix_group_id"],
        group_size=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.autocast(device_type=device, dtype=dtype):
        restored_logits = forward_with_prefix_grouper(
            model=model,
            pg_batch=pg_batch,
            original_input_ids=input_ids,
        )
    logits_flat_pg = restored_logits[:, :-1, :].reshape(-1, logits_baseline.size(-1))
    masked_logits_pg = logits_flat_pg[loss_mask]
    loss_pg = torch.nn.functional.cross_entropy(masked_logits_pg.float(), masked_labels, reduction="mean")
    loss_pg.backward()
    grad_norms_pg = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]

    uninstall_prefix_grouper_attention_patch("flash_attention_2")

    # ─── Compare ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Experiment E: Train-mode equivalence (baseline vs PG)")
    print(f"{'='*60}")
    print(f"  Baseline loss:  {loss_baseline.item():.8f}")
    print(f"  PG       loss:  {loss_pg.item():.8f}")
    loss_diff = abs(loss_pg.item() - loss_baseline.item())
    print(f"  Loss     diff:  {loss_diff:.8f}")

    # Compare gradient norms
    n_params = min(len(grad_norms_baseline), len(grad_norms_pg))
    grad_diffs = [abs(grad_norms_baseline[i] - grad_norms_pg[i]) for i in range(n_params)]
    max_grad_diff = max(grad_diffs) if grad_diffs else 0.0
    mean_grad_diff = sum(grad_diffs) / len(grad_diffs) if grad_diffs else 0.0
    print(f"  Max grad norm diff:  {max_grad_diff:.8f}")
    print(f"  Mean grad norm diff: {mean_grad_diff:.8f}")
    print(f"  Baseline grad_norm[0]: {grad_norms_baseline[0]:.6f}" if grad_norms_baseline else "  No gradients in baseline")
    print(f"  PG       grad_norm[0]: {grad_norms_pg[0]:.6f}" if grad_norms_pg else "  No gradients in PG")

    # BF16 tolerance check
    rtol, atol = 2e-2, 2e-2
    loss_pass = loss_diff <= max(atol, rtol * abs(loss_baseline.item()))
    grad_pass = max_grad_diff <= max(atol, rtol * max(grad_norms_baseline + [1.0]))
    print(f"\n  Loss within BF16 rtol={rtol}, atol={atol}? {'✅ PASS' if loss_pass else '❌ FAIL'}")
    print(f"  Grads within BF16 rtol={rtol}, atol={atol}? {'✅ PASS' if grad_pass else '❌ FAIL'}")
    print(f"{'='*60}")

    return {"loss_baseline": loss_baseline.item(), "loss_pg": loss_pg.item(),
            "max_grad_diff": max_grad_diff, "mean_grad_diff": mean_grad_diff}


def compare_results(name: str, baseline: dict, result: dict, rtol: float = 1e-2, atol: float = 1e-3):
    """Compare experimental results to baseline. Print pass/fail."""
    loss_diff = abs(result["loss"] - baseline["loss"])
    max_logprob_diff = abs(result["log_probs"] - baseline["log_probs"]).max().item()
    mean_logprob_diff = abs(result["log_probs"] - baseline["log_probs"]).mean().item()

    loss_pass = loss_diff <= max(atol, rtol * abs(baseline["loss"]))
    logprob_pass = max_logprob_diff <= max(atol, rtol * abs(baseline["log_probs"]).max().item())

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Baseline loss:       {baseline['loss']:.8f}")
    print(f"  {name} loss:         {result['loss']:.8f}")
    print(f"  Loss diff:           {loss_diff:.8f}  {'PASS' if loss_pass else 'FAIL'}")
    print(f"  Max logprob diff:    {max_logprob_diff:.6f}  {'PASS' if logprob_pass else 'FAIL'}")
    print(f"  Mean logprob diff:   {mean_logprob_diff:.6f}")
    print(f"  Logit max diff:      {(result['logits'] - baseline['logits']).abs().max().item():.6f}")
    print(f"  Logit mean diff:     {(result['logits'] - baseline['logits']).abs().mean().item():.6f}")

    return loss_pass and logprob_pass


def main():
    parser = argparse.ArgumentParser(description="PrefixGrouper Phase 0 experiments")
    parser.add_argument("--experiment", choices=["all", "baseline", "monkey_patch", "prefix_grouper", "pg_count", "train_equivalence"], default="all")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=2, help="Number of groups (each with G=2 rows)")
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--response_len", type=int, default=32)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Save results JSON to path")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = args.device
    print(f"Device: {device}  |  Dtype: {args.dtype}")
    print(f"Batch: {args.batch_size} groups × {args.group_size} = {args.batch_size * args.group_size} rows")

    # Skip GPU experiments on CPU
    if device == "cpu":
        print("WARNING: Running on CPU. PrefixGrouper attention requires GPU for meaningful results.")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Create batch
    batch = make_synthetic_batch(
        tokenizer,
        batch_size=args.batch_size,
        group_size=args.group_size,
        prompt_len=args.prompt_len,
        response_len=args.response_len,
        device=device,
        dtype=dtype,
    )
    print(f"Input shape: {batch['input_ids'].shape}")

    results = {}
    all_pass = True

    # Experiment A: Baseline
    if args.experiment in ("all", "baseline"):
        print("\n--- Experiment A: Baseline forward ---")
        t0 = time.time()
        baseline = run_experiment_baseline(model, tokenizer, batch, dtype=dtype, device=device)
        t_baseline = time.time() - t0
        print(f"  Loss: {baseline['loss']:.8f}  |  Time: {t_baseline:.3f}s")
        results["baseline"] = {"loss": baseline["loss"], "time": t_baseline}
    else:
        baseline = run_experiment_baseline(model, tokenizer, batch, dtype=dtype, device=device)
        results["baseline"] = {"loss": baseline["loss"]}

    # Experiment B: Monkey-patch fallback
    if args.experiment in ("all", "monkey_patch"):
        print("\n--- Experiment B: Monkey-patch (prefix_grouper=None) ---")
        t0 = time.time()
        monkey = run_experiment_monkey_patch(model, tokenizer, batch, dtype=dtype, device=device)
        t_monkey = time.time() - t0
        pass_b = compare_results("Monkey-patch vs Baseline", baseline, monkey, rtol=1e-4, atol=1e-5)
        results["monkey_patch"] = {"loss": monkey["loss"], "time": t_monkey, "pass": pass_b}
        all_pass = all_pass and pass_b
    else:
        monkey = None

    # Experiment C: PrefixGrouper
    if args.experiment in ("all", "prefix_grouper"):
        print("\n--- Experiment C: PrefixGrouper grouped forward ---")
        t0 = time.time()
        pg = run_experiment_prefix_grouper(model, tokenizer, batch, dtype=dtype, device=device)
        t_pg = time.time() - t0
        pass_c = compare_results("PrefixGrouper vs Baseline", baseline, pg, rtol=2e-2, atol=2e-2)
        results["prefix_grouper"] = {"loss": pg["loss"], "time": t_pg, "pass": pass_c}
        all_pass = all_pass and pass_c

        # Also log group statistics
        pg_batch = pg["pg_batch"]
        print(f"\n  Group statistics:")
        print(f"    Groups: {pg_batch.num_groups}")
        print(f"    Grouped input shape: {pg_batch.input_ids.shape}")
        print(f"    Original total tokens: {batch['input_ids'].ne(tokenizer.pad_token_id).sum().item()}")
        print(f"    Grouped total tokens: {pg_batch.input_ids.ne(tokenizer.pad_token_id).sum().item()}")
        compression = 1 - pg_batch.input_ids.ne(tokenizer.pad_token_id).sum().item() / batch["input_ids"].ne(tokenizer.pad_token_id).sum().item()
        print(f"    Token compression: {compression:.2%}")

    # Experiment D: PG attention count
    if args.experiment in ("all", "pg_count"):
        print("\n--- Experiment D: PG attention call count (参数透传) ---")
        call_count = run_experiment_pg_count(model, tokenizer, batch, dtype=dtype, device=device)
        results["pg_count"] = {"call_count": call_count}
        all_pass = all_pass and (call_count > 0)

    # Experiment E: Train-mode equivalence
    if args.experiment in ("all", "train_equivalence"):
        print("\n--- Experiment E: Train-mode equivalence ---")
        eq_results = run_experiment_pg_train_equivalence(model, tokenizer, batch, dtype=dtype, device=device)
        results["train_equivalence"] = eq_results
        # Determine pass/fail based on BF16 tolerance
        rtol, atol = 2e-2, 2e-2
        loss_diff = abs(eq_results["loss_pg"] - eq_results["loss_baseline"])
        loss_pass = loss_diff <= max(atol, rtol * abs(eq_results["loss_baseline"]))
        grad_pass = eq_results["max_grad_diff"] <= max(atol, rtol * (1.0 if eq_results["loss_baseline"] == 0 else abs(eq_results["loss_baseline"])))
        all_pass = all_pass and loss_pass and grad_pass

    print(f"\n{'='*60}")
    print(f"  OVERALL: {'ALL PASSED' if all_pass else 'SOME EXPERIMENTS FAILED'}")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w") as f:
            # Remove tensors before saving
            json_safe = {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, bool, str))}
                         for k, v in results.items()}
            json.dump(json_safe, f, indent=2)
        print(f"Results saved to {args.output}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
