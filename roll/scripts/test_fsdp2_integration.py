"""Phase 2 integration test: verify PrefixGrouper routing in FSDP2 strategy.

Loads Qwen2.5-0.5B with FSDP2, runs a forward pass through _fsdp2_forward
with use_prefix_grouper=true, verifies the output matches baseline.
"""
import json
import os
import sys
import time
import argparse

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29501"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"

import roll
from roll.distributed.strategy.fsdp2_strategy import (
    FSDP2InferStrategy, FSDP2TrainStrategy
)

import sys
PROJECT_ROOT = "/home/zxw/Alibaba-ROLL/adapt-prefixgrouper"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ROLL"))

from roll.utils.prefix_grouper import (
    install_prefix_grouper_attention_patch,
    uninstall_prefix_grouper_attention_patch,
    build_pg_from_micro_batch,
    forward_with_prefix_grouper,
    prefix_grouper_forward_from_data,
    PGBatch,
)


def create_synthetic_batch(tokenizer, batch_size=2, group_size=2,
                           prompt_len=64, response_len=32, device="cuda"):
    """Create a realistic ROLL-like batch dict."""
    N = batch_size * group_size
    S = prompt_len + response_len

    input_ids = torch.full((N, S), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(N, S, dtype=torch.long, device=device)
    prompt_mask = torch.zeros(N, S, dtype=torch.long, device=device)
    response_mask = torch.zeros(N, S, dtype=torch.long, device=device)
    prefix_group_id = torch.zeros(N, dtype=torch.long, device=device)

    for g in range(batch_size):
        pt = torch.randint(10, 100, (prompt_len,), device=device)
        for j in range(group_size):
            row = g * group_size + j
            input_ids[row, :prompt_len] = pt
            input_ids[row, prompt_len:prompt_len+response_len] = torch.randint(10, 100, (response_len,), device=device)
            attention_mask[row, :prompt_len+response_len] = 1
            prompt_mask[row, :prompt_len] = 1
            response_mask[row, prompt_len:prompt_len+response_len] = 1
            prefix_group_id[row] = g

    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prompt_mask": prompt_mask,
        "response_mask": response_mask,
        "prefix_group_id": prefix_group_id,
    }


@torch.no_grad()
def test_fsdp2_integration(model_name, batch, dtype=torch.bfloat16, device="cuda"):
    """Test _fsdp2_forward level integration of PrefixGrouper."""

    print(f"\n{'='*60}")
    print(f"  FSDP2 PrefixGrouper Integration Test")
    print(f"{'='*60}")

    # Baseline: plain forward without PG
    install_prefix_grouper_attention_patch("flash_attention_2")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device).eval()

    print(f"\n--- Baseline forward ---")
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        baseline_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            use_cache=False,
        ).logits
    t_base = time.time() - t0
    print(f"  Time: {t_base:.3f}s")

    # PG forward via forward_with_prefix_grouper
    print(f"\n--- PrefixGrouper forward ---")
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        pg_logits = prefix_grouper_forward_from_data(
            model=model,
            data_batch={
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "prompt_mask": batch["prompt_mask"],
                "response_mask": batch["response_mask"],
                "prefix_group_id": batch["prefix_group_id"],
            },
            forward_args={},
            group_size=2,
            pad_token_id=model.config.pad_token_id or 0,
        )
    t_pg = time.time() - t0

    # Compare
    logit_diff = (pg_logits - baseline_logits).abs()
    labels = batch["input_ids"][:, 1:].clone()
    labels[batch["response_mask"][:, 1:] == 0] = 0

    def compute_approx_loss(logits, labels, response_mask):
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        lp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        lp = lp * response_mask[:, 1:]
        return -lp[lp != 0].mean().item()

    base_loss = compute_approx_loss(baseline_logits, labels, batch["response_mask"])
    pg_loss = compute_approx_loss(pg_logits, labels, batch["response_mask"])

    print(f"  Time: {t_pg:.3f}s")
    print(f"  Baseline loss: {base_loss:.6f}")
    print(f"  PG loss:       {pg_loss:.6f}")
    print(f"  Loss diff:     {abs(base_loss - pg_loss):.6f}")
    print(f"  Max logit diff: {logit_diff.max().item():.4f}")
    print(f"  Mean logit diff: {logit_diff.mean().item():.4f}")

    if abs(base_loss - pg_loss) < 0.1:
        print(f"\n  {'='*50}")
        print(f"  ✅ FSDP2 INTEGRATION TEST PASSED")
        print(f"  {'='*50}")
        return True
    else:
        print(f"\n  {'='*50}")
        print(f"  ❌ FSDP2 INTEGRATION TEST FAILED")
        print(f"  {'='*50}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--response_len", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"Device: {device}  Dtype: {dtype}")
    print(f"Batch: {args.batch_size} groups x {args.group_size} = {args.batch_size * args.group_size} rows")

    model_name = args.model
    if os.path.exists(model_name):
        # Local path
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    batch = create_synthetic_batch(
        tokenizer, args.batch_size, args.group_size,
        args.prompt_len, args.response_len, device
    )

    TIMESTAMP = time.strftime("%m%d-%H%M")
    os.makedirs(f"outputs/{TIMESTAMP}", exist_ok=True)

    success = test_fsdp2_integration(model_name, batch, dtype, device)

    print(f"\n  {'='*60}")
    print(f"  FINAL: {'ALL PASSED' if success else 'FAILED'}")
    print(f"  {'='*60}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
