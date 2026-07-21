"""ROLL adapter for PrefixGrouper.

Provides the integration layer between ROLL FSDP2 training and PrefixGrouper's
shared-prefix attention. Implements the four modules described in docs/adapt-roll.md:

1. Attention monkey-patch (idempotent)
2. Continuous group builder from ROLL DataProto
3. Grouped forward + logits restore
4. Unified FSDP2 strategy helper
"""

import contextlib
import importlib.metadata
from contextvars import ContextVar
from typing import Optional, Tuple, List, TYPE_CHECKING

import torch
import torch.nn as nn
from packaging.version import Version
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from prefix_grouper import PrefixGrouper
from prefix_grouper.utils.register_transformers import register_attention as pg_register_attention

# ---------------------------------------------------------------------------
# Module 1: Idempotent attention monkey-patch
# ---------------------------------------------------------------------------

_ORIGINAL_ATTN_FUNCS: dict = {}
_PATCHED_IMPLEMENTATIONS: set = set()
_MIN_FLASH_ATTN_VERSION = Version("2.7.4.post1")

# ── Spy counters for Phase 1.2 integration tests ──────────────
# pg_outer_calls:  wrapper sees non-None prefix_grouper (per-layer PG route entry)
# plain_fallback_calls: wrapper sees None prefix_grouper, NOT inside PG (plain forward)
# delegate_calls: wrapper sees None prefix_grouper, INSIDE PG (orig attn called by AttentionForward)
_PG_ACTIVE_DEPTH: ContextVar[int] = ContextVar("prefix_grouper_active_depth", default=0)
_PG_OUTER_CALLS: int = 0
_PLAIN_FALLBACK_CALLS: int = 0
_DELEGATE_CALLS: int = 0


def _validate_fa2_runtime(module: nn.Module) -> None:
    """Fail fast unless the model can provide right-aligned FA2 causality.

    PrefixGrouper suffix attention has ``Q=Ri`` and ``K/V=[P,Ri]``.  Its
    correctness and performance both require FA2's post-2.1 bottom-right
    causal behaviour; SDPA and explicit-mask fallbacks are intentionally not
    part of this production adapter.
    """
    try:
        installed_version = Version(importlib.metadata.version("flash-attn"))
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "PrefixGrouper requires flash-attn>=2.7.4.post1; install the verified "
            "verl-compatible FA2 runtime and restart all ROLL workers."
        ) from exc
    if installed_version < _MIN_FLASH_ATTN_VERSION:
        raise RuntimeError(
            f"PrefixGrouper requires flash-attn>={_MIN_FLASH_ATTN_VERSION}, found "
            f"{installed_version}. Older FA2 uses incorrect top-left causal alignment "
            "for suffix Q_len < K_len."
        )
    if not getattr(module, "is_causal", False):
        raise RuntimeError("PrefixGrouper requires causal FlashAttention modules")
    if getattr(module, "_flash_attn_uses_top_left_mask", False):
        raise RuntimeError(
            "PrefixGrouper requires right-aligned FlashAttention causality "
            "(_flash_attn_uses_top_left_mask=False). Reload the model after upgrading FA2."
        )


def reset_spy_counts():
    """Zero all three spy counters."""
    global _PG_OUTER_CALLS, _PLAIN_FALLBACK_CALLS, _DELEGATE_CALLS
    _PG_OUTER_CALLS = 0
    _PLAIN_FALLBACK_CALLS = 0
    _DELEGATE_CALLS = 0


def get_spy_counts() -> dict:
    """Return dict with current spy counts."""
    return {
        "pg_outer_calls": _PG_OUTER_CALLS,
        "plain_fallback_calls": _PLAIN_FALLBACK_CALLS,
        "delegate_calls": _DELEGATE_CALLS,
    }


def install_prefix_grouper_attention_patch(attn_implementation: str = "flash_attention_2"):
    """Monkey-patch the Transformers attention function to support ``prefix_grouper`` kwarg.

    Idempotent — calling twice on the same ``attn_implementation`` is a no-op.

    The wrapper checks for ``prefix_grouper`` in the attention kwargs:
    - ``None`` → calls the original attention (no change to baseline).
    - otherwise → delegates to PrefixGrouper's ``AttentionForward``, using the original
      attention as the inner function for the per-suffix call.
    """
    if attn_implementation in _PATCHED_IMPLEMENTATIONS:
        return

    # Ensure the prefix_grouper_attention is registered in Transformers
    pg_register_attention()
    pg_fn = ALL_ATTENTION_FUNCTIONS.get("prefix_grouper_attention")
    if pg_fn is None:
        raise RuntimeError("prefix_grouper_attention not found after register_attention()")
    _ORIGINAL_ATTN_FUNCS["prefix_grouper_attention"] = pg_fn

    # Save original and replace
    orig_fn = ALL_ATTENTION_FUNCTIONS.get(attn_implementation)
    if orig_fn is None:
        raise RuntimeError(f"Attention implementation '{attn_implementation}' not found")
    _ORIGINAL_ATTN_FUNCS[attn_implementation] = orig_fn

    def _wrapped_fn(module, query, key, value, attention_mask, *args, **kwargs):
        global _PG_OUTER_CALLS, _PLAIN_FALLBACK_CALLS, _DELEGATE_CALLS
        prefix_grouper = kwargs.pop("prefix_grouper", None)
        pg_attn_func = kwargs.pop("prefix_grouper_attn_func", None)
        if prefix_grouper is None:
            if _PG_ACTIVE_DEPTH.get() > 0:
                _DELEGATE_CALLS += 1
                # Both PrefixGrouper branches must run the original FA2
                # implementation. A verified FA2 runtime provides right-
                # aligned causal semantics for the suffix's Q_len < K_len.
                return orig_fn(module, query, key, value, attention_mask, *args, **kwargs)
            else:
                _PLAIN_FALLBACK_CALLS += 1
            return orig_fn(module, query, key, value, attention_mask, *args, **kwargs)
        # When prefix_grouper is active, delegate to the registered PG forward.
        # It will call back into orig_fn for each per-suffix attention.
        _PG_OUTER_CALLS += 1
        _validate_fa2_runtime(module)
        active_token = _PG_ACTIVE_DEPTH.set(_PG_ACTIVE_DEPTH.get() + 1)
        try:
            return pg_fn(
                module, query, key, value, attention_mask, *args,
                prefix_grouper=prefix_grouper,
                prefix_grouper_attn_func=pg_attn_func or attn_implementation,
                **kwargs,
            )
        finally:
            _PG_ACTIVE_DEPTH.reset(active_token)

    ALL_ATTENTION_FUNCTIONS[attn_implementation] = _wrapped_fn
    _PATCHED_IMPLEMENTATIONS.add(attn_implementation)


def uninstall_prefix_grouper_attention_patch(attn_implementation: str = "flash_attention_2"):
    """Restore the original attention function. Idempotent."""
    if attn_implementation not in _PATCHED_IMPLEMENTATIONS:
        return
    if attn_implementation in _ORIGINAL_ATTN_FUNCS:
        ALL_ATTENTION_FUNCTIONS[attn_implementation] = _ORIGINAL_ATTN_FUNCS[attn_implementation]
    _PATCHED_IMPLEMENTATIONS.discard(attn_implementation)


# ---------------------------------------------------------------------------
# Module 2: Continuous group builder from ROLL DataProto
# ---------------------------------------------------------------------------

class PGBatch:
    """Container for a PrefixGrouper micro-batch.

    Attributes:
        grouper: PrefixGrouper instance for this micro-batch.
        input_ids: Grouped input IDs [num_groups, max_total_len].
        attention_mask: Grouped attention mask [num_groups, max_total_len].
        position_ids: Manually constructed 2D position IDs [num_groups, max_total_len].
        group_map: List of (start_row, end_row, group_id) tuples mapping to original rows.
        group_size: Expected number of completions per group.
    """

    def __init__(
        self,
        grouper: PrefixGrouper,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        group_map: List[Tuple[int, int, int]],
        group_size: int,
    ):
        self.grouper = grouper
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.group_map = group_map
        self.group_size = group_size

    @property
    def num_groups(self) -> int:
        return len(self.group_map)


def _find_continuous_runs(ids: torch.Tensor) -> List[Tuple[int, int, int]]:
    """Find continuous runs of the same ID.

    Returns list of (start_row, end_row, group_id) — ``end_row`` is exclusive.
    """
    if ids.numel() == 0:
        return []
    runs = []
    start = 0
    for i in range(1, ids.size(0)):
        if ids[i] != ids[start]:
            runs.append((start, i, int(ids[start])))
            start = i
    runs.append((start, ids.size(0), int(ids[start])))
    return runs


def _separate_prompt_response(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separate input_ids into prompt and response parts.

    Returns:
        prompt_ids: [N, max_prompt_len] — padded with pad_token_id.
        response_ids: [N, max_response_len] — padded with pad_token_id.
        prompt_mask: [N, max_prompt_len].
        response_mask_raw: [N, max_response_len].
    """
    prompt_mask = attention_mask.bool() & ~response_mask.bool()
    response_mask_raw = attention_mask.bool() & response_mask.bool()

    max_prompt_len = int(prompt_mask.sum(dim=1).max().item())
    max_response_len = int(response_mask_raw.sum(dim=1).max().item())

    batch_size = input_ids.size(0)
    prompt_ids = torch.full((batch_size, max_prompt_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
    response_ids = torch.full((batch_size, max_response_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
    prompt_mask_out = torch.zeros((batch_size, max_prompt_len), dtype=input_ids.dtype, device=input_ids.device)
    response_mask_out = torch.zeros((batch_size, max_response_len), dtype=input_ids.dtype, device=input_ids.device)

    for i in range(batch_size):
        p_len = int(prompt_mask[i].sum().item())
        r_len = int(response_mask_raw[i].sum().item())
        prompt_ids[i, :p_len] = input_ids[i, prompt_mask[i]]
        response_ids[i, :r_len] = input_ids[i, response_mask_raw[i]]
        prompt_mask_out[i, :p_len] = 1
        response_mask_out[i, :r_len] = 1

    return prompt_ids, response_ids, prompt_mask_out, response_mask_out


def build_pg_from_micro_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
    response_mask: torch.Tensor,
    prefix_group_id: torch.Tensor,
    group_size: int,
    pad_token_id: int,
) -> PGBatch:
    """Parse continuous ``prefix_group_id`` runs into a ``PGBatch``.

    Args:
        input_ids: [N, S] right-padded.
        attention_mask: [N, S] right-padded.
        prompt_mask: [N, S] — 1 for prompt tokens.
        response_mask: [N, S] — 1 for response tokens.
        prefix_group_id: [N] — group IDs, continuous runs of length == group_size.
        group_size: Expected completions per prompt.
        pad_token_id: Token ID used for padding.

    Returns:
        PGBatch with grouped tensors and PrefixGrouper instance.

    Raises:
        ValueError: If runs don't match ``group_size``, or prompt tokens differ within a run.
    """
    device = input_ids.device
    runs = _find_continuous_runs(prefix_group_id)

    # Validate each run
    for start, end, gid in runs:
        run_len = end - start
        if run_len != group_size:
            raise ValueError(
                f"Group {gid} has {run_len} samples, expected {group_size}. "
                f"All groups must have exactly {group_size} samples when prefix_grouper is enabled."
            )
        # Verify same prompt within the run
        run_prompts = prompt_mask[start:end]
        run_input_ids = input_ids[start:end]
        for j in range(1, run_len):
            # Compare the actual prompt token IDs (masked by prompt_mask)
            p0_tokens = run_input_ids[0][run_prompts[0].bool()]
            pj_tokens = run_input_ids[j][run_prompts[j].bool()]
            if not torch.equal(p0_tokens, pj_tokens):
                raise ValueError(f"Group {gid} has mismatched prompt tokens between samples 0 and {j}.")

    # Separate prompt and response
    prompt_ids, response_ids, p_mask, r_mask = _separate_prompt_response(
        input_ids, attention_mask, response_mask, pad_token_id
    )

    # Build PrefixGrouper for each group
    group_prompts = []
    group_responses = []
    group_info_list = []

    for start, end, gid in runs:
        p_ids = prompt_ids[start]
        p_len = int(p_mask[start].sum().item())
        r_ids = response_ids[start:end]
        r_lens = [int(r_mask[start + j].sum().item()) for j in range(end - start)]

        group_prompts.append(p_ids[:p_len])  # no padding in group
        group_responses.append(torch.cat([r_ids[j, :r_lens[j]] for j in range(end - start)]))
        group_info_list.append([p_len, *r_lens])

    # Create PrefixGrouper
    grouper = PrefixGrouper(group_info=group_info_list, device=device)

    # Build grouped input_ids
    max_total_len = max(grouper.total_lens).item()
    num_groups = len(runs)
    grouped_input_ids = torch.full((num_groups, max_total_len), pad_token_id, dtype=input_ids.dtype, device=device)
    grouped_attention_mask = torch.zeros((num_groups, max_total_len), dtype=input_ids.dtype, device=device)
    grouped_position_ids = torch.zeros((num_groups, max_total_len), dtype=torch.long, device=device)

    for g in range(num_groups):
        start, end, _ = runs[g]
        p_len = int(p_mask[start].sum().item())
        r_lens = [int(r_mask[start + j].sum().item()) for j in range(end - start)]

        # Fill prompt
        p_tokens = input_ids[start, :p_len]
        grouped_input_ids[g, :p_len] = p_tokens
        grouped_attention_mask[g, :p_len] = 1
        grouped_position_ids[g, :p_len] = torch.arange(0, p_len, device=device)

        # Fill each suffix, resetting position IDs to p_len each time
        offset = p_len
        for j in range(end - start):
            r_len = r_lens[j]
            r_start = int(prompt_mask[start + j].sum().item())
            r_tokens = input_ids[start + j, r_start:r_start + r_len]
            grouped_input_ids[g, offset:offset + r_len] = r_tokens
            grouped_attention_mask[g, offset:offset + r_len] = 1
            grouped_position_ids[g, offset:offset + r_len] = torch.arange(p_len, p_len + r_len, device=device)
            offset += r_len

    return PGBatch(
        grouper=grouper,
        input_ids=grouped_input_ids,
        attention_mask=grouped_attention_mask,
        position_ids=grouped_position_ids,
        group_map=runs,
        group_size=group_size,
    )


# ---------------------------------------------------------------------------
# Module 3: Grouped forward + logits restore
# ---------------------------------------------------------------------------


def forward_with_prefix_grouper(
    model: nn.Module,
    pg_batch: PGBatch,
    original_input_ids: torch.Tensor,
    forward_args: Optional[dict] = None,
) -> torch.Tensor:
    """Run grouped forward with PrefixGrouper and restore logits to original layout.

    Args:
        model: HF model (already on correct device/dtype).
        pg_batch: Built by ``build_pg_from_micro_batch``.
        original_input_ids: [N, S] — the original ungrouped input IDs for scatter reference.
        forward_args: Additional model kwargs (copied to avoid in-place contamination).

    Returns:
        Restored logits [N, S, V] matching the original ``input_ids`` layout.
    """
    device = next(model.parameters()).device
    fwd_args = dict(forward_args) if forward_args else {}
    fwd_args.pop("use_cache", None)
    # Only the grouped position IDs should be passed
    fwd_args.pop("position_ids", None)

    # Run grouped forward
    grouped_logits = model(
        input_ids=pg_batch.input_ids.to(device),
        attention_mask=pg_batch.attention_mask.to(device),
        position_ids=pg_batch.position_ids.to(device),
        use_cache=False,
        prefix_grouper=pg_batch.grouper,
        **_filter_forward_args(fwd_args, model.forward),
    ).logits  # [num_groups, max_total_len, V]

    # ─── Logits restore ───────────────────────────────────────────────
    # ROLL loss convention (op_compute_log_probs):
    #   labels = input_ids[:, 1:]
    #   logits[:, k, :] predicts token at position k+1 = labels[:, k]
    # So logits[:, p_len-1, :] predicts input_ids[:, p_len] = first response token.
    return restore_grouped_logits(grouped_logits, pg_batch, original_input_ids.shape)


def restore_grouped_logits(
    grouped_logits: torch.Tensor,
    pg_batch: PGBatch,
    original_shape: Tuple[int, int],
) -> torch.Tensor:
    """Restore grouped logits to original ROLL layout [N, S, V].

    Uses PrefixGrouper's ``split_output(include_prefix_last=1)`` so that every
    completion's first-response-token prediction comes from the shared prefix's
    last hidden state (``grouped[p_len-1]``), matching baseline ``[P,R]``
    semantics. The previous manual restore used ``offset-1`` per completion,
    which pointed at the previous completion's last token (e.g. R1[-1] for
    R2[0]) — a real semantic bug, not a tolerance issue.

    Pure tensor operation — no model forward, no side effects.
    Can be tested CPU-only with synthetic data.

    Args:
        grouped_logits: [num_groups, max_total_len, V]
        pg_batch: PGBatch from ``build_pg_from_micro_batch``.
        original_shape: (N, S) — original ungrouped input_ids shape.

    Returns:
        Restored logits [N, S, V].
    """
    N, S, V = original_shape[0], original_shape[1], grouped_logits.shape[-1]
    restored_logits = torch.zeros(N, S, V, dtype=grouped_logits.dtype, device=grouped_logits.device)

    # split_output returns per-sample suffix with prefix_last prepended:
    #   suffix_out[j, 0]      = P[-1] prediction (grouped[p_len-1])
    #   suffix_out[j, 1..r_j] = Rj[0..r_j-1] predictions
    _, _, suffix_out, _ = pg_batch.grouper.split_output(
        grouped_logits, include_prefix_last=1)

    sample_idx = 0
    for g_idx, (start, end, _) in enumerate(pg_batch.group_map):
        p_len = int(pg_batch.grouper.prefix_lens[g_idx].item())

        # Prefix next-token predictions — logits[0:p_len-1] predict prompt[1:p_len].
        # All completions in this group share the same prefix prediction.
        if p_len > 1:
            prefix_logits = grouped_logits[g_idx, :p_len - 1, :].unsqueeze(0)
            for j in range(end - start):
                restored_logits[start + j, :p_len - 1, :] = prefix_logits

        # Response prediction logits per sample.
        # suffix_out[sample_idx] has length r_len+1: [P[-1] pred, R[0] pred, ..., R[-1] pred].
        # ROLL layout: restored[row, p_len-1+k] predicts R[k], k=0..r_len-1.
        #   restored[row, p_len-1]     = suffix_out[sample_idx, 0]  = P[-1] pred → R[0]
        #   restored[row, p_len-1+k]   = suffix_out[sample_idx, k]  = R[k-1] pred → R[k]
        for j in range(end - start):
            r_len = int(pg_batch.grouper.ungrouped_suffix_lens[sample_idx].item())
            if r_len > 0:
                tgt_start = p_len - 1
                tgt_size = min(r_len, S - tgt_start)
                if tgt_size > 0:
                    restored_logits[start + j, tgt_start : tgt_start + tgt_size, :] = (
                        suffix_out[sample_idx, :tgt_size, :]
                    )
            sample_idx += 1

    return restored_logits


def _filter_forward_args(forward_args: dict, forward_fn) -> dict:
    """Keep only kwargs that the forward function accepts."""
    import inspect
    sig = inspect.signature(forward_fn)
    valid = set(sig.parameters.keys())
    # Always allow **kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return forward_args
    return {k: v for k, v in forward_args.items() if k in valid}


# ---------------------------------------------------------------------------
# Module 4: FSDP2 strategy helper — unified entry for forward_step & train_step
# ---------------------------------------------------------------------------


def make_prefix_grouper_forward(group_size: int, pad_token_id: int):
    """Create a ``_prefix_grouper_forward`` closure for FSDP2 strategies.

    Usage in ``fsdp2_strategy.py``::

        self._prefix_grouper_forward = make_prefix_grouper_forward(
            group_size=G, pad_token_id=tokenizer.pad_token_id,
        )
    """

    def _fn(model, input_ids, attention_mask, position_ids, forward_args):
        # We need prompt_mask and response_mask from the batch.
        # In ROLL, these are stored in the DataProto batch.
        # The FSDP2 hook should pass the full DataProto, not just decomposed tensors.
        raise NotImplementedError(
            "make_prefix_grouper_forward creates a closure that expects "
            "the full DataProto. Use the DataProto variant instead."
        )

    return _fn


def prefix_grouper_forward_from_data(
    model: nn.Module,
    data_batch: dict,
    forward_args: dict,
    group_size: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Unified PrefixGrouper forward from ROLL DataProto batch dict.

    Args:
        model: HF model.
        data_batch: Dict with keys ``input_ids``, ``attention_mask``, ``prompt_mask``,
                    ``response_mask``, ``prefix_group_id``.
        forward_args: Additional model kwargs.
        group_size: Expected completions per group.
        pad_token_id: Token ID used for padding.

    Returns:
        Restored logits [N, S, V].
    """
    pg_batch = build_pg_from_micro_batch(
        input_ids=data_batch["input_ids"],
        attention_mask=data_batch["attention_mask"],
        prompt_mask=data_batch["prompt_mask"],
        response_mask=data_batch["response_mask"],
        prefix_group_id=data_batch["prefix_group_id"],
        group_size=group_size,
        pad_token_id=pad_token_id,
    )
    return forward_with_prefix_grouper(
        model=model,
        pg_batch=pg_batch,
        original_input_ids=data_batch["input_ids"],
        forward_args=forward_args,
    )
