"""CPU unit tests for the ROLL PrefixGrouper adapter.

Tests the group-building and restore logic without requiring a GPU.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from roll.utils.prefix_grouper import (
    _find_continuous_runs,
    _separate_prompt_response,
    build_pg_from_micro_batch,
    PGBatch,
)


def test_find_continuous_runs():
    ids = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3])
    runs = _find_continuous_runs(ids)
    assert runs == [(0, 2, 0), (2, 5, 1), (5, 6, 2), (6, 8, 3)], f"Got {runs}"
    print("  PASS: test_find_continuous_runs")


def test_separate_prompt_response():
    # [P][R][pad] right-padded layout
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0],
                              [1, 2, 9, 8, 7, 6, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 0]])
    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0],
                                  [0, 0, 0, 1, 1, 1, 0]])
    prompt_ids, response_ids, p_mask, r_mask = _separate_prompt_response(
        input_ids, attention_mask, response_mask, pad_token_id=0
    )
    # Row 0: prompt length 2, response length 3
    assert prompt_ids[0, :2].tolist() == [1, 2], f"Got {prompt_ids[0].tolist()}"
    assert response_ids[0, :3].tolist() == [3, 4, 5], f"Got {response_ids[0].tolist()}"
    # Row 1: prompt length 3, response length 3
    assert prompt_ids[1, :3].tolist() == [1, 2, 9], f"Got {prompt_ids[1].tolist()}"
    assert response_ids[1, :3].tolist() == [8, 7, 6], f"Got {response_ids[1].tolist()}"
    print("  PASS: test_separate_prompt_response")


def test_build_pg_group_size():
    """Test basic group building with G=2."""
    N, S = 4, 16
    input_ids = torch.zeros(N, S, dtype=torch.long)
    attention_mask = torch.zeros(N, S, dtype=torch.long)
    prompt_mask = torch.zeros(N, S, dtype=torch.long)
    response_mask = torch.zeros(N, S, dtype=torch.long)
    prefix_group_id = torch.zeros(N, dtype=torch.long)

    # Group 0: rows 0-1, prompt len=5, response len=4 and 3
    # Group 1: rows 2-3, prompt len=6, response len=3 and 4
    prompt_lens = [5, 5, 6, 6]
    response_lens = [4, 3, 3, 4]
    for i in range(N):
        p, r = prompt_lens[i], response_lens[i]
        input_ids[i, :p] = torch.randint(10, 50, (p,))
        input_ids[i, p:p+r] = torch.randint(50, 100, (r,))
        attention_mask[i, :p+r] = 1
        prompt_mask[i, :p] = 1
        response_mask[i, p:p+r] = 1
        prefix_group_id[i] = i // 2  # Groups of 2

    # Override: ensure group 0 has same prompt, group 1 has same prompt but different
    pt0 = input_ids[0, :5].clone()
    pt1 = input_ids[2, :6].clone()
    for i in range(2):
        input_ids[i, :5] = pt0
    for i in range(2, 4):
        input_ids[i, :6] = pt1
        prefix_group_id[i] = i // 2  # Groups of 2

    pg = build_pg_from_micro_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_mask=prompt_mask,
        response_mask=response_mask,
        prefix_group_id=prefix_group_id,
        group_size=2,
        pad_token_id=0,
    )
    assert pg.num_groups == 2, f"Expected 2 groups, got {pg.num_groups}"
    assert pg.group_size == 2
    print(f"  Grouped input shape: {pg.input_ids.shape}")
    print("  PASS: test_build_pg_group_size")


def test_build_pg_varying_length():
    """Test with varying prompt and response lengths across groups."""
    N = 6  # 3 groups × 2
    S = 32
    input_ids = torch.zeros(N, S, dtype=torch.long)
    attention_mask = torch.zeros(N, S, dtype=torch.long)
    prompt_mask = torch.zeros(N, S, dtype=torch.long)
    response_mask = torch.zeros(N, S, dtype=torch.long)
    prefix_group_id = torch.tensor([0, 0, 1, 1, 2, 2])

    pg_lens = [(4, 3), (4, 5), (8, 2), (8, 4), (6, 6), (6, 2)]
    # Group 0: rows 0-1, prompt len=4 (same prompt)
    pt0 = torch.randint(10, 50, (4,))
    # Group 1: rows 2-3, prompt len=8 (same prompt)
    pt1 = torch.randint(10, 50, (8,))
    # Group 2: rows 4-5, prompt len=6 (same prompt)
    pt2 = torch.randint(10, 50, (6,))
    prompt_templates = [pt0, pt0, pt1, pt1, pt2, pt2]
    for i, (p, r) in enumerate(pg_lens):
        input_ids[i, :p] = prompt_templates[i]
        input_ids[i, p:p+r] = torch.randint(50, 100, (r,))
        attention_mask[i, :p+r] = 1
        prompt_mask[i, :p] = 1
        response_mask[i, p:p+r] = 1

    pg = build_pg_from_micro_batch(input_ids, attention_mask, prompt_mask, response_mask, prefix_group_id, 2, 0)
    assert pg.num_groups == 3
    print(f"  Groups: {pg.num_groups}, Grouped input shape: {pg.input_ids.shape}")
    print("  PASS: test_build_pg_varying_length")


def test_reject_wrong_group_size():
    """Test that wrong group sizes are rejected."""
    N, S = 3, 16  # 3 rows, can't form group_size=2 groups
    prefix_group_id = torch.tensor([0, 0, 1])
    try:
        pg = build_pg_from_micro_batch(
            torch.zeros(N, S), torch.ones(N, S),
            torch.ones(N, S), torch.zeros(N, S),
            prefix_group_id, 2, 0
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly rejected: {e}")
        print("  PASS: test_reject_wrong_group_size")


def test_reject_mismatched_prompt():
    """Test that mismatched prompts within a group are rejected."""
    N, S = 4, 16
    input_ids = torch.zeros(N, S, dtype=torch.long)
    attention_mask = torch.ones(N, S, dtype=torch.long)
    prompt_mask = torch.ones(N, S, dtype=torch.long)
    response_mask = torch.zeros(N, S, dtype=torch.long)
    prefix_group_id = torch.tensor([0, 0, 1, 1])

    # Give row 0 and 1 different prompts
    input_ids[0, :8] = 1
    input_ids[1, :8] = 2

    try:
        pg = build_pg_from_micro_batch(input_ids, attention_mask, prompt_mask, response_mask, prefix_group_id, 2, 0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly rejected: {e}")
        print("  PASS: test_reject_mismatched_prompt")


def run_all():
    print("\n=== PrefixGrouper Adapter CPU Unit Tests ===\n")
    test_find_continuous_runs()
    test_separate_prompt_response()
    test_build_pg_group_size()
    test_build_pg_varying_length()
    test_reject_wrong_group_size()
    test_reject_mismatched_prompt()
    print("\n=== ALL CPU TESTS PASSED ===\n")


if __name__ == "__main__":
    run_all()
