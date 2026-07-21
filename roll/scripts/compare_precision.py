"""Phase 3: Precision comparison of PG OFF vs PG ON with fixed restore."""
import json
import re
import os
import sys
import argparse

METRICS = [
    "actor/pg_loss@sum", "actor/kl_loss@sum", "actor/total_loss@sum",
    "actor/approxkl@sum", "actor/policykl@sum", "actor_train/grad_norm",
    "critic/entropy/mean", "critic/score/mean", "critic/advantages/mean",
    "time/actor_train/train_step/total", "system/actor_train/tps",
    "system/max_memory_allocated@max", "system/max_memory_reserved@max",
]

def extract_step_metrics(log_path: str, step: int = 1) -> dict:
    """Extract metrics dictionary for a specific step from the pipeline log."""
    with open(log_path) as f:
        content = f.read()

    # Find the metrics JSON for the matching step
    # Pattern: system/step": <step>
    pattern = r'"system/step":\s*' + str(step)
    matches = [m.start() for m in re.finditer(pattern, content)]
    if not matches:
        print(f"  WARNING: step {step} not found in {log_path}")
        return {}

    # Take the last occurrence (step 1 produces 2 metrics lines - step 0 and step 1)
    start = matches[-1]
    # Find the enclosing { } for the JSON dict
    brace_start = content.rfind("{", 0, start)
    if brace_start == -1:
        brace_start = start - 200
    # Find matching closing brace
    depth = 0
    end = brace_start
    for i in range(brace_start, len(content)):
        if content[i] == '{': depth += 1
        elif content[i] == '}': depth -= 1
        if depth == 0 and i > brace_start:
            end = i + 1
            break

    json_str = content[brace_start:end]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error at {brace_start}:{end}: {e}")
        # Try to fix truncated JSON
        json_str = json_str.rsplit("}", 1)[0] + "}"
        try:
            data = json.loads(json_str)
        except:
            return {}

    result = {}
    for key in METRICS:
        if key in data:
            result[key] = data[key]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pg_off_log", required=True)
    parser.add_argument("--pg_on_log", required=True)
    args = parser.parse_args()

    off_metrics = extract_step_metrics(args.pg_off_log, step=1)
    on_metrics = extract_step_metrics(args.pg_on_log, step=1)

    print(f"\n{'='*70}")
    print(f"  Precision Comparison: PG OFF vs PG ON (step 1)")
    print(f"{'='*70}")

    if not off_metrics:
        print("  ERROR: Could not extract PG OFF metrics")
        sys.exit(1)
    if not on_metrics:
        print("  ERROR: Could not extract PG ON metrics")
        sys.exit(1)

    all_pass = True
    print(f"\n  {'Metric':<40} {'OFF':<18} {'ON':<18} {'Diff':<18} {'Status'}")
    print(f"  {'-'*40} {'-'*18} {'-'*18} {'-'*18} {'-'*8}")

    for key in METRICS:
        off_val = off_metrics.get(key)
        on_val = on_metrics.get(key)
        if off_val is None and on_val is None:
            continue
        off_str = f"{off_val:.10f}" if off_val is not None else "N/A"
        on_str = f"{on_val:.10f}" if on_val is not None else "N/A"
        diff_str = ""
        status = ""
        if off_val is not None and on_val is not None:
            diff = abs(on_val - off_val)
            if "tps" in key or "memory" in key or "time" in key:
                # Performance metrics - higher/lower is expected
                rel = diff / max(abs(off_val), 1e-10) * 100
                diff_str = f"{diff:.4f} ({rel:.1f}%)"
                pct = abs(off_val - on_val) / max(abs(off_val), 1e-10) * 100
                if pct < 15:
                    status = "✅"
                else:
                    status = "⚠️"
                    all_pass = False
            elif "loss" in key or "grad_norm" in key or "approxkl" in key or "policykl" in key:
                # Precision metrics - must be close
                rel = diff / max(abs(off_val), 1e-10) * 100
                diff_str = f"{diff:.10f} ({rel:.2f}%)"
                if rel < 5.0 and diff < 0.1:
                    status = "✅"
                elif key == "actor/kl_loss@sum" and on_val > 10 * off_val:
                    status = "❌ KL DIVERGENCE"
                    all_pass = False
                elif key == "actor/pg_loss@sum" and off_val > 0.0001 and on_val == 0:
                    status = "❌ ZERO PG LOSS"
                    all_pass = False
                else:
                    status = "⚠️"
                    all_pass = False
            else:
                diff_str = f"{diff:.4f}"
                status = ""
            diff_str = f"{diff_str:<18}"
        else:
            diff_str = f"{'N/A':<18}"

        print(f"  {key:<40} {off_str:<18} {on_str:<18} {diff_str} {status}")

    print(f"\n{'='*70}")
    if all_pass:
        print(f"  ✅ PRECISION ALIGNMENT PASSED")
    else:
        print(f"  ❌ PRECISION ALIGNMENT FAILED - see flagged metrics above")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
