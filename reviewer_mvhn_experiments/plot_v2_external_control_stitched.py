import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import torch


def safe_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_manifest(path):
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def numeric(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_name(text):
    keep = []
    for char in str(text):
        keep.append(char if (char.isalnum() or char in "._=-") else "_")
    return "".join(keep).strip("_") or "unknown"


def parse_constraints(text):
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def find_bf_file(output_root_dir, bf_glob):
    if not output_root_dir or not os.path.isdir(output_root_dir):
        return None
    matches = sorted(glob.glob(os.path.join(output_root_dir, "**", bf_glob), recursive=True))
    return matches[0] if matches else None


def entropy_groups_from_bf(path):
    data = safe_load(path)
    if not (isinstance(data, (list, tuple)) and len(data) == 3):
        raise ValueError(f"Not a [bf_values, overall_bf, profile] bf file: {path}")
    _, _, profile = data
    return profile.get("entropy", [])


def window_curve(entropy_groups, window_size, metric):
    sums = defaultdict(float)
    counts = defaultdict(int)
    for group in entropy_groups:
        for sample in group:
            for pos, entropy in enumerate(sample):
                value = numeric(entropy)
                if value is None or not math.isfinite(value):
                    continue
                bucket = pos // window_size
                sums[bucket] += value
                counts[bucket] += 1
    curve = []
    for bucket in sorted(counts.keys()):
        mean_entropy = sums[bucket] / counts[bucket]
        y = math.exp(mean_entropy) if metric == "bf" else mean_entropy
        curve.append((bucket * window_size, y))
    return curve


def stitch_curves(curve_by_constraint, offset_by_constraint, constraints):
    stitched = {}
    previous = None
    previous_constraint = None
    for idx, constraint in enumerate(constraints):
        local = curve_by_constraint.get(constraint)
        if not local:
            continue
        offset = offset_by_constraint.get(constraint, 0)
        if idx == 0 or previous is None:
            curve = [(x, y, "self_conditioned_baseline") for x, y in local]
        else:
            prefix = [
                (x, y, f"copied_from_constraint_{previous_constraint}")
                for x, y in previous if x < offset
            ]
            shifted = [
                (x + offset, y, f"generated_after_constraint_{constraint}")
                for x, y in local
            ]
            curve = prefix + shifted
        curve.sort(key=lambda item: item[0])
        stitched[constraint] = curve
        previous = [(x, y) for x, y, _ in curve]
        previous_constraint = constraint
    return stitched


def label_for_constraint(constraint, first_constraint, offset):
    if constraint == first_constraint:
        return f"C={constraint} self-conditioned baseline"
    return f"C={constraint} external_control (offset={offset})"


def write_stitched_csv(path, model, metric, stitched, offset_by_constraint):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "metric", "constraint_level", "offset", "x", "y", "segment_source"],
        )
        writer.writeheader()
        for constraint, curve in stitched.items():
            offset = offset_by_constraint.get(constraint, 0)
            for x, y, source in curve:
                writer.writerow({
                    "model": model,
                    "metric": metric,
                    "constraint_level": constraint,
                    "offset": offset,
                    "x": x,
                    "y": y,
                    "segment_source": source,
                })


def plot_stitched(output_dir, model, metric, stitched, offset_by_constraint, ylabel):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    first_constraint = min(stitched.keys())
    plt.figure(figsize=(9, 5))
    for constraint in sorted(stitched.keys()):
        curve = stitched[constraint]
        xs = [x for x, _, _ in curve]
        ys = [y for _, y, _ in curve]
        plt.plot(
            xs, ys, marker="o", linewidth=1.8, markersize=3,
            label=label_for_constraint(constraint, first_constraint, offset_by_constraint.get(constraint, 0)),
        )
    plt.xlabel("Aligned generation position (generated position + model-token prefix offset)")
    plt.ylabel(ylabel)
    plt.title(f"{model}: Randomness Control")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"stitched_{metric}.pdf"), dpi=200)
    print(f"Saved stitched plot to {os.path.join(output_dir, f'stitched_{metric}.png')}")
    plt.close()


def rows_by_model(rows):
    grouped = defaultdict(list)
    for row in rows:
        model = row.get("model_basename") or "model_unknown"
        grouped[model].append(row)
    return grouped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot v2 stitched external-control curves directly from the model-token-prefix manifest and bf files."
    )
    parser.add_argument("--manifest", required=True, help="manifest.csv or manifest.jsonl from the v2 builder.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--constraints", default=None,
                        help="Comma-separated constraint levels in stitch order. Defaults to all in manifest, ascending.")
    parser.add_argument("--model", default=None, help="Optional model_basename filter.")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--metric", choices=["bf", "entropy"], default="bf")
    parser.add_argument("--bf_glob", default="*_bf.pt", help="Glob used inside each output_root_dir to find the bf file.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_manifest(args.manifest)
    if args.model is not None:
        rows = [row for row in rows if row.get("model_basename") == args.model]
    if not rows:
        raise ValueError("No manifest rows after filtering.")

    ylabel = "exp(mean entropy) (BF)" if args.metric == "bf" else "Mean entropy"
    explicit_constraints = parse_constraints(args.constraints)
    missing = []

    for model, model_rows in rows_by_model(rows).items():
        curve_by_constraint = {}
        offset_by_constraint = {}
        for row in model_rows:
            constraint = numeric(row.get("constraint_level"))
            if constraint is None:
                print(f"Skipping row {row} because constraint_level is None")
                continue
            constraint = int(constraint)
            offset = numeric(row.get("model_token_prefix_tokens"))
            offset = int(offset) if offset is not None else 0
            bf_file = find_bf_file(row.get("output_root_dir", ""), args.bf_glob)
            if bf_file is None:
                missing.append({"model": model, "constraint_level": constraint,
                                "output_root_dir": row.get("output_root_dir", "")})
                continue
            entropy_groups = entropy_groups_from_bf(bf_file)
            curve = window_curve(entropy_groups, args.window_size, args.metric)
            if curve:
                curve_by_constraint[constraint] = curve
                offset_by_constraint[constraint] = offset

        if not curve_by_constraint:
            print(f"No curves found for model {model}")
            continue
        constraints = explicit_constraints or sorted(curve_by_constraint.keys())
        stitched = stitch_curves(curve_by_constraint, offset_by_constraint, constraints)
        if not stitched:
            print(f"No stitched curves found for model {model}")
            continue
        model_dir = os.path.join(args.output_dir, safe_name(model))
        write_stitched_csv(
            os.path.join(model_dir, f"stitched_{args.metric}.csv"),
            model, args.metric, stitched, offset_by_constraint,
        )
        plot_stitched(model_dir, model, args.metric, stitched, offset_by_constraint, ylabel)

    summary = {
        "manifest": args.manifest,
        "metric": args.metric,
        "window_size": args.window_size,
        "output_dir": args.output_dir,
        "missing_bf_files": missing,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
