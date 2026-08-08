"""Plot step-wise (per-interaction-turn) Branching Factor for the multi-turn agentic
experiment produced by ``run_agentic_multistep_bf.py``.

For each model it draws BF vs. interaction turn, one line per feedback condition
(control / adversarial / random_noise), with an optional +/-1 std band across
scenarios. This is the "how does BF evolve as the agent interacts" view.

Local-friendly: only needs the CSV(s) + matplotlib (no torch / vLLM).

Examples:
    # scan a results root for every model's agentic_multistep_bf.csv
    python tmlr_additional_experiments/plot_agentic_stepwise_bf.py \
        --root tmlr_additional_experiments/outputs/agentic_multistep_bf --error_band

    # or point at explicit CSVs
    python tmlr_additional_experiments/plot_agentic_stepwise_bf.py --csv path/to/agentic_multistep_bf.csv
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Importing this also applies the shared paper rcParams.
from stitched_plot_utils import MACARON_PALETTE

# Stable color/label assignment so a condition keeps its color across models.
CONDITION_ORDER = ["control", "adversarial", "random_noise"]
CONDITION_LABELS = {
    "control": "Control",
    "adversarial": "Adversarial",
    "random_noise": "Random Noise",
}


def discover_csvs(root):
    return sorted(glob.glob(os.path.join(root, "**", "agentic_multistep_bf.csv"), recursive=True))


def load_rows(csv_paths):
    """Return {model: {condition: [(turn, bf, bf_std), ...sorted]}}."""
    by_model = defaultdict(lambda: defaultdict(list))
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                model = (row.get("model") or "unknown").strip()
                condition = (row.get("condition") or "unknown").strip()
                turn = int(float(row["turn"]))
                bf = float(row["bf"])
                bf_std = float(row.get("bf_std") or 0.0)
                by_model[model][condition].append((turn, bf, bf_std))
    for model in by_model:
        for condition in by_model[model]:
            by_model[model][condition].sort(key=lambda t: t[0])
    return by_model


def color_for(condition, seen_order):
    if condition in CONDITION_ORDER:
        idx = CONDITION_ORDER.index(condition)
    else:
        if condition not in seen_order:
            seen_order.append(condition)
        idx = len(CONDITION_ORDER) + seen_order.index(condition)
    return MACARON_PALETTE[idx % len(MACARON_PALETTE)]


def condition_sort_key(condition):
    return (CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else 99, condition)


def plot_model(path, model, series, error_band):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    extra_seen = []
    for condition in sorted(series.keys(), key=condition_sort_key):
        pts = series[condition]
        turns = [p[0] for p in pts]
        bfs = [p[1] for p in pts]
        stds = [p[2] for p in pts]
        color = color_for(condition, extra_seen)
        ax.plot(turns, bfs, marker="o", linewidth=2.5, markersize=10, color=color,
                label=CONDITION_LABELS.get(condition, condition))
        if error_band and any(s > 0 for s in stds):
            lo = [b - s for b, s in zip(bfs, stds)]
            hi = [b + s for b, s in zip(bfs, stds)]
            ax.fill_between(turns, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Interaction turn")
    ax.set_ylabel("BF")
    ax.set_title(f"{model}")
    # Integer turn ticks.
    all_turns = sorted({p[0] for pts in series.values() for p in pts})
    if all_turns:
        ax.set_xticks(all_turns)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def safe_name(text):
    return "".join(c if (c.isalnum() or c in "._=-") else "_" for c in str(text)).strip("_") or "unknown"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None,
                        help="Scan this dir recursively for agentic_multistep_bf.csv files.")
    parser.add_argument("--csv", nargs="+", default=None, help="Explicit CSV path(s).")
    parser.add_argument("--output_dir", default=os.path.join(THIS_DIR, "agentic_plots"))
    parser.add_argument("--error_band", action="store_true",
                        help="Shade +/-1 std (across scenarios) around each line.")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_paths = list(args.csv or [])
    if args.root:
        csv_paths.extend(discover_csvs(args.root))
    csv_paths = sorted(set(csv_paths))
    if not csv_paths:
        raise SystemExit("No CSVs given. Pass --csv <file> or --root <dir>.")

    by_model = load_rows(csv_paths)
    for model, series in sorted(by_model.items()):
        out_path = os.path.join(args.output_dir, f"stepwise_bf_{safe_name(model)}.png")
        plot_model(out_path, model, series, args.error_band)
    print(f"Plotted {len(by_model)} model(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
