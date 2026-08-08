"""Plot the Branching Factor (BF) *shift* induced by each agentic feedback condition
relative to the matched control feedback, per model.

Input is the CSV produced by ``summarize_prefix_source_results.py`` (one row per
``*_bf.pt`` with columns ``mode, model, overall_bf, early_bf, late_bf, ...``). For
every model we take the control condition as the baseline and plot

    delta = BF(condition) - BF(control)

as grouped bars (one group per non-control condition). A positive bar means that
feedback condition makes the model's next-step generation *more* branching / less
certain than benign control feedback. This isolates the feedback effect and removes
per-model baseline differences, so it is usually the most legible summary panel.

This script is local-friendly: it only needs the CSV + matplotlib (no torch / vLLM).

Example:
    python tmlr_additional_experiments/plot_agentic_delta_from_control.py \
        --summary_csv outputs/agentic_feedback_bf/summary.csv \
        --output_dir tmlr_additional_experiments/agentic_plots
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from statistics import mean

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Importing this also applies the shared paper rcParams (serif fonts, etc.) and puts
# the repo root on sys.path so uncertainty_quantification is importable.
from stitched_plot_utils import MACARON_PALETTE
from uncertainty_quantification.visualization_utils import (
    base_aligned_model_name_mapping, model_name_visualization_name_mapping)

CONDITION_LABELS = {
    "adversarial": "Adversarial",
    "random_noise": "Random Noise",
    "control": "Control",
}


def canonical_model_key(raw):
    """Normalize a model label to the visualization-basename space used by
    base_aligned_model_name_mapping (e.g. 'meta-llama_Meta-Llama-3-8B' -> 'Llama-3-8B').

    The BF runs name model dirs as ``${MODEL//\\//_}`` (org/model -> org_model), so we
    turn '_' back into '/' before applying the repo's standard viz mapping + basename.
    """
    import os as _os
    name = str(raw).replace("\\", "/").replace("_", "/")
    return _os.path.basename(model_name_visualization_name_mapping(name))


def order_models_base_first(models):
    """Order models so each family is grouped base-first, aligned-second, using the
    repo's base<->aligned mapping. Models with no known partner keep a stable sorted
    order at the end."""
    _, base_to_aligned = base_aligned_model_name_mapping()
    by_canon = defaultdict(list)
    for model in models:
        by_canon[canonical_model_key(model)].append(model)

    ordered = []
    used = set()
    for base_viz, aligned_viz in base_to_aligned.items():
        # base_viz processed before aligned_viz -> base comes first within the family.
        for canon in (base_viz, aligned_viz):
            for label in sorted(by_canon.get(canon, [])):
                if label not in used:
                    ordered.append(label)
                    used.add(label)
    for label in sorted(models):
        if label not in used:
            ordered.append(label)
            used.add(label)
    return ordered


def condition_from_mode(mode, mode_prefix):
    return mode[len(mode_prefix):] if mode.startswith(mode_prefix) else mode


def condition_from_file(file_path, mode_prefix):
    """Infer the feedback condition from the file path.

    Older summary.csv files have an unhelpful ``mode`` column (every row equals the
    results-root name ``agentic_feedback_bf``, because the old infer_mode matched the
    root dir which also starts with ``agentic_feedback_``). The real condition lives
    in the path as ``.../agentic_feedback_<condition>/<model>/<file>``, so we read it
    from there. The results-root token (suffix ``bf``) is dropped.
    """
    if not file_path:
        return None
    parts = os.path.normpath(file_path).split(os.sep)
    candidates = [p[len(mode_prefix):] for p in parts if p.startswith(mode_prefix)]
    candidates = [c for c in candidates if c and c != "bf"]
    return candidates[-1] if candidates else None


def model_from_file(file_path):
    # Layout is .../<condition>/<model>/<file>_bf.pt, so the model is the parent dir.
    if not file_path:
        return None
    return os.path.basename(os.path.dirname(os.path.normpath(file_path)))


def load_summary(summary_csv, mode_prefix, metric):
    """Return {model: {condition: mean_metric}}.

    Condition and model are taken from the ``file`` path when possible (robust to the
    old buggy ``mode`` column), falling back to the ``mode``/``model`` columns.
    """
    by_model = defaultdict(lambda: defaultdict(list))
    # utf-8-sig tolerates a BOM if the CSV was hand-edited/saved by another tool.
    with open(summary_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            value = (row.get(metric) or "").strip()
            if value == "":
                continue
            file_path = (row.get("file") or "").strip()
            mode = (row.get("mode") or "").strip()
            condition = condition_from_file(file_path, mode_prefix)
            if condition is None and mode.startswith(mode_prefix):
                condition = condition_from_mode(mode, mode_prefix)
            if condition is None:
                continue
            model = (row.get("model") or "").strip() or model_from_file(file_path) or "unknown"
            by_model[model][condition].append(float(value))
    return {
        model: {cond: mean(vals) for cond, vals in conds.items()}
        for model, conds in by_model.items()
    }


def compute_deltas(by_model, control_condition):
    """Return (models, conditions, deltas[model][condition]) for non-control conditions."""
    deltas = {}
    all_conditions = set()
    for model, conds in by_model.items():
        if control_condition not in conds:
            print(f"  skipping {model}: no '{control_condition}' baseline among {sorted(conds)}")
            continue
        base = conds[control_condition]
        model_deltas = {c: v - base for c, v in conds.items() if c != control_condition}
        if not model_deltas:
            continue
        deltas[model] = model_deltas
        all_conditions.update(model_deltas.keys())
    models = order_models_base_first(list(deltas.keys()))
    conditions = sorted(all_conditions)
    return models, conditions, deltas


def load_delta_csv(path):
    """Load a precomputed ``agentic_delta_from_control.csv`` (columns
    ``model, condition, delta_*``) so the figure can be re-rendered locally without
    the server-side summary CSV. Returns (models, conditions, deltas)."""
    deltas = defaultdict(dict)
    conditions = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 3 or not row[2].strip():
                continue
            model, cond, val = row[0].strip(), row[1].strip(), row[2].strip()
            deltas[model][cond] = float(val)
            conditions.add(cond)
    models = order_models_base_first(list(deltas.keys()))
    return models, sorted(conditions), dict(deltas)


def write_delta_csv(path, models, conditions, deltas, metric, control_condition):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "condition", f"delta_{metric}_vs_{control_condition}"])
        for model in models:
            for cond in conditions:
                if cond in deltas.get(model, {}):
                    writer.writerow([model, cond, deltas[model][cond]])


def plot_deltas(path, models, conditions, deltas, metric, control_condition):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n_models = len(models)
    n_cond = max(len(conditions), 1)
    group_width = 0.8
    bar_width = group_width / n_cond

    # Keep the canvas compact: this single panel is placed at 0.6\linewidth, so an
    # overly wide figure gets down-scaled and its (inherited) text shrinks. A tighter
    # width keeps the font size on-page comparable to the other paper figures.
    fig, ax = plt.subplots(figsize=(max(9.0, 1.6 * n_models), 6.0))
    for ci, cond in enumerate(conditions):
        color = MACARON_PALETTE[ci % len(MACARON_PALETTE)]
        xs, ys = [], []
        for mi, model in enumerate(models):
            value = deltas.get(model, {}).get(cond)
            if value is None:
                continue
            xs.append(mi - group_width / 2 + bar_width * (ci + 0.5))
            ys.append(value)
        ax.bar(xs, ys, width=bar_width, color=color, edgecolor="white",
               label=CONDITION_LABELS.get(cond, cond))

    ax.axhline(0.0, color="black", linewidth=2, linestyle="--")
    ax.set_xticks(range(n_models))
    model_nametags = [x.split("_")[-1] for x in models]
    ax.set_xticklabels(model_nametags, rotation=20, ha="right")
    ax.set_ylabel(rf"$\Delta$ BF  (condition - {control_condition})")
    ax.set_title(f"BF Shift from Control in Environment Feedback")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary_csv",
                        help="CSV from summarize_prefix_source_results.py (needs the 'model' column).")
    parser.add_argument("--delta_csv",
                        help="Precomputed agentic_delta_from_control.csv to re-plot directly "
                             "(local re-render path; skips summary parsing).")
    parser.add_argument("--output_dir", default=os.path.join(THIS_DIR, "agentic_plots"))
    parser.add_argument("--metric", default="overall_bf", choices=["overall_bf", "early_bf", "late_bf"])
    parser.add_argument("--mode_prefix", default="agentic_feedback_",
                        help="Only rows whose 'mode' starts with this are used.")
    parser.add_argument("--control_condition", default="control",
                        help="Condition (mode minus prefix) used as the per-model baseline.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.delta_csv:
        models, conditions, deltas = load_delta_csv(args.delta_csv)
        if not models:
            raise SystemExit(f"No rows to plot in {args.delta_csv}.")
    else:
        if not args.summary_csv:
            raise SystemExit("Provide --summary_csv (server results) or --delta_csv (local re-render).")
        by_model = load_summary(args.summary_csv, args.mode_prefix, args.metric)
        if not by_model:
            raise SystemExit(
                f"No rows with mode starting '{args.mode_prefix}' (and a numeric {args.metric}) "
                f"in {args.summary_csv}.")
        models, conditions, deltas = compute_deltas(by_model, args.control_condition)
        if not models:
            raise SystemExit(f"No models had a '{args.control_condition}' baseline to diff against.")
        write_delta_csv(os.path.join(args.output_dir, "agentic_delta_from_control.csv"),
                        models, conditions, deltas, args.metric, args.control_condition)
    plot_deltas(os.path.join(args.output_dir, "agentic_delta_from_control.png"),
                models, conditions, deltas, args.metric, args.control_condition)


if __name__ == "__main__":
    main()
