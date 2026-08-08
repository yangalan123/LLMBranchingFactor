"""
Gather BF values from demo pipeline outputs and plot a conference-ready
histogram comparing relative BF across model alignment stages.

Works with any task (math, storytelling, mmlu, ...) -- just point
--results_dir at the right output folder.

Usage:
    # Basic (auto-discover models)
    python plot_bf_histogram.py --results_dir ../demo/response_math

    # Filter by dataset and constraint level
    python plot_bf_histogram.py --results_dir ../demo/response_math --dataset math500 --constraint_level 1

    # Rename + order (the order you list --rename pairs = x-axis order)
    python plot_bf_histogram.py --results_dir ../demo/response_math \
        --rename "Qwen2.5-Math-7B=Math-7B (base),Qwen2.5-7B-SFT=SFT,Qwen2.5-7B-DPO=DPO"

    # Storytelling example
    python plot_bf_histogram.py --results_dir ../demo/response_storywriting \
        --constraint_level 3 --rename "OLMo2-7B-1124=OLMo2-7B Base"
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import glob
import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from uncertainty_quantification.visualization_utils import (
    DEFAULT_FIG_SIZE, DEFAULT_FONT_SIZE, DEFAULT_LINE_WIDTH,
    axis_standardize
)

DEFAULT_BAR_FIG_SIZE = (DEFAULT_FIG_SIZE[0], int(DEFAULT_FIG_SIZE[1] * 0.65))

# ---------------------------------------------------------------------------
# Default color cycle (extends automatically if > 10 models)
# ---------------------------------------------------------------------------

_DEFAULT_COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
    "#795548", "#009688", "#E91E63", "#3F51B5", "#CDDC39",
]


def _get_colors(n: int):
    if n <= len(_DEFAULT_COLORS):
        return _DEFAULT_COLORS[:n]
    cmap = plt.cm.get_cmap("tab20", n)
    return [matplotlib.colors.rgb2hex(cmap(i)) for i in range(n)]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_rename(rename_str: str):
    """
    Parse 'old1=new1,old2=new2' into an OrderedDict {old1: new1, ...}.
    The insertion order doubles as the desired x-axis order.
    """
    mapping = OrderedDict()
    if not rename_str:
        return mapping
    for pair in rename_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        old, new = pair.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


# ---------------------------------------------------------------------------
# Gathering
# ---------------------------------------------------------------------------

def gather_bf_values(
    results_dir: str,
    dataset_filter: str = None,
    constraint_level: str = None,
):
    """
    Walk results_dir for *_bf.pt files and extract per-model BF statistics.

    Args:
        results_dir: root directory to search recursively.
        dataset_filter: only include files whose name contains this string
                        (e.g. "math500", "aime24", "creative_storygen").
        constraint_level: only include files under a directory whose name
                          contains "multi_constraints_{level}" (default: no
                          filter).  Pass "1" to match constraint level 1, etc.
    Returns:
        dict  model_key -> {"bf_per_prompt": np.array,
                            "overall_bf": float, "file": str}
    """
    pattern = os.path.join(results_dir, "**", "*_bf.pt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"No *_bf.pt files found under {results_dir}")
        return {}

    # --- constraint level filter (on directory path) ---
    if constraint_level is not None:
        tag = f"multi_constraints_{constraint_level}"
        before = len(files)
        files = [f for f in files if tag in f.replace("\\", "/")]
        print(f"  Constraint level '{constraint_level}': {before} -> {len(files)} files")

    # --- dataset filter (on filename) ---
    if dataset_filter:
        before = len(files)
        files = [f for f in files if dataset_filter in os.path.basename(f)]
        print(f"  Dataset filter '{dataset_filter}': {before} -> {len(files)} files")

    if not files:
        print("  No files remain after filtering.")
        return {}

    model_results = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        model_basename = fname.split("_response_n_")[0]
        key = model_basename

        try:
            data = torch.load(fpath, weights_only=False)
            if len(data) >= 2:
                bf_per_prompt, overall_bf = data[0], data[1]
                bf_per_prompt = np.array(bf_per_prompt, dtype=np.float64)
                bf_per_prompt = bf_per_prompt[np.isfinite(bf_per_prompt)]
                model_results[key] = {
                    "bf_per_prompt": bf_per_prompt,
                    "overall_bf": float(overall_bf),
                    "file": fpath,
                }
                print(f"  {key}: overall BF = {overall_bf:.4f}  "
                      f"(median = {np.median(bf_per_prompt):.4f}, "
                      f"n = {len(bf_per_prompt)})")
        except Exception as e:
            print(f"  Error loading {fpath}: {e}")
    return model_results


def resolve_ordering_and_labels(
    model_results: dict,
    rename_map: OrderedDict = None,
):
    """
    Decide the left-to-right ordering of models and their display labels.

    If --rename is given, only the matched models are included (in the
    order specified).  If --rename is not given, all discovered models
    are sorted alphabetically.

    Returns:
        ordered_keys: list of keys into model_results
        labels:       list of display strings (same length)
    """
    rename_map = rename_map or OrderedDict()

    ordered_keys = []
    matched = set()
    for pattern in rename_map:
        for k in model_results:
            if pattern in k and k not in matched:
                ordered_keys.append(k)
                matched.add(k)

    if not rename_map:
        ordered_keys = sorted(model_results.keys())

    # Build display labels
    labels = []
    for k in ordered_keys:
        label = k
        for pattern, display in rename_map.items():
            if pattern in k:
                label = display
                break
        labels.append(label)

    return ordered_keys, labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _apply_rcparams(fontsize=DEFAULT_FONT_SIZE, linewidth=DEFAULT_LINE_WIDTH):
    plt.rcParams.update({
        'font.size': fontsize,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': linewidth * 0.25,
        'xtick.major.width': linewidth * 0.2,
        'ytick.major.width': linewidth * 0.2,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.pad': 6,
        'ytick.major.pad': 6,
    })


def plot_bf_bars(
    model_results: dict,
    ordered_keys: list,
    labels: list,
    output_path: str = "bf_comparison.pdf",
    figsize=DEFAULT_BAR_FIG_SIZE,
    fontsize=DEFAULT_FONT_SIZE,
    linewidth=DEFAULT_LINE_WIDTH,
    show_relative=True,
):
    """Conference-ready bar chart of BF values."""
    if not ordered_keys:
        print("No models to plot.")
        return

    bf_means = [model_results[k]["overall_bf"] for k in ordered_keys]
    colors = _get_colors(len(ordered_keys))
    _apply_rcparams(fontsize, linewidth)

    edge_lw = linewidth * 0.25
    err_lw = linewidth * 0.3
    err_cap = linewidth * 1.2

    # --- Plot 1: Absolute BF values ---
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(ordered_keys))

    y_floor = max(0, min(bf_means) * 0.95)
    y_floor = np.floor(y_floor * 10) / 10
    y_floor = max(y_floor, 0.8)

    sem = [np.std(model_results[k]["bf_per_prompt"]) /
           np.sqrt(len(model_results[k]["bf_per_prompt"]))
           for k in ordered_keys]

    bars = ax.bar(x, [v - y_floor for v in bf_means], bottom=y_floor,
                  color=colors, edgecolor='black',
                  linewidth=edge_lw, width=0.6, zorder=3)
    ax.errorbar(x, bf_means, yerr=sem, fmt='none', ecolor='black',
                capsize=err_cap, capthick=err_lw, linewidth=err_lw, zorder=4)

    y_ceil = max(bf_means) + max(sem) + 0.05
    ax.set_ylim(y_floor, y_ceil)

    label_fs = fontsize * 0.85
    for i, (bar, val) in enumerate(zip(bars, bf_means)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + sem[i] + 0.005,
                f'{val:.2f}', ha='center', va='bottom', fontsize=label_fs,
                fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=fontsize * 0.9)
    ax.set_ylabel(r'BF', fontsize=fontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    axis_standardize(ax, simple_adjust=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved absolute BF plot to {output_path}")

    # --- Plot 2: Relative BF drop w.r.t. first model ---
    if show_relative and len(ordered_keys) > 1:
        base_bf = bf_means[0]
        if base_bf > 0:
            rel_drops = [(base_bf - v) / base_bf * 100 for v in bf_means]
        else:
            rel_drops = [0.0] * len(bf_means)

        fig2, ax2 = plt.subplots(figsize=figsize)
        bars2 = ax2.bar(x, rel_drops, color=colors, edgecolor='black',
                        linewidth=edge_lw, width=0.6, zorder=3)

        for bar, val in zip(bars2, rel_drops):
            y_pos = max(val, 0) + 0.5
            ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f'{val:.1f}%', ha='center', va='bottom',
                     fontsize=label_fs, fontweight='bold')

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=25, ha='right', fontsize=fontsize * 0.9)
        ax2.set_ylabel(r'Relative BF Drop (\%)', fontsize=fontsize)
        ax2.axhline(y=0, color='black', linewidth=edge_lw, linestyle='-')
        ax2.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax2.set_axisbelow(True)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        axis_standardize(ax2, simple_adjust=True)
        plt.tight_layout()

        rel_path = output_path.replace('.pdf', '_relative_drop.pdf')
        plt.savefig(rel_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved relative BF drop plot to {rel_path}")


def plot_bf_distribution(
    model_results: dict,
    ordered_keys: list,
    labels: list,
    output_path: str = "bf_distribution.pdf",
    figsize=DEFAULT_BAR_FIG_SIZE,
    fontsize=DEFAULT_FONT_SIZE,
    linewidth=DEFAULT_LINE_WIDTH,
):
    """Violin plot of per-prompt BF distributions across models."""
    if not ordered_keys:
        return

    _apply_rcparams(fontsize, linewidth)
    colors = _get_colors(len(ordered_keys))

    data = [model_results[k]["bf_per_prompt"] for k in ordered_keys]

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data, positions=range(len(data)),
                          showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    for partname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(linewidth * 0.3)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=fontsize * 0.9)
    ax.set_ylabel(r'Per-Prompt BF', fontsize=fontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    axis_standardize(ax, simple_adjust=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved BF distribution plot to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gather BF values from demo outputs and plot comparison figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # I/O
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Root directory to search for *_bf.pt files")
    parser.add_argument("--output", type=str, default="bf_comparison.pdf",
                        help="Output PDF path for the bar chart")

    # Filters
    parser.add_argument("--dataset", type=str, default=None,
                        help="Keep only files whose name contains this string "
                             "(e.g. 'math500', 'aime24', 'creative_storygen')")
    parser.add_argument("--constraint_level", type=str, default="1",
                        help="Keep only files under a directory matching "
                             "'multi_constraints_{level}' (default: 1, "
                             "pass 'all' to disable)")

    # Model ordering & renaming (order is inferred from --rename order)
    parser.add_argument("--rename", type=str, default=None,
                        help="Comma-separated old=new pairs to rename models "
                             "in the figure. The order they appear also "
                             "controls the left-to-right x-axis order. E.g. "
                             "'Qwen2.5-Math-7B=Math-7B (base),Qwen2.5-7B-SFT=SFT'")

    # Figure style (defaults from visualization_utils)
    parser.add_argument("--figsize_w", type=float, default=DEFAULT_BAR_FIG_SIZE[0])
    parser.add_argument("--figsize_h", type=float, default=DEFAULT_BAR_FIG_SIZE[1])
    parser.add_argument("--fontsize", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument("--linewidth", type=float, default=DEFAULT_LINE_WIDTH)
    parser.add_argument("--no_relative", action="store_true",
                        help="Skip relative BF drop plot")
    parser.add_argument("--no_distribution", action="store_true",
                        help="Skip per-prompt BF distribution plot")
    args = parser.parse_args()

    # --- gather ---
    constraint = args.constraint_level if args.constraint_level != "all" else None
    print(f"Searching for BF results in: {args.results_dir}")
    model_results = gather_bf_values(
        args.results_dir,
        dataset_filter=args.dataset,
        constraint_level=constraint,
    )
    if not model_results:
        print("No results found. Check --results_dir / filter flags.")
        return

    # --- resolve ordering & labels ---
    rename_map = _parse_rename(args.rename)
    ordered_keys, labels = resolve_ordering_and_labels(
        model_results, rename_map=rename_map)

    print(f"\nPlotting {len(ordered_keys)} models: {labels}")

    # --- plot ---
    plot_bf_bars(
        model_results, ordered_keys, labels,
        output_path=args.output,
        figsize=(args.figsize_w, args.figsize_h),
        fontsize=args.fontsize,
        linewidth=args.linewidth,
        show_relative=not args.no_relative,
    )

    if not args.no_distribution:
        dist_path = args.output.replace('.pdf', '_distribution.pdf')
        plot_bf_distribution(
            model_results, ordered_keys, labels,
            output_path=dist_path,
            figsize=(args.figsize_w + 2, args.figsize_h),
            fontsize=args.fontsize,
            linewidth=args.linewidth,
        )

    # --- summary table ---
    if ordered_keys:
        base_bf = model_results[ordered_keys[0]]["overall_bf"]
        print("\n" + "=" * 70)
        print(f"{'Model':<30} {'BF':>8} {'Median':>8} {'Rel Drop':>10}")
        print("-" * 70)
        for k, lbl in zip(ordered_keys, labels):
            r = model_results[k]
            drop = (base_bf - r['overall_bf']) / base_bf * 100 if base_bf > 0 else 0
            print(f"{lbl:<30} {r['overall_bf']:>8.4f} "
                  f"{np.median(r['bf_per_prompt']):>8.4f} {drop:>9.1f}%")
        print("=" * 70)


if __name__ == "__main__":
    main()
