"""Lightweight, dependency-free helpers for the stitched random-string BF plots.

Only the standard library + matplotlib are needed here (no torch / vLLM / pipeline
imports), so this module can be imported both on the server (by
``plot_v2_from_pipeline_outputs.py``) and locally (by ``replot_stitched_from_csv.py``)
to keep a single source of truth for the CSV schema and the figure styling.

The CSV schema written/read here is: ``model, series, x, bf``.
"""

import csv
import os
import sys

# Make uncertainty_quantification importable no matter which sibling script imports
# this module (the local replot script only puts reviewer_mvhn_experiments/ on the
# path, not the repo root). consts/visualization_utils only need numpy, so this stays
# light enough to run locally without torch/vLLM.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(f"matplotlib unavailable ({exc}); install it with `pip install matplotlib`.")
from uncertainty_quantification.visualization_utils import (
    DEFAULT_FONT_SIZE, DEFAULT_LINE_WIDTH, DEFAULT_FIG_SIZE)
# The reference paper figures use font.size=DEFAULT_FONT_SIZE (50) on a (20,15)
# canvas -> a font/width ratio of ~2.5. These stitched/agentic panels render on a
# much smaller canvas and are then down-scaled by \includegraphics, so the old
# 0.35 multiplier made their text noticeably smaller than the surrounding figures.
# Bump to 0.5 (font.size=25) so the on-page text roughly matches the other panels.
_STITCHED_FONT_SIZE = int(DEFAULT_FONT_SIZE * 0.5)
plt.rcParams.update({
    'font.size': _STITCHED_FONT_SIZE,
    'axes.titlesize': _STITCHED_FONT_SIZE,
    'axes.labelsize': _STITCHED_FONT_SIZE,
    'xtick.labelsize': int(_STITCHED_FONT_SIZE * 0.9),
    'ytick.labelsize': int(_STITCHED_FONT_SIZE * 0.9),
    'legend.fontsize': int(_STITCHED_FONT_SIZE * 0.85),
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    # 'axes.linewidth': DEFAULT_LINE_WIDTH,
})

CSV_FIELDNAMES = ["model", "series", "x", "bf"]


def write_rows_csv(path, model, rows):
    """Write stitched rows (each a dict with ``series``/``x``/``bf``) to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({"model": model, **row})


def read_rows_csv(path):
    """Inverse of write_rows_csv: returns (model, rows) where each row has
    numeric ``x``/``bf`` and a ``series`` label."""
    model = None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for record in csv.DictReader(f):
            model = model or record.get("model")
            rows.append({
                "series": record["series"],
                "x": float(record["x"]),
                "bf": float(record["bf"]),
            })
    return model, rows

series_name_mapping ={
    "self_conditioned_baseline": "Self-Conditioned",
    "external_random_delta_2": "External Random (Offset=30)",
    "external_random_delta_4": "External Random (Offset=60)",
}
offset_mapping = {
    "external_random_delta_2": 30,
    "external_random_delta_4": 60,
}
# Macaron-style palette: soft pastels that are still saturated enough to stay
# distinguishable on white. Ordered so the first few series get maximally
# separated hues (rose / blue / mint / lemon / lavender / ...).
MACARON_PALETTE = [
    "#FF6F91",  # raspberry rose
    "#5B8DEF",  # blueberry
    "#3FC1A6",  # pistachio mint
    "#FFC75F",  # lemon caramel
    "#9B72CF",  # ube lavender
    "#FF9671",  # apricot
    "#F49AC2",  # sakura pink
    "#4FB0C6",  # sky teal
    "#C39BD3",  # taro
    "#88C999",  # matcha
]
def plot_rows(path, model, rows, *, figsize=DEFAULT_FIG_SIZE, dpi=200, log_y=False,
              title=None, also_pdf=True):
    """Render stitched BF series to ``path`` (PNG) and, by default, the sibling PDF.

    This is the single place that defines the figure styling, so tweaking
    visualization details here updates both the server pipeline and the local
    re-plot script.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.figure(figsize=figsize)
    colors = MACARON_PALETTE
    # Keep the line visibly thinner than the paper default, but NEVER let it round to
    # 0: int(DEFAULT_LINE_WIDTH * 0.1) == int(0.5) == 0 dropped the connecting line
    # entirely, making each curve look like detached dots.
    line_w = max(1.5, DEFAULT_LINE_WIDTH * 0.6)
    marker_sz = max(2.5, line_w * 1.4)
    for idx, series in enumerate(sorted({row["series"] for row in rows})):
        color = colors[idx % len(colors)]
        pts = sorted([r for r in rows if r["series"] == series], key=lambda r: r["x"])
        plt.plot([r["x"] for r in pts], [r["bf"] for r in pts],
                 marker="o", markersize=marker_sz, linewidth=line_w, linestyle='-',
                 color=color, label=series_name_mapping.get(series, series))
        if series in offset_mapping:
            plt.axvline(x=offset_mapping[series], color=color, linestyle='--',
                        linewidth=max(1.0, line_w * 0.7))
    if log_y:
        plt.yscale("log")
    plt.xlabel("Aligned Output Position")
    plt.ylabel("BF")
    plt.title(title or f"{model}")
    # plt.legend(fontsize=DEFAULT_FONT_SIZE)
    plt.legend()
    plt.tight_layout()
    # bbox_inches="tight" expands the saved canvas to include the large paper-style
    # title/legend/labels instead of clipping them at the figure edge.
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        plt.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return True
