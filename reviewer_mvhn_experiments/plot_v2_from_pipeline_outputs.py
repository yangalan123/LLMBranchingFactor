import argparse
import glob
import json
import os
import sys

import torch

# run_random_string_bf_pipeline_v2 inserts the repo root on sys.path at import
# time; make sure this sibling module is importable regardless of CWD.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from uncertainty_quantification.uncertainty_computation import compute_bf_curve_from_profile
from stitched_plot_utils import plot_rows, write_rows_csv
from run_random_string_bf_pipeline_v2 import (
    distribution_profile_from_bf,
    load_distribution_profile,
)


def safe_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def build_recomputed_curves_pt(curves_pt, asymptotic_limit, min_prompts_per_position, entropy_only, top_p):
    """Reload the baseline + external distribution_profiles referenced by curves.pt
    and recompute their BF curves with the requested estimator settings.

    Profile locations (see run_random_string_bf_pipeline_v2.process_model):
      * baseline  -> curves_pt["baseline_artifact"] (item dict; bf or raw vLLM)
      * external  -> curves_pt["external"][delta]["bf_path"] (a *_bf.pt)
    No vLLM/model call is involved; the per-token entropy/loglik already live in
    those files.
    """
    baseline_item = curves_pt.get("baseline_artifact")
    if baseline_item is None:
        raise ValueError("curves.pt has no 'baseline_artifact'; cannot reload the baseline distribution_profile.")
    baseline_profile = load_distribution_profile(baseline_item, top_p=top_p)
    baseline_curve = compute_bf_curve_from_profile(
        baseline_profile,
        asymptotic_limit=asymptotic_limit,
        min_prompts_per_position=min_prompts_per_position,
        entropy_only=entropy_only,
    )

    external = {}
    for delta, payload in (curves_pt.get("external") or {}).items():
        bf_path = payload.get("bf_path")
        new_payload = {
            "offset_tokens": payload.get("offset_tokens", 0),
            "bf_path": bf_path,
            "dataset_path": payload.get("dataset_path"),
            "curve": [],
        }
        if bf_path and os.path.exists(bf_path):
            profile = distribution_profile_from_bf(bf_path)
            new_payload["curve"] = compute_bf_curve_from_profile(
                profile,
                asymptotic_limit=asymptotic_limit,
                min_prompts_per_position=min_prompts_per_position,
                entropy_only=entropy_only,
            )
        else:
            print(f"  external delta={delta}: missing bf_path ({bf_path}); skipping recompute.")
        external[delta] = new_payload

    recomputed = dict(curves_pt)
    recomputed["baseline_curve"] = baseline_curve
    recomputed["external"] = external
    return recomputed


def safe_name(text):
    keep = []
    for char in str(text):
        keep.append(char if (char.isalnum() or char in "._=-") else "_")
    return "".join(keep).strip("_") or "unknown"


def stitched_rows_from_curves(curves_pt):
    # Rebuild the same stitched series run_random_string_bf_pipeline_v2.py plots:
    # a self-conditioned baseline, plus one external_random_delta_<d> series per
    # delta whose prefix is copied from the baseline (position < offset) and whose
    # tail is the externally-seeded curve shifted by the model-token offset.
    baseline_curve = curves_pt.get("baseline_curve", []) or []
    rows = [
        {"series": "self_conditioned_baseline", "x": row["position"], "bf": row["bf"]}
        for row in baseline_curve
    ]
    print("#(baseline_curve):", len(baseline_curve))
    assert len(rows) > 0, "No baseline curve found"
    missing_deltas = []
    # baseline_series = rows[0]
    for delta, payload in sorted((curves_pt.get("external") or {}).items()):
        offset = int(payload.get("offset_tokens", 0) or 0)
        external_curve = payload.get("curve", []) or []
        series = f"external_random_delta_{delta}"
        if not external_curve:
            missing_deltas.append({"delta": delta, "bf_path": payload.get("bf_path")})
            continue
        # for row in baseline_curve:
            # if row["position"] < offset:
            #     rows.append({"series": series, "x": baseline_series["position"], "bf": baseline_series["bf"]})
        for row in external_curve:
            # if row["position"] < offset:
            #     continue
            rows.append({"series": series, "x": row["position"] + offset, "bf": row["bf"]})
    return rows, missing_deltas


def discover_curve_files(pipeline_output_dir):
    return sorted(glob.glob(os.path.join(pipeline_output_dir, "*", "curves.pt")))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-plot stitched v2 BF curves directly from run_random_string_bf_pipeline_v2 outputs "
                    "(reads each model's curves.pt). No generation rerun, no manifest."
    )
    parser.add_argument(
        "--pipeline_output_dir",
        required=True,
        help="The v2 pipeline --output_dir, e.g. .../outputs/random_string_bf_pipeline_v2",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write stitched figures/csv. Defaults to <pipeline_output_dir>/stitched_plots.",
    )
    parser.add_argument("--model", default=None, help="Optional model_basename (subdir name) filter.")
    parser.add_argument(
        "--use_stored_curves",
        action="store_true",
        help="Plot the curves stored in curves.pt as-is (skip reloading profiles and recomputing BF).",
    )
    parser.add_argument(
        "--asymptotic_limit",
        type=int,
        default=200,
        help="Prefix length below which BF uses the entropy estimate before switching to the "
             "loglik estimate. Random strings converge slowly, so use a few hundred or more.",
    )
    parser.add_argument(
        "--entropy_only",
        action="store_true",
        help="Always use the entropy-based BF (asymptotic_limit -> infinity); never switch to loglik.",
    )
    parser.add_argument(
        "--min_prompts_per_position",
        type=int,
        default=1,
        help="Stop a curve once fewer than this many prompts reach a position.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Only used if a baseline artifact is a raw vLLM dump whose entropy must be recomputed.",
    )
    return parser.parse_args()


def setting_tag(args):
    if args.use_stored_curves:
        return "stored"
    if args.entropy_only:
        return "entropy_only"
    return f"aL{args.asymptotic_limit}"


def main():
    args = parse_args()
    tag = setting_tag(args)
    default_output_dir = os.path.join(args.pipeline_output_dir, f"stitched_plots__{tag}")
    output_dir = args.output_dir or default_output_dir
    curve_files = discover_curve_files(args.pipeline_output_dir)
    if not curve_files:
        raise ValueError(
            f"No <model>/curves.pt found under {args.pipeline_output_dir}. "
            "Point --pipeline_output_dir at the v2 pipeline --output_dir."
        )

    summary = {
        "pipeline_output_dir": args.pipeline_output_dir,
        "output_dir": output_dir,
        "recompute": not args.use_stored_curves,
        "entropy_only": args.entropy_only,
        "asymptotic_limit": "inf" if args.entropy_only else args.asymptotic_limit,
        "min_prompts_per_position": args.min_prompts_per_position,
        "models": [],
    }
    for curves_path in curve_files:
        model_dir = os.path.dirname(curves_path)
        model = os.path.basename(model_dir)
        if args.model is not None and model != args.model:
            continue
        curves_pt = safe_load(curves_path)
        if not args.use_stored_curves:
            print(f"Recomputing BF for {model} "
                  f"({'entropy_only' if args.entropy_only else f'asymptotic_limit={args.asymptotic_limit}'})")
            curves_pt = build_recomputed_curves_pt(
                curves_pt,
                asymptotic_limit=args.asymptotic_limit,
                min_prompts_per_position=args.min_prompts_per_position,
                entropy_only=args.entropy_only,
                top_p=args.top_p,
            )
        rows, missing_deltas = stitched_rows_from_curves(curves_pt)
        if not rows:
            print(f"No curve points for {model}; skipping.")
            summary["models"].append({"model": model, "status": "no_points", "missing_deltas": missing_deltas})
            continue
        out_model_dir = os.path.join(output_dir, safe_name(model))
        write_rows_csv(os.path.join(out_model_dir, "stitched_bf.csv"), model, rows)
        plot_rows(os.path.join(out_model_dir, "stitched_bf.png"), model, rows)
        summary["models"].append({
            "model": model,
            "status": "ok",
            "num_series": len({r["series"] for r in rows}),
            "missing_deltas": missing_deltas,
        })
        if missing_deltas:
            print(f"WARNING {model}: no external curve for deltas "
                  f"{[d['delta'] for d in missing_deltas]} (bf generation likely incomplete).")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
