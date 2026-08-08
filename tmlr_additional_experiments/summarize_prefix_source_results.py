import argparse
import csv
import glob
import os

import numpy as np
import torch
from tqdm import tqdm


def safe_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def infer_mode(path):
    parts = os.path.normpath(path).split(os.sep)
    for part in parts:
        if part.startswith("agentic_feedback_"):
            return part
        if part.startswith("random_strings_control_"):
            return part
        if part.startswith("random_strings_prefix_"):
            return part
    base = os.path.basename(path)
    if "control_unstructured_random" in base:
        return "random_strings_control_unstructured_random"
    if "control_self_conditioned_random" in base:
        return "random_strings_control_self_conditioned_random"
    if "control_shuffled_self_conditioned_random" in base:
        return "random_strings_control_shuffled_self_conditioned_random"
    if "control_iid_vocab_random" in base:
        return "random_strings_control_iid_vocab_random"
    if "control_structured_feedback_control" in base:
        return "random_strings_control_structured_feedback_control"
    if "control_structured_feedback_adversarial" in base:
        return "random_strings_control_structured_feedback_adversarial"
    if "control_structured_feedback_random_noise" in base:
        return "random_strings_control_structured_feedback_random_noise"
    if "agentic_feedback_control" in base:
        return "agentic_feedback_control"
    if "agentic_feedback_adversarial" in base:
        return "agentic_feedback_adversarial"
    if "agentic_feedback_random_noise" in base:
        return "agentic_feedback_random_noise"
    if "prefix_model_generated" in base:
        return "random_strings_prefix_model_generated"
    if "prefix_shuffled_model_generated" in base:
        return "random_strings_prefix_shuffled_model_generated"
    if "prefix_iid_vocab" in base:
        return "random_strings_prefix_iid_vocab"
    if "prefix_original_prompts" in base:
        return "random_strings_prefix_original_prompts"
    return "unknown"


def infer_model(path):
    # Layout produced by the BF runs is .../<mode>/<model>/<file>_bf.pt, so the model
    # is the directory immediately containing the file.
    return os.path.basename(os.path.dirname(os.path.normpath(path)))


def window_bf(entropy_groups, start, end):
    values = []
    for group in entropy_groups:
        for entropy in group:
            if len(entropy) <= start:
                continue
            vals = entropy[start:min(end, len(entropy))]
            if vals:
                values.append(float(np.mean(vals)))
    if not values:
        return ""
    return float(np.exp(np.mean(values)))


def summarize_bf_file(path, early_window, late_window):
    data = safe_load(path)
    if not (isinstance(data, (list, tuple)) and len(data) == 3):
        raise ValueError(f"Not a new-format *_bf.pt file: {path}")
    bf_values, overall_bf, profile = data
    entropy_groups = profile.get("entropy", [])
    return {
        "mode": infer_mode(path),
        "model": infer_model(path),
        "file": path,
        "overall_bf": float(overall_bf),
        "num_prompt_bf": len(bf_values),
        "early_bf": window_bf(entropy_groups, 0, early_window),
        "late_bf": window_bf(entropy_groups, late_window[0], late_window[1]),
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = ["mode", "model", "overall_bf", "early_bf", "late_bf", "num_prompt_bf", "file"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize generation-randomness BF outputs.")
    parser.add_argument("--results_root", required=True, help="Root containing generation-randomness *_bf.pt files.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--early_window", type=int, default=20)
    parser.add_argument("--late_window_start", type=int, default=80)
    parser.add_argument("--late_window_end", type=int, default=120)
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.results_root, "**", "*_bf.pt"), recursive=True))
    rows = []
    for path in tqdm(
            files,
            desc="Summarizing BF files",
            unit="file",
            dynamic_ncols=True,
            disable=args.no_progress):
        try:
            rows.append(summarize_bf_file(
                path,
                early_window=args.early_window,
                late_window=(args.late_window_start, args.late_window_end),
            ))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
