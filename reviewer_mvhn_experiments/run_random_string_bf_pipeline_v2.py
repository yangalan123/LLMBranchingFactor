import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from uncertainty_quantification.loglik_computation import (
    compute_loglik,
    get_tokenwise_entropy_from_vllm_outputs,
    get_tokenwise_logprob_from_vllm_outputs,
)
from uncertainty_quantification.uncertainty_computation import compute_bf_curve_from_profile

from random_string_artifact_diagnostics import (
    expand_patterns,
    extract_path_metadata,
    infer_model_path_from_basename,
    load_patterns_file,
    safe_load,
    sanitize_name,
)


def parse_csv_ints(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def load_json_mapping(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_model_path(model_basename, model_map):
    if model_basename in model_map:
        return model_map[model_basename]
    inferred = infer_model_path_from_basename(model_basename)
    if inferred is not None:
        return inferred
    if model_basename.startswith("OLMo"):
        return f"allenai/{model_basename}"
    if model_basename.startswith("Meta-Llama") or model_basename.startswith("Llama"):
        return f"meta-llama/{model_basename}"
    if model_basename.startswith("Qwen"):
        return f"Qwen/{model_basename}"
    return model_basename


def model_filter_matches(model_basename, model_path, model_filter):
    if not model_filter:
        return True
    candidates = {
        model_basename,
        model_path,
        os.path.basename(str(model_path).rstrip("/")),
        str(model_basename).replace("/", "_"),
        str(model_path).replace("/", "_"),
    }
    return model_filter in candidates


def discover_artifact_paths(patterns_file, patterns):
    all_patterns = []
    if patterns_file:
        all_patterns.extend(load_patterns_file(patterns_file))
    all_patterns.extend(patterns or [])
    paths = expand_patterns(all_patterns)
    seen = set()
    unique = []
    for path in paths:
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            unique.append(path)
            seen.add(abs_path)
    return unique


def artifact_kind(path):
    if os.path.basename(path).endswith("_bf.pt"):
        return "bf"
    return "raw_vllm"


def build_artifact_index(args):
    model_map = load_json_mapping(args.model_map_json)
    groups = defaultdict(lambda: defaultdict(list))
    skipped = []
    for path in discover_artifact_paths(args.artifact_patterns_file, args.artifact_pattern):
        meta = extract_path_metadata(path)
        model_basename = meta.get("model_basename")
        constraint = meta.get("constraint_level")
        if model_basename is None or constraint is None:
            skipped.append({"path": path, "reason": "missing model or constraint metadata"})
            continue
        model_path = resolve_model_path(model_basename, model_map)
        if not model_filter_matches(model_basename, model_path, args.model_filter):
            continue
        item = {
            "path": path,
            "kind": artifact_kind(path),
            "meta": meta,
            "model_basename": model_basename,
            "model_path": model_path,
            "constraint_level": int(constraint),
        }
        groups[model_basename][int(constraint)].append(item)
    return groups, skipped


def prefer_artifact(items):
    # BF files already contain distribution_profile; raw dumps are a fallback.
    return sorted(items, key=lambda item: 0 if item["kind"] == "bf" else 1)[0]


def prompt_from_request(data):
    prompt = getattr(data, "prompt", None)
    if prompt is not None:
        return prompt
    prompt_token_ids = getattr(data, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        return prompt_token_ids
    return ""


def distribution_profile_from_raw(path, top_p):
    database = safe_load(path)
    prompts = [prompt_from_request(data) for data in database]
    metadata = [prompts]
    args = SimpleNamespace(top_p=top_p)
    profile = {
        "prompt": [],
        "output": [],
        "prompt_per_token_logprob": [],
        "output_per_token_logprob": [],
        "output_per_token_logprob_truncated": [],
        "entropy": [],
        "metadata": metadata,
    }
    for idx, data in enumerate(database):
        prompt = prompts[idx]
        all_outputs = data.outputs
        if getattr(data, "prompt_logprobs", None) is None:
            prompt_loglik = None
            prompt_per_token_loglik = None
        else:
            prompt_loglik = compute_loglik(data.prompt_token_ids, data.prompt_logprobs)
            prompt_per_token_loglik = get_tokenwise_logprob_from_vllm_outputs(
                data.prompt_token_ids,
                data.prompt_logprobs,
            )
        profile["prompt"].append([prompt_loglik, len(getattr(data, "prompt_token_ids", [])), prompt])
        profile["output"].extend([
            [x.cumulative_logprob, len(x.token_ids), x.text, idx]
            for x in all_outputs
        ])
        profile["prompt_per_token_logprob"].append(prompt_per_token_loglik)
        profile["output_per_token_logprob"].append([
            get_tokenwise_logprob_from_vllm_outputs(x.token_ids, x.logprobs)
            for x in all_outputs
        ])
        profile["output_per_token_logprob_truncated"].append([
            get_tokenwise_logprob_from_vllm_outputs(x.token_ids, x.logprobs, top_p=args.top_p)
            for x in all_outputs
        ])
        entropies = get_tokenwise_entropy_from_vllm_outputs(all_outputs, args.top_p, top_p_mode=True)
        profile["entropy"].append([x[0] for x in entropies])
    return profile


def distribution_profile_from_bf(path):
    data = safe_load(path)
    if not (isinstance(data, (list, tuple)) and len(data) == 3):
        raise ValueError(f"BF file has no distribution_profile: {path}")
    return data[2]


def load_distribution_profile(item, top_p):
    if item["kind"] == "bf":
        return distribution_profile_from_bf(item["path"])
    return distribution_profile_from_raw(item["path"], top_p=top_p)


def bf_curve_from_distribution_profile(profile, asymptotic_limit, min_prompts_per_position):
    # Thin wrapper around the shared per-position BF curve helper.
    return compute_bf_curve_from_profile(
        profile,
        asymptotic_limit=asymptotic_limit,
        min_prompts_per_position=min_prompts_per_position,
    )


def save_curve(path, rows, extra):
    out_rows = []
    for row in rows:
        merged = dict(extra)
        merged.update(row)
        out_rows.append(merged)
    write_csv(path, out_rows)
    return out_rows


def load_random_strings(path, max_examples, seed):
    rng = random.Random(seed)
    strings = torch.load(path, map_location="cpu", weights_only=False)
    strings = [str(x).strip() for x in strings if str(x).strip()]
    if max_examples is not None and len(strings) > max_examples:
        strings = rng.sample(strings, max_examples)
    return strings


def build_token_prefix_dataset(random_strings, model_path, prefix_tokens, output_path, trust_remote_code):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    prompts = []
    for text in random_strings:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < prefix_tokens:
            continue
        prompt = tokenizer.decode(token_ids[:prefix_tokens], skip_special_tokens=True).strip()
        if prompt:
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No random strings have at least {prefix_tokens} model tokens for {model_path}.")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(prompts, output_path)
    with open(output_path.replace(".pt", ".jsonl"), "w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts[:100]):
            f.write(json.dumps({"id": idx, "prompt": prompt}, ensure_ascii=False) + "\n")
    return prompts


def find_existing_bf(output_root):
    files = glob.glob(os.path.join(output_root, "**", "*_bf.pt"), recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def run_demo_for_dataset(args, model_path, dataset_path, output_root, chat_template_path):
    existing = find_existing_bf(output_root)
    if existing and not args.force_external:
        return existing
    cmd = [
        sys.executable,
        args.demo_script,
        "--task_type", "language_modeling",
        "--model", model_path,
        "--dataset_path", dataset_path,
        "--dataset_name", "",
        "--dataset_sample_counts", str(args.dataset_sample_counts),
        "--sample_counts", str(args.sample_counts),
        "--max_tokens", str(args.max_tokens),
        "--log_probs", str(args.log_probs),
        "--top_p", str(args.top_p),
        "--min_p", str(args.min_p),
        "--seed", str(args.seed),
        "--min_word_count", "0",
        "--max_constraint_level", "0",
        "--constraint_level", "0",
        "--output_root_dir", output_root,
    ]
    if chat_template_path:
        cmd.extend(["--chat_template_path", chat_template_path])
    print(json.dumps({"cmd": cmd}, indent=2))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    bf_path = find_existing_bf(output_root)
    if bf_path is None and not args.dry_run:
        raise FileNotFoundError(f"demo.py finished without producing *_bf.pt under {output_root}")
    return bf_path


def baseline_prefix_rows(baseline_curve, offset):
    return [
        {
            "x": row["position"],
            "bf": row["bf"],
            "segment": "baseline_prefix",
        }
        for row in baseline_curve
        if row["position"] < offset
    ]


def shifted_external_rows(external_curve, offset):
    return [
        {
            "x": row["position"] + offset,
            "bf": row["bf"],
            "segment": "external",
        }
        for row in external_curve
    ]


def plot_model_curves(model_dir, model_basename, plot_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(9, 5))
    for label in sorted({row["series"] for row in plot_rows}):
        rows = sorted([row for row in plot_rows if row["series"] == label], key=lambda x: x["x"])
        plt.plot([row["x"] for row in rows], [row["bf"] for row in rows], marker="o", label=label)
    plt.xlabel("Output position")
    plt.ylabel("BF")
    plt.title(model_basename)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "bf_by_position.pdf"))
    plt.savefig(os.path.join(model_dir, "bf_by_position.png"), dpi=200)
    plt.close()


def select_baseline_constraint(args, model_basename, constraint_items):
    candidates = [
        constraint
        for constraint in sorted(constraint_items)
        if constraint >= args.min_baseline_constraint_level
    ]
    if not candidates:
        available = ", ".join(str(x) for x in sorted(constraint_items))
        raise ValueError(
            f"No baseline constraint for {model_basename} is >= "
            f"{args.min_baseline_constraint_level}. Available constraints: {available}"
        )
    return candidates[0]


def process_model(args, model_basename, constraint_items, random_strings):
    least_constraint = select_baseline_constraint(args, model_basename, constraint_items)
    baseline_item = prefer_artifact(constraint_items[least_constraint])
    model_path = baseline_item["model_path"]
    multiplier = baseline_item["meta"].get("word_level_constraint_multiplier") or args.default_multiplier
    multiplier = int(multiplier)
    model_dir = os.path.join(args.output_dir, sanitize_name(model_basename, max_len=100))
    os.makedirs(model_dir, exist_ok=True)

    baseline_profile = load_distribution_profile(baseline_item, top_p=args.top_p)
    baseline_curve = bf_curve_from_distribution_profile(
        baseline_profile,
        asymptotic_limit=args.asymptotic_limit,
        min_prompts_per_position=args.min_prompts_per_position,
    )
    all_rows = save_curve(
        os.path.join(model_dir, "self_conditioned_curve.csv"),
        baseline_curve,
        {
            "model": model_basename,
            "series": "self_conditioned_baseline",
            "constraint_level": least_constraint,
            "offset_tokens": 0,
            "artifact_path": baseline_item["path"],
        },
    )
    plot_rows = [
        {"series": "self_conditioned_baseline", "x": row["position"], "bf": row["bf"]}
        for row in baseline_curve
    ]
    curves_pt = {
        "model": model_basename,
        "model_path": model_path,
        "least_constraint_level": least_constraint,
        "multiplier": multiplier,
        "baseline_artifact": baseline_item,
        "baseline_curve": baseline_curve,
        "external": {},
    }

    if args.skip_external:
        plot_model_curves(model_dir, model_basename, plot_rows)
        torch.save(curves_pt, os.path.join(model_dir, "curves.pt"))
        write_csv(os.path.join(model_dir, "curves.csv"), all_rows)
        return all_rows

    chat_template_map = load_json_mapping(args.chat_template_map_json)
    chat_template_path = chat_template_map.get(model_basename, args.default_chat_template_path)
    for delta in parse_csv_ints(args.external_deltas):
        offset = delta * multiplier
        dataset_path = os.path.join(
            model_dir,
            "datasets",
            f"random_strings_external_delta_{delta}_prefix_tokens_{offset}.pt",
        )
        build_token_prefix_dataset(
            random_strings,
            model_path=model_path,
            prefix_tokens=offset,
            output_path=dataset_path,
            trust_remote_code=args.trust_remote_code,
        )
        output_root = os.path.join(
            model_dir,
            "external_randomness",
            f"delta_{delta}_prefix_tokens_{offset}",
        )
        bf_path = run_demo_for_dataset(
            args,
            model_path=model_path,
            dataset_path=dataset_path,
            output_root=output_root,
            chat_template_path=chat_template_path,
        )
        external_curve = []
        if bf_path:
            profile = distribution_profile_from_bf(bf_path)
            external_curve = bf_curve_from_distribution_profile(
                profile,
                asymptotic_limit=args.asymptotic_limit,
                min_prompts_per_position=args.min_prompts_per_position,
            )
        curves_pt["external"][delta] = {
            "offset_tokens": offset,
            "dataset_path": dataset_path,
            "bf_path": bf_path,
            "curve": external_curve,
        }
        curve_rows = save_curve(
            os.path.join(model_dir, f"external_delta_{delta}_curve.csv"),
            external_curve,
            {
                "model": model_basename,
                "series": f"external_random_delta_{delta}",
                "constraint_level": least_constraint + delta,
                "offset_tokens": offset,
                "artifact_path": bf_path or "",
            },
        )
        all_rows.extend(curve_rows)
        for row in baseline_prefix_rows(baseline_curve, offset):
            row["series"] = f"external_random_delta_{delta}"
            plot_rows.append(row)
        for row in shifted_external_rows(external_curve, offset):
            row["series"] = f"external_random_delta_{delta}"
            plot_rows.append(row)

    torch.save(curves_pt, os.path.join(model_dir, "curves.pt"))
    write_csv(os.path.join(model_dir, "curves.csv"), all_rows)
    write_csv(os.path.join(model_dir, "plot_rows.csv"), plot_rows)
    plot_model_curves(model_dir, model_basename, plot_rows)
    return all_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean v2 random-string BF pipeline: artifact baseline, external token-prefix reruns, aligned plots."
    )
    parser.add_argument("--artifact_patterns_file", required=True)
    parser.add_argument("--artifact_pattern", action="append", default=[])
    parser.add_argument("--random_strings_pt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_filter", default=None)
    parser.add_argument("--model_map_json", default=None)
    parser.add_argument("--default_multiplier", type=int, default=15)
    parser.add_argument(
        "--min_baseline_constraint_level",
        type=int,
        default=1,
        help="Ignore lower constraint levels when choosing the self-conditioned baseline.",
    )
    parser.add_argument("--external_deltas", default="2,4")
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--dataset_sample_counts", type=int, default=20)
    parser.add_argument("--sample_counts", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--log_probs", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asymptotic_limit", type=int, default=50)
    parser.add_argument("--min_prompts_per_position", type=int, default=1)
    parser.add_argument("--demo_script", default=os.path.join("demo", "demo.py"))
    parser.add_argument("--chat_template_map_json", default=None)
    parser.add_argument("--default_chat_template_path", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--force_external", action="store_true")
    parser.add_argument("--skip_external", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    groups, skipped = build_artifact_index(args)
    with open(os.path.join(args.output_dir, "artifact_index_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "models": {
                model: {
                    str(constraint): [item["path"] for item in items]
                    for constraint, items in sorted(constraint_items.items())
                }
                for model, constraint_items in sorted(groups.items())
            },
            "skipped": skipped,
        }, f, indent=2)
    if not groups:
        raise ValueError("No usable artifacts found. Check artifact patterns and optional model filter.")

    random_strings = load_random_strings(args.random_strings_pt, args.max_examples, args.seed)
    combined_rows = []
    for model_basename in sorted(groups):
        combined_rows.extend(process_model(args, model_basename, groups[model_basename], random_strings))
    write_csv(os.path.join(args.output_dir, "all_curves.csv"), combined_rows)


if __name__ == "__main__":
    main()
