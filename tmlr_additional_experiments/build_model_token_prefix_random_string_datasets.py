import argparse
import csv
import json
import os
import random
from collections import defaultdict

import torch
from tqdm import tqdm

from random_string_artifact_diagnostics import (
    expand_patterns,
    extract_path_metadata,
    infer_model_path_from_basename,
    load_patterns_file,
    sanitize_name,
)


def load_json_mapping(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_csv_ints(text):
    if text is None or text == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def discover_artifacts(patterns, patterns_file=None):
    all_patterns = list(patterns or [])
    if patterns_file is not None:
        all_patterns.extend(load_patterns_file(patterns_file))
    paths = expand_patterns(all_patterns)
    return [path for path in paths if not path.endswith(".metadata")]


def model_groups_from_artifacts(paths):
    grouped = defaultdict(list)
    for path in paths:
        meta = extract_path_metadata(path)
        model = meta.get("model_basename")
        if model is None:
            continue
        grouped[model].append({"path": path, "meta": meta})
    for model in grouped:
        grouped[model].sort(
            key=lambda item: (
                10**9 if item["meta"].get("constraint_level") is None else item["meta"].get("constraint_level"),
                item["path"],
            )
        )
    return grouped


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
    # Final fallback for local caches whose tokenizer directory matches the basename.
    return model_basename


def select_random_strings(random_strings_pt, max_examples, seed):
    rng = random.Random(seed)
    strings = torch.load(random_strings_pt, map_location="cpu", weights_only=False)
    strings = [str(x).strip() for x in strings if str(x).strip()]
    if max_examples is not None and len(strings) > max_examples:
        strings = rng.sample(strings, max_examples)
    return strings


def prefix_by_token_ids(tokenizer, token_ids, token_count):
    if token_count <= 0:
        return ""
    prefix_ids = token_ids[:token_count]
    return tokenizer.decode(prefix_ids, skip_special_tokens=True).strip()


def encode_source_strings(tokenizer, random_strings, min_token_count, require_full_prefix, show_progress):
    encoded = []
    iterator = tqdm(
        random_strings,
        desc="Encoding and filtering source strings",
        unit="string",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for text in iterator:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if require_full_prefix and len(token_ids) < min_token_count:
            continue
        encoded.append((text, token_ids))
    return encoded


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_for_model(model_basename, model_path, constraints, random_strings, token_multiplier,
                    lowest_constraint_level, output_dir, trust_remote_code,
                    require_full_prefix, show_progress):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_dir = os.path.join(output_dir, sanitize_name(model_basename, max_len=100))
    os.makedirs(model_dir, exist_ok=True)
    manifest_rows = []
    token_prefix_counts = {
        constraint: max(0, int((constraint - lowest_constraint_level) * token_multiplier))
        for constraint in constraints
    }
    max_token_prefix_count = max(token_prefix_counts.values()) if token_prefix_counts else 0
    encoded_sources = encode_source_strings(
        tokenizer,
        random_strings,
        min_token_count=max_token_prefix_count,
        require_full_prefix=require_full_prefix,
        show_progress=show_progress,
    )
    if require_full_prefix and not encoded_sources:
        raise ValueError(
            f"No source strings for {model_basename} have at least "
            f"{max_token_prefix_count} model tokens. Lower the constraints or pass "
            "--allow_short_prefixes."
        )

    iterator = tqdm(
        constraints,
        desc=f"Building token-prefix datasets for {model_basename}",
        unit="constraint",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for constraint in iterator:
        token_prefix_count = token_prefix_counts[constraint]
        prompts = [
            prefix_by_token_ids(tokenizer, token_ids, token_prefix_count)
            for _, token_ids in encoded_sources
        ]
        if token_prefix_count > 0:
            prompts = [prompt for prompt in prompts if prompt]
        constraint_dir = os.path.join(
            model_dir,
            f"application_ctrlgen_multi_constraints_{constraint}_model_token_prefix_tokens_{token_prefix_count}",
        )
        os.makedirs(constraint_dir, exist_ok=True)
        dataset_path = os.path.join(
            constraint_dir,
            f"random_strings_model_token_prefix_tokens_{token_prefix_count}.pt",
        )
        preview_path = os.path.join(
            constraint_dir,
            f"random_strings_model_token_prefix_tokens_{token_prefix_count}.jsonl",
        )
        torch.save(prompts, dataset_path)
        write_jsonl(preview_path, [
            {
                "id": idx,
                "model_basename": model_basename,
                "constraint_level": constraint,
                "model_token_prefix_tokens": token_prefix_count,
                "prompt": prompt,
            }
            for idx, prompt in enumerate(prompts[:20])
        ])
        manifest_rows.append({
            "model_basename": model_basename,
            "model_path": model_path,
            "constraint_level": constraint,
            "lowest_constraint_level": lowest_constraint_level,
            "token_multiplier": token_multiplier,
            "model_token_prefix_tokens": token_prefix_count,
            "max_model_token_prefix_tokens": max_token_prefix_count,
            "require_full_prefix": require_full_prefix,
            "num_source_strings": len(encoded_sources),
            "num_prompts": len(prompts),
            "dataset_path": dataset_path,
            "output_root_dir": os.path.join(
                output_dir,
                "bf_outputs",
                sanitize_name(model_basename, max_len=100),
                f"application_ctrlgen_multi_constraints_{constraint}_model_token_prefix_tokens_{token_prefix_count}",
            ),
        })
    return manifest_rows


def infer_constraints(model_items, explicit_constraints):
    if explicit_constraints is not None:
        return explicit_constraints
    constraints = []
    for item in model_items:
        constraint = item["meta"].get("constraint_level")
        if constraint is not None:
            constraints.append(int(constraint))
    return sorted(set(constraints))


def infer_token_multiplier(model_items, fallback):
    for item in model_items:
        multiplier = item["meta"].get("word_level_constraint_multiplier")
        if multiplier is not None:
            return int(multiplier)
    return int(fallback)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build random-string datasets using exact model-token prefix offsets inferred from artifact paths."
    )
    parser.add_argument("--artifact_pattern", action="append", default=[], help="Artifact glob pattern. Repeatable.")
    parser.add_argument("--artifact_patterns_file", default=None)
    parser.add_argument("--random_strings_pt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_map_json", default=None, help="Optional mapping from filename model basename to HF/local model path.")
    parser.add_argument("--constraints", default=None, help="Comma-separated constraints. Defaults to constraints found in artifacts.")
    parser.add_argument("--lowest_constraint_level", type=int, default=None, help="Defaults to the minimum discovered constraint.")
    parser.add_argument("--token_multiplier", type=int, default=None, help="Defaults to filename multiplier, then 15.")
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--allow_short_prefixes",
        action="store_true",
        help="Keep source strings shorter than the largest requested model-token prefix.",
    )
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    artifact_paths = discover_artifacts(args.artifact_pattern, args.artifact_patterns_file)
    if not artifact_paths:
        raise ValueError("No artifact paths found. Provide --artifact_pattern or --artifact_patterns_file.")

    model_map = load_json_mapping(args.model_map_json)
    groups = model_groups_from_artifacts(artifact_paths)
    random_strings = select_random_strings(args.random_strings_pt, args.max_examples, args.seed)
    explicit_constraints = parse_csv_ints(args.constraints)
    show_progress = not args.no_progress
    all_manifest_rows = []

    for model_basename in sorted(groups.keys()):
        model_items = groups[model_basename]
        constraints = infer_constraints(model_items, explicit_constraints)
        if not constraints:
            continue
        lowest = args.lowest_constraint_level
        if lowest is None:
            lowest = min(constraints)
        multiplier = args.token_multiplier
        if multiplier is None:
            multiplier = infer_token_multiplier(model_items, fallback=15)
        model_path = resolve_model_path(model_basename, model_map)
        rows = build_for_model(
            model_basename=model_basename,
            model_path=model_path,
            constraints=constraints,
            random_strings=random_strings,
            token_multiplier=multiplier,
            lowest_constraint_level=lowest,
            output_dir=args.output_dir,
            trust_remote_code=args.trust_remote_code,
            require_full_prefix=not args.allow_short_prefixes,
            show_progress=show_progress,
        )
        all_manifest_rows.extend(rows)

    manifest_csv = os.path.join(args.output_dir, "manifest.csv")
    manifest_jsonl = os.path.join(args.output_dir, "manifest.jsonl")
    write_csv(manifest_csv, all_manifest_rows)
    write_jsonl(manifest_jsonl, all_manifest_rows)
    print(json.dumps({
        "num_models": len(groups),
        "num_rows": len(all_manifest_rows),
        "manifest_csv": manifest_csv,
        "manifest_jsonl": manifest_jsonl,
    }, indent=2))


if __name__ == "__main__":
    main()
