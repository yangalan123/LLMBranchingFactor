import argparse
import csv
import json
import os
import subprocess
import sys


def load_rows(path):
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


def load_json_mapping(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_row_index(args):
    if args.row_index is not None:
        return args.row_index
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        raise ValueError("Provide --row_index or run under a SLURM array.")
    return int(task_id)


def row_float(row, key, default=0):
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def row_model_keys(row):
    model_path = row.get("model_path", "")
    model_basename = row.get("model_basename", "")
    model_path_basename = os.path.basename(model_path.rstrip("/"))
    return {
        model_path,
        model_basename,
        model_path_basename,
        model_path.replace("/", "_"),
        model_basename.replace("/", "_"),
        model_path_basename.replace("/", "_"),
    }


def model_matches(row, model_filter):
    if model_filter is None or model_filter == "":
        return True
    return model_filter in row_model_keys(row)


def row_prompt_length_key(index_and_row):
    index, row = index_and_row
    return (
        row_float(row, "model_token_prefix_tokens"),
        row_float(row, "constraint_level"),
        index,
    )


def select_rows(args, rows):
    if args.model_filter:
        selected = [
            (idx, row)
            for idx, row in enumerate(rows)
            if model_matches(row, args.model_filter)
        ]
        if not selected:
            raise ValueError(f"No manifest rows match model filter: {args.model_filter}")
        if args.selection == "model_max_constraint":
            return [max(selected, key=row_prompt_length_key)]
        return selected

    row_index = get_row_index(args)
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"Manifest row index {row_index} out of range for {len(rows)} rows.")
    return [(row_index, rows[row_index])]


def optional_arg(cmd, flag, value):
    if value is not None and str(value) != "":
        cmd.extend([flag, str(value)])


def build_command(args, row):
    chat_template_map = load_json_mapping(args.chat_template_map_json)
    model_basename = row.get("model_basename", "")
    chat_template_path = chat_template_map.get(model_basename, args.default_chat_template_path)

    cmd = [
        sys.executable,
        args.demo_script,
        "--task_type", "language_modeling",
        "--model", row["model_path"],
        "--dataset_path", row["dataset_path"],
        "--dataset_name", "",
        "--dataset_sample_counts", str(args.sample_counts),
        "--sample_counts", str(args.sample_counts),
        "--max_tokens", str(args.max_tokens),
        "--log_probs", str(args.log_probs),
        "--top_p", str(args.top_p),
        "--min_p", str(args.min_p),
        "--seed", str(args.seed),
        "--min_word_count", "0",
        "--max_constraint_level", "0",
        "--constraint_level", "0",
        "--output_root_dir", row["output_root_dir"],
    ]
    optional_arg(cmd, "--chat_template_path", chat_template_path)
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(description="Run demo/demo.py for one row of a dataset manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--row_index", type=int, default=None)
    parser.add_argument("--model_filter", default=None, help="Run manifest rows matching this model path or basename.")
    parser.add_argument(
        "--selection",
        choices=["model_all", "model_max_constraint"],
        default="model_all",
        help="With --model_filter, run all matching rows or only the row with the longest model-token prefix.",
    )
    parser.add_argument("--demo_script", default=os.path.join("demo", "demo.py"))
    parser.add_argument("--sample_counts", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--log_probs", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat_template_map_json", default=None)
    parser.add_argument("--default_chat_template_path", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.manifest)
    selected_rows = select_rows(args, rows)

    print(json.dumps({
        "manifest": args.manifest,
        "model_filter": args.model_filter,
        "selection": args.selection if args.model_filter else "row",
        "num_selected_rows": len(selected_rows),
        "selected_row_indices": [idx for idx, _ in selected_rows],
    }, indent=2))

    for row_index, row in selected_rows:
        cmd = build_command(args, row)
        print(json.dumps({
            "row_index": row_index,
            "row": row,
            "cmd": cmd,
        }, indent=2))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
