import argparse
import csv
import glob
import hashlib
import importlib
import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm


def _make_pickle_placeholder_class(name, module_name):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state

    return type(name, (), {
        "__module__": module_name,
        "__init__": __init__,
        "__setstate__": __setstate__,
    })


def install_pickle_placeholder(module_name, attr_name):
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    if not hasattr(module, attr_name):
        setattr(module, attr_name, _make_pickle_placeholder_class(attr_name, module_name))
    return True


def install_vllm_pickle_compat():
    # Older vLLM response dumps may pickle bookkeeping classes that diagnostics
    # never read. Newer vLLM versions can remove or move them, breaking unpickle.
    for name in ["RequestMetrics"]:
        install_pickle_placeholder("vllm.sequence", name)


def maybe_patch_missing_pickle_attr(error):
    match = re.search(r"Can't get attribute '([^']+)' on <module '([^']+)'", str(error))
    if not match:
        return False
    attr_name, module_name = match.groups()
    if not module_name.startswith("vllm."):
        return False
    return install_pickle_placeholder(module_name, attr_name)


def add_repo_to_path(repo_root):
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def safe_load(path):
    install_vllm_pickle_compat()
    last_error = None
    for _ in range(5):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except AttributeError as exc:
            last_error = exc
            if not maybe_patch_missing_pickle_attr(exc):
                raise
    raise last_error



def sanitize_name(text, max_len=160):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text).strip("_")
    if len(text) <= max_len:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{text[:max_len - 11]}_{digest}"


def parse_numeric(text):
    if text is None:
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def extract_path_metadata(path):
    norm_path = path.replace("\\", "/")
    base = os.path.basename(norm_path)
    meta = {
        "artifact_path": path,
        "artifact_basename": base,
        "artifact_stem": base,
        "constraint_level": None,
        "model_basename": None,
        "sample_counts": None,
        "max_tokens": None,
        "log_probs": None,
        "min_p": None,
        "top_p": None,
        "seed": None,
        "word_level_constraint_multiplier": None,
        "model_token_prefix_tokens": None,
        "is_bf_file": base.endswith("_bf.pt"),
        "is_update_full_spectrum": base.endswith(".pt.update_full_spectrum"),
    }
    if base.endswith("_bf.pt"):
        meta["artifact_stem"] = base[:-6]
    elif base.endswith(".pt.update_full_spectrum"):
        meta["artifact_stem"] = base[:-len(".pt.update_full_spectrum")]
    elif base.endswith(".pt"):
        meta["artifact_stem"] = base[:-3]

    constraint_match = re.search(r"application_ctrlgen_multi_constraints_(-?\d+)", norm_path)
    if constraint_match:
        meta["constraint_level"] = int(constraint_match.group(1))
    token_prefix_match = re.search(r"model_token_prefix_tokens_(-?\d+)", norm_path)
    if token_prefix_match:
        meta["model_token_prefix_tokens"] = int(token_prefix_match.group(1))

    model_match = re.match(r"(.+?)_response_n_", base)
    if model_match:
        meta["model_basename"] = model_match.group(1)

    patterns = {
        "sample_counts": r"_response_n_(-?\d+)",
        "max_tokens": r"_max_tokens_(-?\d+)",
        "log_probs": r"_log_probs_(-?\d+)",
        "min_p": r"_min_p_([^_]+)",
        "top_p": r"_top_p_([^_]+)",
        "seed": r"_seed(-?\d+)",
        "word_level_constraint_multiplier": r"_word_level_constraint_multiplier_(-?\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, base)
        if match:
            meta[key] = parse_numeric(match.group(1))
    return meta


def artifact_output_dir(base_output_dir, path):
    meta = extract_path_metadata(path)
    constraint = meta.get("constraint_level")
    constraint_name = f"constraint_{constraint}" if constraint is not None else "constraint_unknown"
    model_name = sanitize_name(meta.get("model_basename") or "model_unknown", max_len=80)
    stem_name = sanitize_name(meta.get("artifact_stem") or os.path.basename(path), max_len=120)
    digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:10]
    return os.path.join(base_output_dir, "per_artifact", constraint_name, model_name, f"{stem_name}_{digest}")


def infer_model_path_from_basename(model_basename):
    if not model_basename:
        return None
    try:
        from uncertainty_quantification.consts import ALL_MODELS
    except Exception:
        return None
    for model in ALL_MODELS:
        if os.path.basename(model) == model_basename or model == model_basename:
            return model
    return None


def get_logprob_value(value):
    if isinstance(value, float):
        return value
    if hasattr(value, "logprob"):
        return value.logprob
    return float(value)


def top1_prob_from_logprob_dict(logprob_dict):
    if logprob_dict is None or len(logprob_dict) == 0:
        return None
    vals = [get_logprob_value(v) for v in logprob_dict.values()]
    return float(np.exp(np.max(vals)))


def selected_prob_from_logprob_dict(logprob_dict, token_id):
    if logprob_dict is None or token_id not in logprob_dict:
        return None
    return float(np.exp(get_logprob_value(logprob_dict[token_id])))


def entropy_from_logprob_dict(logprob_dict, top_p=0.9, selected_token_id=None):
    if logprob_dict is None or len(logprob_dict) == 0:
        return None
    items = [(k, get_logprob_value(v)) for k, v in logprob_dict.items()]
    items.sort(key=lambda x: x[1], reverse=True)

    kept = []
    total = 0.0
    for token_id, logp in items:
        prob = float(np.exp(logp))
        if prob <= 0:
            continue
        kept.append((token_id, logp))
        total += prob
        if total >= top_p:
            break

    if selected_token_id is not None and selected_token_id not in {x[0] for x in kept}:
        if selected_token_id in logprob_dict:
            kept.append((selected_token_id, get_logprob_value(logprob_dict[selected_token_id])))

    if not kept:
        return None
    logits = torch.tensor([x[1] for x in kept], dtype=torch.float32)
    probs = torch.softmax(logits, dim=0)
    entropy = -torch.sum(probs * torch.log(probs)).item()
    return float(entropy)


def normalize_token(token):
    text = str(token)
    return text.replace(" ", "").strip()


def text_tokens(text):
    if text is None:
        return []
    return [x for x in str(text).replace("\n", " ").split(" ") if x]


def make_record(prompt, output_text, tokens=None, entropies=None, top1_probs=None,
                selected_probs=None, source=None, constraint=None, model=None):
    return {
        "prompt": prompt,
        "output_text": output_text,
        "tokens": tokens or [],
        "entropies": entropies or [],
        "top1_probs": top1_probs or [],
        "selected_probs": selected_probs or [],
        "source": source,
        "constraint": constraint,
        "model": model,
    }


def load_bf_file(path):
    data = safe_load(path)
    if not (isinstance(data, (list, tuple)) and len(data) == 3):
        raise ValueError("Expected a new-format *_bf.pt file: [bf_values, overall_bf, distribution_profile].")

    bf_values, overall_bf, profile = data
    records = []
    outputs_by_prompt = defaultdict(list)
    for item in profile.get("output", []):
        if len(item) >= 4:
            outputs_by_prompt[item[3]].append(item)

    entropy_groups = profile.get("entropy", [])
    prompts = profile.get("prompt", [])
    logprob_groups = profile.get("output_per_token_logprob_truncated", [])

    for prompt_idx, entropy_group in enumerate(entropy_groups):
        prompt = None
        if prompt_idx < len(prompts) and len(prompts[prompt_idx]) >= 3:
            prompt = prompts[prompt_idx][2]
        output_items = outputs_by_prompt.get(prompt_idx, [])
        logprob_group = logprob_groups[prompt_idx] if prompt_idx < len(logprob_groups) else []
        for sample_idx, entropies in enumerate(entropy_group):
            output_text = None
            if sample_idx < len(output_items) and len(output_items[sample_idx]) >= 3:
                output_text = output_items[sample_idx][2]
            selected_probs = []
            if sample_idx < len(logprob_group):
                selected_probs = [float(np.exp(x)) for x in logprob_group[sample_idx] if x is not None]
            records.append(make_record(
                prompt=prompt,
                output_text=output_text,
                tokens=text_tokens(output_text),
                entropies=[float(x) for x in entropies],
                selected_probs=selected_probs,
                source=os.path.basename(path),
            ))
    meta = {"artifact_type": "bf_file", "overall_bf": float(overall_bf), "num_prompt_bf": len(bf_values)}
    return records, meta


def iter_entropy_profile_checkpoint(path, constraint_filter=None, model_filter=None, p_filter=None):
    ckpt = safe_load(path)
    for constraint, model_dict in ckpt.items():
        if constraint_filter is not None and str(constraint) != str(constraint_filter):
            continue
        for model, p_dict in model_dict.items():
            if model_filter is not None and model_filter not in str(model):
                continue
            for p, entropy_profile in p_dict.items():
                if p_filter is not None and str(p) != str(p_filter):
                    continue
                yield constraint, model, p, entropy_profile


def load_entropy_profile(path, constraint_filter=None, model_filter=None, p_filter=None):
    records = []
    profile_count = 0
    for constraint, model, p, entropy_profile in iter_entropy_profile_checkpoint(
            path, constraint_filter, model_filter, p_filter):
        profile_count += 1
        for prompt, output_texts, token_texts, entropies in entropy_profile:
            for sample_idx, entropy in enumerate(entropies):
                output_text = output_texts[sample_idx] if sample_idx < len(output_texts) else None
                tokens = token_texts[sample_idx] if sample_idx < len(token_texts) else text_tokens(output_text)
                records.append(make_record(
                    prompt=prompt,
                    output_text=output_text,
                    tokens=tokens,
                    entropies=[float(x) for x in entropy],
                    source=os.path.basename(path),
                    constraint=constraint,
                    model=model,
                ))
    meta = {"artifact_type": "entropy_profile", "profiles_loaded": profile_count}
    return records, meta


def load_raw_vllm(path, metadata_path=None, model=None, top_p=0.9):
    database = safe_load(path)
    prompts = [None] * len(database)
    if metadata_path is not None and os.path.exists(metadata_path):
        metadata = safe_load(metadata_path)
        if isinstance(metadata, (list, tuple)) and len(metadata) > 0:
            prompts = metadata[0]

    tokenizer = None
    if model is not None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception:
            tokenizer = None

    records = []
    for idx, data in enumerate(database):
        prompt = prompts[idx] if idx < len(prompts) else None
        for output in data.outputs:
            entropies = []
            top1_probs = []
            selected_probs = []
            token_ids = list(output.token_ids)
            for pos, token_id in enumerate(token_ids):
                if pos >= len(output.logprobs):
                    break
                logprob_dict = output.logprobs[pos]
                entropy = entropy_from_logprob_dict(logprob_dict, top_p=top_p, selected_token_id=token_id)
                if entropy is not None:
                    entropies.append(entropy)
                top1 = top1_prob_from_logprob_dict(logprob_dict)
                if top1 is not None:
                    top1_probs.append(top1)
                selected = selected_prob_from_logprob_dict(logprob_dict, token_id)
                if selected is not None:
                    selected_probs.append(selected)

            if tokenizer is not None:
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
            else:
                tokens = [str(x) for x in token_ids]

            records.append(make_record(
                prompt=prompt,
                output_text=getattr(output, "text", None),
                tokens=tokens,
                entropies=entropies,
                top1_probs=top1_probs,
                selected_probs=selected_probs,
                source=os.path.basename(path),
                model=model,
            ))
    meta = {"artifact_type": "raw_vllm", "num_requests": len(database)}
    return records, meta


def infer_artifact_kind(path, explicit_kind=None):
    if explicit_kind is not None and explicit_kind != "auto":
        return explicit_kind
    base = os.path.basename(path)
    if base.endswith("_bf.pt"):
        return "bf_file"
    if base.endswith(".pt.update_full_spectrum"):
        return "raw_vllm"
    if base.endswith(".pt"):
        # The common random-string response dumps and BF inputs share .pt.
        # Try raw_vllm first; entropy checkpoint can be requested explicitly.
        return "raw_vllm"
    return "raw_vllm"


def default_metadata_path(path):
    if path.endswith(".pt.update_full_spectrum"):
        return path.replace(".pt.update_full_spectrum", ".metadata")
    if path.endswith("_bf.pt"):
        return path[:-len("_bf.pt")] + ".metadata"
    if path.endswith(".pt"):
        return path[:-len(".pt")] + ".metadata"
    return None


def load_artifact(path, artifact_kind="auto", metadata_path=None, model=None, top_p=0.9,
                  constraint_filter=None, model_filter=None, p_filter=None):
    kind = infer_artifact_kind(path, artifact_kind)
    if kind == "bf_file":
        records, meta = load_bf_file(path)
    elif kind == "entropy_profile":
        records, meta = load_entropy_profile(
            path,
            constraint_filter=constraint_filter,
            model_filter=model_filter,
            p_filter=p_filter,
        )
    elif kind == "raw_vllm":
        inferred_model = model
        if inferred_model is None:
            inferred_model = infer_model_path_from_basename(extract_path_metadata(path).get("model_basename"))
        inferred_metadata = metadata_path
        if inferred_metadata is None:
            candidate = default_metadata_path(path)
            if candidate is not None and os.path.exists(candidate):
                inferred_metadata = candidate
        records, meta = load_raw_vllm(
            path,
            metadata_path=inferred_metadata,
            model=inferred_model,
            top_p=top_p,
        )
        if inferred_model is not None:
            meta["model_path_for_tokenizer"] = inferred_model
        if inferred_metadata is not None:
            meta["metadata_path"] = inferred_metadata
    else:
        raise ValueError(f"Unknown artifact kind: {kind}")
    meta["artifact_type"] = kind
    return records, meta


def bucket_index(pos, window_size):
    return int(pos // window_size)


def token_metrics_by_window(record, window_size):
    tokens = [normalize_token(x) for x in record["tokens"]]
    prompt_text = record.get("prompt") or ""
    prompt_text = str(prompt_text)
    metrics = defaultdict(lambda: defaultdict(list))
    seen = set()
    for pos, token in enumerate(tokens):
        if token == "":
            continue
        bucket = bucket_index(pos, window_size)
        metrics[bucket]["immediate_repeat"].append(1.0 if pos > 0 and token == tokens[pos - 1] else 0.0)
        metrics[bucket]["seen_before"].append(1.0 if token in seen else 0.0)
        metrics[bucket]["copy_from_prompt_rough"].append(1.0 if token and token in prompt_text else 0.0)
        seen.add(token)
    return metrics


def summarize_records(records, window_size):
    windows = defaultdict(lambda: defaultdict(list))
    record_rows = []

    for rec_id, record in enumerate(records):
        entropies = record["entropies"]
        for pos, entropy in enumerate(entropies):
            bucket = bucket_index(pos, window_size)
            windows[bucket]["entropy"].append(float(entropy))
        for pos, val in enumerate(record.get("top1_probs", [])):
            windows[bucket_index(pos, window_size)]["top1_prob"].append(float(val))
        for pos, val in enumerate(record.get("selected_probs", [])):
            windows[bucket_index(pos, window_size)]["selected_prob"].append(float(val))

        token_metrics = token_metrics_by_window(record, window_size)
        for bucket, metric_dict in token_metrics.items():
            for name, vals in metric_dict.items():
                windows[bucket][name].extend(vals)

        finite_entropies = [x for x in entropies if np.isfinite(x)]
        record_rows.append({
            "record_id": rec_id,
            "source": record.get("source"),
            "constraint": record.get("constraint"),
            "model": record.get("model"),
            "output_len_entropy": len(entropies),
            "output_len_tokens": len(record.get("tokens", [])),
            "mean_entropy": float(np.mean(finite_entropies)) if finite_entropies else "",
            "bf_from_mean_entropy": float(np.exp(np.mean(finite_entropies))) if finite_entropies else "",
            "output_preview": (record.get("output_text") or "")[:200].replace("\r", " ").replace("\n", " "),
        })

    window_rows = []
    for bucket in sorted(windows.keys()):
        row = {
            "window_id": bucket,
            "start_pos": bucket * window_size,
            "end_pos": (bucket + 1) * window_size,
        }
        for name, vals in sorted(windows[bucket].items()):
            finite = [x for x in vals if np.isfinite(x)]
            row[f"{name}_n"] = len(finite)
            row[f"{name}_mean"] = float(np.mean(finite)) if finite else ""
            row[f"{name}_std"] = float(np.std(finite)) if finite else ""
        if row.get("entropy_mean") != "":
            row["bf_from_entropy_mean"] = float(math.exp(row["entropy_mean"]))
        else:
            row["bf_from_entropy_mean"] = ""
        window_rows.append(row)

    return window_rows, record_rows


def _csv_safe_value(value):
    # Collapse any embedded carriage returns / newlines so a single stray
    # control character in model output can never crash csv writing for the
    # whole artifact (was raising "_csv.Error: need to escape, but no
    # escapechar set").
    if isinstance(value, str):
        return value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return value


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_safe_value(val) for key, val in row.items()})


def build_model_index(summaries):
    grouped = defaultdict(list)
    for summary in summaries:
        path_meta = summary.get("path_metadata", {})
        model = path_meta.get("model_basename") or "model_unknown"
        row = flatten_summary_for_index(summary)
        row["constraint_sort_key"] = path_meta.get("constraint_level")
        grouped[model].append(row)

    model_index = {}
    index_rows = []
    for model in sorted(grouped.keys()):
        rows = sorted(
            grouped[model],
            key=lambda x: (
                10**9 if x.get("constraint_sort_key") is None else x.get("constraint_sort_key"),
                str(x.get("artifact_path") or ""),
            ),
        )
        model_index[model] = rows
        for order, row in enumerate(rows):
            output_row = dict(row)
            output_row["model_index_order"] = order
            output_row["model_group"] = model
            index_rows.append(output_row)
    return model_index, index_rows


def write_model_index(output_dir, summaries):
    model_index, index_rows = build_model_index(summaries)
    with open(os.path.join(output_dir, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
    write_csv(os.path.join(output_dir, "model_index.csv"), index_rows)
    return model_index


def numeric_value(value):
    if value == "" or value is None:
        return None
    if isinstance(value, (int, float)):
        if np.isfinite(value):
            return float(value)
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(value):
        return value
    return None


def rows_by_model_and_artifact(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model = row.get("model_basename") or "model_unknown"
        artifact = row.get("artifact_path") or row.get("artifact") or "artifact_unknown"
        grouped[model][artifact].append(row)
    return grouped


def line_label_for_artifact(rows, artifact):
    if not rows:
        return os.path.basename(str(artifact))
    row = rows[0]
    constraint = row.get("constraint_level")
    offset = row.get("prompt_offset_tokens")
    min_p = row.get("min_p")
    max_tokens = row.get("max_tokens")
    label_parts = []
    if constraint is not None:
        label_parts.append(f"C={constraint}")
    if offset is not None:
        label_parts.append(f"offset={offset}")
    if min_p is not None:
        label_parts.append(f"min_p={min_p}")
    if max_tokens is not None:
        label_parts.append(f"T={max_tokens}")
    if label_parts:
        return ", ".join(label_parts)
    return os.path.basename(str(artifact))


def maybe_plot_model_aligned(output_dir, aggregate_rows):
    if not aggregate_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    metric_specs = [
        ("entropy_mean", "Mean entropy"),
        ("bf_from_entropy_mean", "exp(mean entropy)"),
        ("selected_prob_mean", "Selected token probability"),
        ("top1_prob_mean", "Top-1 probability"),
        ("immediate_repeat_mean", "Immediate repeat rate"),
        ("seen_before_mean", "Seen-before token rate"),
        ("copy_from_prompt_rough_mean", "Rough prompt-copy rate"),
    ]
    grouped = rows_by_model_and_artifact(aggregate_rows)
    per_model_root = os.path.join(output_dir, "per_model")
    os.makedirs(per_model_root, exist_ok=True)

    for model, artifact_rows in grouped.items():
        model_dir = os.path.join(per_model_root, sanitize_name(model, max_len=100))
        os.makedirs(model_dir, exist_ok=True)
        model_rows = []
        for rows in artifact_rows.values():
            model_rows.extend(rows)
        write_csv(os.path.join(model_dir, "aligned_window_summary.csv"), model_rows)

        for metric_name, ylabel in metric_specs:
            has_metric = any(numeric_value(row.get(metric_name)) is not None for row in model_rows)
            if not has_metric:
                continue
            plt.figure(figsize=(9, 5))
            plotted = False
            sorted_artifacts = sorted(
                artifact_rows.items(),
                key=lambda item: (
                    10**9 if item[1][0].get("constraint_level") is None else item[1][0].get("constraint_level"),
                    str(item[0]),
                ),
            )
            for artifact, rows in sorted_artifacts:
                points = []
                for row in rows:
                    x = numeric_value(row.get("aligned_start_pos"))
                    y = numeric_value(row.get(metric_name))
                    if x is None or y is None:
                        continue
                    points.append((x, y))
                points.sort(key=lambda x: x[0])
                if not points:
                    continue
                xs = [x for x, _ in points]
                ys = [y for _, y in points]
                plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3, label=line_label_for_artifact(rows, artifact))
                plotted = True
            if not plotted:
                plt.close()
                continue
            plt.xlabel("Aligned output position (generated position + prompt offset)")
            plt.ylabel(ylabel)
            plt.title(model)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f"aligned_{metric_name}.png"), dpi=200)
            plt.close()


def maybe_plot(output_dir, window_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [row["start_pos"] for row in window_rows if row.get("entropy_mean") != ""]
    if not xs:
        return
    entropy = [row["entropy_mean"] for row in window_rows if row.get("entropy_mean") != ""]
    bf = [row["bf_from_entropy_mean"] for row in window_rows if row.get("entropy_mean") != ""]

    plt.figure(figsize=(8, 4))
    plt.plot(xs, entropy, marker="o")
    plt.xlabel("Output position")
    plt.ylabel("Mean entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "window_entropy.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(xs, bf, marker="o")
    plt.xlabel("Output position")
    plt.ylabel("exp(mean entropy)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "window_bf_proxy.png"), dpi=200)
    plt.close()


def flatten_summary_for_index(summary):
    row = {}
    path_meta = summary.get("path_metadata", {})
    meta = summary.get("meta", {})
    for key, value in path_meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[f"meta_{key}"] = value
    row["artifact"] = summary.get("artifact")
    row["artifact_path"] = summary.get("artifact_path")
    row["artifact_output_dir"] = summary.get("artifact_output_dir")
    row["num_records"] = summary.get("num_records")
    row["status"] = summary.get("status", "ok")
    row["error"] = summary.get("error", "")
    row["window_size"] = summary.get("window_size")
    return row


def aggregate_window_rows(window_rows, summary):
    path_meta = summary.get("path_metadata", {})
    rows = []
    constraint_level = path_meta.get("constraint_level")
    multiplier = path_meta.get("word_level_constraint_multiplier")
    prompt_offset = 0
    if isinstance(path_meta.get("model_token_prefix_tokens"), (int, float)):
        prompt_offset = int(path_meta["model_token_prefix_tokens"])
    elif isinstance(constraint_level, (int, float)) and isinstance(multiplier, (int, float)):
        prompt_offset = int(constraint_level * multiplier)
    for row in window_rows:
        merged = {}
        for key, value in path_meta.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                merged[key] = value
        merged["artifact"] = summary.get("artifact")
        merged["artifact_path"] = summary.get("artifact_path")
        merged["artifact_output_dir"] = summary.get("artifact_output_dir")
        merged["prompt_offset_tokens"] = prompt_offset
        merged.update(row)
        if isinstance(merged.get("start_pos"), (int, float)):
            merged["aligned_start_pos"] = merged["start_pos"] + prompt_offset
        if isinstance(merged.get("end_pos"), (int, float)):
            merged["aligned_end_pos"] = merged["end_pos"] + prompt_offset
        if isinstance(merged.get("aligned_start_pos"), (int, float)) and isinstance(merged.get("aligned_end_pos"), (int, float)):
            merged["aligned_mid_pos"] = (merged["aligned_start_pos"] + merged["aligned_end_pos"]) / 2
        rows.append(merged)
    return rows


def process_one_artifact(config):
    path = config["path"]
    output_dir = config["output_dir"]
    try:
        records, meta = load_artifact(
            path,
            artifact_kind=config.get("artifact_kind", "auto"),
            metadata_path=config.get("metadata_path"),
            model=config.get("model"),
            top_p=config.get("top_p", 0.9),
            constraint_filter=config.get("constraint_filter"),
            model_filter=config.get("model_filter"),
            p_filter=config.get("p_filter"),
        )
        os.makedirs(output_dir, exist_ok=True)
        window_rows, record_rows = summarize_records(records, config.get("window_size", 10))
        write_csv(os.path.join(output_dir, "window_summary.csv"), window_rows)
        write_csv(os.path.join(output_dir, "record_summary.csv"), record_rows)
        if not config.get("no_plots", False):
            maybe_plot(output_dir, window_rows)

        path_metadata = extract_path_metadata(path)
        default_multiplier = config.get("default_word_level_constraint_multiplier")
        if path_metadata.get("word_level_constraint_multiplier") is None and default_multiplier is not None:
            path_metadata["word_level_constraint_multiplier"] = default_multiplier
        summary = {
            "status": "ok",
            "artifact": os.path.basename(path),
            "artifact_path": path,
            "artifact_output_dir": output_dir,
            "path_metadata": path_metadata,
            "meta": meta,
            "num_records": len(records),
            "window_size": config.get("window_size", 10),
            "outputs": {
                "window_summary": "window_summary.csv",
                "record_summary": "record_summary.csv",
            },
        }
        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary, window_rows
    except Exception as exc:
        os.makedirs(output_dir, exist_ok=True)
        path_metadata = extract_path_metadata(path)
        default_multiplier = config.get("default_word_level_constraint_multiplier")
        if path_metadata.get("word_level_constraint_multiplier") is None and default_multiplier is not None:
            path_metadata["word_level_constraint_multiplier"] = default_multiplier
        summary = {
            "status": "error",
            "artifact": os.path.basename(path),
            "artifact_path": path,
            "artifact_output_dir": output_dir,
            "path_metadata": path_metadata,
            "meta": {},
            "num_records": 0,
            "window_size": config.get("window_size", 10),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary, []


def expand_patterns(patterns):
    paths = []
    for pattern in patterns:
        expanded = glob.glob(os.path.expanduser(pattern), recursive=True)
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(pattern):
            paths.append(pattern)
    # Preserve stable order while removing duplicates.
    seen = set()
    unique = []
    for path in sorted(paths):
        norm = os.path.abspath(path)
        if norm not in seen:
            seen.add(norm)
            unique.append(path)
    return unique


def load_patterns_file(path):
    patterns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    return patterns


def default_patterns_from_root(root, include_raw=True, include_bf=True, include_update=True):
    patterns = []
    if include_bf:
        patterns.append(os.path.join(root, "**", "*_bf.pt"))
    if include_update:
        patterns.append(os.path.join(root, "**", "*.pt.update_full_spectrum"))
    if include_raw:
        # Exclude _bf.pt after expansion in discover_artifacts.
        patterns.append(os.path.join(root, "**", "*_response_n_*.pt"))
    return patterns


def discover_artifacts(args):
    patterns = []
    explicit_paths = []
    for attr in ["bf_file", "entropy_profile_ckpt", "raw_vllm_file"]:
        value = getattr(args, attr, None)
        if value:
            explicit_paths.append(value)
    explicit_paths.extend(args.artifact_path or [])
    patterns.extend(args.artifact_pattern or [])
    if args.artifact_patterns_file:
        patterns.extend(load_patterns_file(args.artifact_patterns_file))
    if args.artifact_root:
        patterns.extend(default_patterns_from_root(
            args.artifact_root,
            include_raw=args.include_raw,
            include_bf=args.include_bf,
            include_update=args.include_update_full_spectrum,
        ))
    paths = explicit_paths + expand_patterns(patterns)
    if args.skip_bf_files:
        paths = [path for path in paths if not path.endswith("_bf.pt")]
    if args.skip_raw_files:
        paths = [
            path for path in paths
            if path.endswith("_bf.pt")
        ]
    if args.path_regex:
        regex = re.compile(args.path_regex)
        paths = [path for path in paths if regex.search(path.replace("\\", "/"))]
    seen = set()
    unique = []
    for path in paths:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        if path.endswith(".metadata"):
            continue
        seen.add(norm)
        unique.append(path)
    if args.max_artifacts is not None:
        unique = unique[:args.max_artifacts]
    return unique


def requested_patterns(args):
    patterns = []
    for attr in ["bf_file", "entropy_profile_ckpt", "raw_vllm_file"]:
        value = getattr(args, attr, None)
        if value:
            patterns.append(value)
    patterns.extend(args.artifact_path or [])
    patterns.extend(args.artifact_pattern or [])
    if args.artifact_patterns_file:
        patterns.extend(load_patterns_file(args.artifact_patterns_file))
    if args.artifact_root:
        patterns.extend(default_patterns_from_root(
            args.artifact_root,
            include_raw=args.include_raw,
            include_bf=args.include_bf,
            include_update=args.include_update_full_spectrum,
        ))
    return patterns


def pattern_parent_status(pattern):
    wildcard_positions = [pos for pos in [pattern.find("*"), pattern.find("?"), pattern.find("[")] if pos >= 0]
    if wildcard_positions:
        prefix = pattern[:min(wildcard_positions)]
        parent = os.path.dirname(prefix.rstrip("/\\"))
    else:
        parent = os.path.dirname(pattern)
    if not parent:
        parent = "."
    return {
        "pattern": pattern,
        "parent": parent,
        "parent_exists": os.path.exists(os.path.expanduser(parent)),
    }


def no_artifacts_message(args):
    pattern_info = [pattern_parent_status(pattern) for pattern in requested_patterns(args)]
    return (
        "No artifacts found for the provided path(s)/pattern(s).\n"
        "This usually means the filename parameters do not match existing files "
        "(e.g. max_tokens/min_p/top_p or _bf.pt vs raw .pt), or the parent path is not visible.\n"
        f"Tried patterns:\n{json.dumps(pattern_info, indent=2)}"
    )


def explicit_kind_for_path(args, path):
    abs_path = os.path.abspath(path)
    if args.bf_file and os.path.abspath(args.bf_file) == abs_path:
        return "bf_file"
    if args.entropy_profile_ckpt and os.path.abspath(args.entropy_profile_ckpt) == abs_path:
        return "entropy_profile"
    if args.raw_vllm_file and os.path.abspath(args.raw_vllm_file) == abs_path:
        return "raw_vllm"
    return args.artifact_kind


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose random-string BF artifacts for reviewer response.")
    parser.add_argument("--repo_root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--bf_file", help="Path to a new-format *_bf.pt artifact.")
    group.add_argument("--entropy_profile_ckpt", help="Path to entropy_profile_generation checkpoint.")
    group.add_argument("--raw_vllm_file", help="Path to raw vLLM response dump.")
    parser.add_argument("--artifact_path", action="append", default=[], help="Explicit artifact path. Repeatable.")
    parser.add_argument(
        "--artifact_pattern",
        action="append",
        default=[],
        help="Glob pattern for artifacts. Repeatable. Quote patterns containing * in the shell.",
    )
    parser.add_argument("--artifact_patterns_file", default=None, help="Text file with one glob pattern per line.")
    parser.add_argument("--artifact_root", default=None, help="Root searched recursively with default artifact patterns.")
    parser.add_argument("--path_regex", default=None, help="Optional regex filter applied after path discovery.")
    parser.add_argument("--artifact_kind", default="auto", choices=["auto", "bf_file", "raw_vllm", "entropy_profile"])
    parser.add_argument("--include_raw", action="store_true", help="When --artifact_root is used, include raw *.pt dumps.")
    parser.add_argument("--include_bf", action="store_true", help="When --artifact_root is used, include *_bf.pt files.")
    parser.add_argument(
        "--include_update_full_spectrum",
        action="store_true",
        help="When --artifact_root is used, include *.pt.update_full_spectrum files.",
    )
    parser.add_argument("--skip_bf_files", action="store_true")
    parser.add_argument("--skip_raw_files", action="store_true")
    parser.add_argument("--max_artifacts", type=int, default=None)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes. Keep small because each artifact can be large.",
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip per-artifact plots in batch mode.")
    parser.add_argument("--no_model_plots", action="store_true", help="Skip aggregate per-model aligned-position plots.")
    parser.add_argument("--metadata_path", default=None, help="Metadata path for raw vLLM dump.")
    parser.add_argument("--model", default=None, help="HF model path/name for raw token decoding.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p truncation for raw entropy computation.")
    parser.add_argument("--constraint", default=None, help="Optional entropy-profile constraint filter.")
    parser.add_argument("--profile_model_filter", default=None, help="Optional substring filter for entropy-profile model.")
    parser.add_argument("--p", default=None, help="Optional entropy-profile p/min-p key filter.")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument(
        "--default_word_level_constraint_multiplier",
        type=int,
        default=None,
        help="Fallback offset multiplier when filenames omit word_level_constraint_multiplier.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main():
    args = parse_args()
    add_repo_to_path(args.repo_root)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.artifact_root and not (args.include_raw or args.include_bf or args.include_update_full_spectrum):
        args.include_bf = True

    artifact_paths = discover_artifacts(args)
    if not artifact_paths:
        raise ValueError(no_artifacts_message(args))

    configs = []
    for path in artifact_paths:
        configs.append({
            "path": path,
            "output_dir": artifact_output_dir(args.output_dir, path) if len(artifact_paths) > 1 else args.output_dir,
            "artifact_kind": explicit_kind_for_path(args, path),
            "metadata_path": args.metadata_path,
            "model": args.model,
            "top_p": args.top_p,
            "constraint_filter": args.constraint,
            "model_filter": args.profile_model_filter,
            "p_filter": args.p,
            "window_size": args.window_size,
            "default_word_level_constraint_multiplier": args.default_word_level_constraint_multiplier,
            "no_plots": args.no_plots or len(artifact_paths) > 1,
        })

    all_summaries = []
    all_window_rows = []
    worker_count = max(1, int(args.num_workers))
    progress_disabled = args.no_progress
    if worker_count == 1 or len(configs) == 1:
        with tqdm(
                total=len(configs),
                desc="Processing artifacts",
                unit="artifact",
                dynamic_ncols=True,
                disable=progress_disabled) as pbar:
            for config in configs:
                summary, window_rows = process_one_artifact(config)
                all_summaries.append(summary)
                all_window_rows.extend(aggregate_window_rows(window_rows, summary))
                tqdm.write(json.dumps(flatten_summary_for_index(summary), indent=2))
                pbar.update(1)
    else:
        worker_count = min(worker_count, len(configs))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_path = {executor.submit(process_one_artifact, config): config["path"] for config in configs}
            with tqdm(
                    total=len(future_to_path),
                    desc=f"Processing artifacts ({worker_count} workers)",
                    unit="artifact",
                    dynamic_ncols=True,
                    disable=progress_disabled) as pbar:
                for future in as_completed(future_to_path):
                    summary, window_rows = future.result()
                    all_summaries.append(summary)
                    all_window_rows.extend(aggregate_window_rows(window_rows, summary))
                    tqdm.write(json.dumps(flatten_summary_for_index(summary), indent=2))
                    pbar.update(1)

    index_rows = [flatten_summary_for_index(summary) for summary in all_summaries]
    write_csv(os.path.join(args.output_dir, "artifact_index.csv"), index_rows)
    write_csv(os.path.join(args.output_dir, "aggregate_window_summary.csv"), all_window_rows)
    write_model_index(args.output_dir, all_summaries)
    if not args.no_model_plots:
        maybe_plot_model_aligned(args.output_dir, all_window_rows)

    batch_summary = {
        "num_artifacts": len(artifact_paths),
        "num_ok": sum(1 for summary in all_summaries if summary.get("status") == "ok"),
        "num_error": sum(1 for summary in all_summaries if summary.get("status") == "error"),
        "window_size": args.window_size,
        "num_workers": worker_count,
        "outputs": {
            "artifact_index": "artifact_index.csv",
            "model_index": "model_index.csv",
            "model_index_json": "model_index.json",
            "aggregate_window_summary": "aggregate_window_summary.csv",
            "per_model_root": "per_model",
            "per_artifact_root": "per_artifact" if len(artifact_paths) > 1 else ".",
        },
    }
    with open(os.path.join(args.output_dir, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2)
    print(json.dumps(batch_summary, indent=2))


if __name__ == "__main__":
    main()
