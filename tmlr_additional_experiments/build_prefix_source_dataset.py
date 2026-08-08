import argparse
import json
import os
import random
from typing import Optional

import torch
from tqdm import tqdm


def safe_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_tokenizer(model: Optional[str]):
    if model is None:
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


def decode_prefix(tokenizer, token_ids, prefix_tokens):
    token_ids = list(token_ids)[:prefix_tokens]
    if tokenizer is None:
        return " ".join(str(x) for x in token_ids)
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def get_special_token_ids(tokenizer):
    if tokenizer is None:
        return set()
    special = set()
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        special.add(int(token_id))
    return special


def sample_iid_vocab_prefixes(tokenizer, n, prefix_tokens, seed, show_progress=True):
    rng = random.Random(seed)
    if tokenizer is None:
        raise ValueError("--model is required for iid_vocab mode.")
    special = get_special_token_ids(tokenizer)
    vocab_size = len(tokenizer)
    prefixes = []
    with tqdm(
            total=n,
            desc="Sampling iid vocab prefixes",
            unit="prefix",
            dynamic_ncols=True,
            disable=not show_progress) as pbar:
        while len(prefixes) < n:
            ids = []
            while len(ids) < prefix_tokens:
                token_id = rng.randrange(vocab_size)
                if token_id not in special:
                    ids.append(token_id)
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            if text:
                prefixes.append(text)
                pbar.update(1)
    return prefixes


def sample_existing_random_strings(random_strings_pt, max_examples, seed):
    rng = random.Random(seed)
    strings = safe_load(random_strings_pt)
    strings = [str(x).strip() for x in strings if str(x).strip()]
    if max_examples is not None and len(strings) > max_examples:
        strings = rng.sample(strings, max_examples)
    return strings


def load_raw_outputs(raw_vllm_file, show_progress=True):
    database = safe_load(raw_vllm_file)
    outputs = []
    for data in tqdm(
            database,
            desc="Loading raw outputs",
            unit="request",
            dynamic_ncols=True,
            disable=not show_progress):
        for output in data.outputs:
            outputs.append(output)
    return outputs


def load_metadata_prompts(metadata_path):
    metadata = safe_load(metadata_path)
    if not isinstance(metadata, (list, tuple)) or len(metadata) == 0:
        raise ValueError("Metadata file must be a tuple/list whose first element is prompts.")
    return list(metadata[0])


def build_model_generated(raw_vllm_file, tokenizer, prefix_tokens, max_examples, show_progress=True):
    outputs = load_raw_outputs(raw_vllm_file, show_progress=show_progress)
    prefixes = []
    for output in tqdm(
            outputs,
            desc="Decoding model-generated prefixes",
            unit="output",
            dynamic_ncols=True,
            disable=not show_progress):
        text = decode_prefix(tokenizer, output.token_ids, prefix_tokens)
        if text:
            prefixes.append(text)
        if max_examples is not None and len(prefixes) >= max_examples:
            break
    return prefixes


def build_shuffled_model_generated(raw_vllm_file, tokenizer, prefix_tokens, max_examples, seed, show_progress=True):
    rng = random.Random(seed)
    outputs = load_raw_outputs(raw_vllm_file, show_progress=show_progress)
    prefixes = []
    for output in tqdm(
            outputs,
            desc="Decoding shuffled model prefixes",
            unit="output",
            dynamic_ncols=True,
            disable=not show_progress):
        ids = list(output.token_ids)[:prefix_tokens]
        if len(ids) < 2:
            continue
        rng.shuffle(ids)
        text = decode_prefix(tokenizer, ids, prefix_tokens)
        if text:
            prefixes.append(text)
        if max_examples is not None and len(prefixes) >= max_examples:
            break
    return prefixes


def build_original_prompts(metadata_path, max_examples):
    prompts = load_metadata_prompts(metadata_path)
    prompts = [str(x).strip() for x in prompts if str(x).strip()]
    if max_examples is not None:
        prompts = prompts[:max_examples]
    return prompts


def build_natural_text_file(text_file, max_examples, prefix_words, show_progress=True):
    prefixes = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in tqdm(
                f,
                desc="Reading natural text",
                unit="line",
                dynamic_ncols=True,
                disable=not show_progress):
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if prefix_words is not None and prefix_words > 0:
                line = " ".join(words[:prefix_words])
            if line:
                prefixes.append(line)
            if max_examples is not None and len(prefixes) >= max_examples:
                break
    return prefixes


AGENT_SCENARIOS = [
    {
        "task": "Play a simplified chess endgame as White. Your goal is to force a win while avoiding stalemate.",
        "state": "Current board: White king on e5, white queen on d4, black king on g7. It is White to move.",
        "plan": "Plan so far: restrict the black king first, then bring the white king closer before delivering checkmate.",
        "control_feedback": "Your last move improved the queen's control of the seventh rank. The black king still has only two legal squares.",
        "adversarial_feedback": "The black king found an escape square because your queen left the diagonal uncovered. A direct check now risks stalemate or repetition.",
    },
    {
        "task": "Control a warehouse robot that must move a fragile package from shelf A to packing station D.",
        "state": "Current state: the robot is at shelf A, the package is secured, corridor B is open, and station D is available.",
        "plan": "Plan so far: move through corridor B, avoid sharp turns, then place the package on the padded tray at station D.",
        "control_feedback": "The robot reached corridor B without collision. The package remains stable and station D is still available.",
        "adversarial_feedback": "A cart is now blocking corridor B, and the package sensor reports a loose grip. The original route is no longer safe.",
    },
    {
        "task": "Debug a small Python data pipeline and decide the next fix.",
        "state": "Current state: the parser loads a CSV file, validates each row, and then writes normalized records to disk.",
        "plan": "Plan so far: reproduce the failing row, verify the schema check, then patch the narrowest failing component.",
        "control_feedback": "The failing row was reproduced. The schema check correctly rejects the malformed timestamp field.",
        "adversarial_feedback": "The schema check passed on the failing row, but a later inspection shows the input file can be empty and the parser silently returns None.",
    },
    {
        "task": "Navigate a search-and-rescue drone through a building to locate a missing person.",
        "state": "Current state: the drone is in hallway H1, the target beacon is strongest toward room R3, and the battery is at 62%.",
        "plan": "Plan so far: enter R3, scan the room, then return through H1 if the beacon weakens.",
        "control_feedback": "The drone entered R3 successfully. The beacon grew stronger and the path back to H1 remains clear.",
        "adversarial_feedback": "Smoke filled R3 and the beacon signal reflected from a metal door. The direct path forward is uncertain and battery use increased.",
    },
]


def random_ascii_noise(rng, length):
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:-_"
    return "".join(rng.choice(alphabet) for _ in range(length))


def build_structured_feedback(mode, max_examples, seed, noise_chars, show_progress=True):
    rng = random.Random(seed)
    prompts = []
    for idx in tqdm(
            range(max_examples),
            desc="Building structured feedback prompts",
            unit="prompt",
            dynamic_ncols=True,
            disable=not show_progress):
        scenario = AGENT_SCENARIOS[idx % len(AGENT_SCENARIOS)]
        prefix = (
            "You are an agent interacting with an environment. Maintain a multi-step plan, "
            "update it after each environment message, and choose the next action.\n\n"
            f"Task: {scenario['task']}\n\n"
            f"{scenario['state']}\n\n"
            f"{scenario['plan']}\n\n"
        )
        if mode == "structured_feedback_control":
            feedback = scenario["control_feedback"]
        elif mode == "structured_feedback_adversarial":
            feedback = scenario["adversarial_feedback"]
        elif mode == "structured_feedback_random_noise":
            feedback = random_ascii_noise(rng, noise_chars)
        else:
            raise ValueError(f"Unknown structured feedback mode: {mode}")
        prompt = (
            prefix
            + f"Environment Feedback: {feedback}\n\n"
            + "Given this feedback, revise the plan if needed and produce the next reasoning step and action."
        )
        prompts.append(prompt)
    return prompts


def write_jsonl(path, prefixes, mode):
    with open(path, "w", encoding="utf-8") as f:
        for idx, prefix in enumerate(prefixes):
            f.write(json.dumps({"id": idx, "mode": mode, "prompt": prefix}, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build generation-randomness control datasets compatible with demo/demo.py."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "unstructured_random",
            "self_conditioned_random",
            "shuffled_self_conditioned_random",
            "iid_vocab_random",
            "original_prompts",
            "natural_text_file",
            "structured_feedback_control",
            "structured_feedback_adversarial",
            "structured_feedback_random_noise",
        ],
    )
    parser.add_argument("--random_strings_pt", default=None, help="Existing random-string .pt list for unstructured mode.")
    parser.add_argument("--raw_vllm_file", default=None, help="Raw vLLM response dump for self-conditioned modes.")
    parser.add_argument("--metadata_path", default=None, help="Metadata file for original prompt mode.")
    parser.add_argument("--model", default=None, help="HF model path/name for tokenizer-dependent modes.")
    parser.add_argument("--text_file", default=None, help="One-prefix-per-line text file for natural_text_file mode.")
    parser.add_argument("--prefix_tokens", type=int, default=64)
    parser.add_argument("--prefix_words", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--noise_chars", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_pt", required=True)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    tokenizer = load_tokenizer(args.model) if args.model is not None else None
    show_progress = not args.no_progress

    if args.mode == "unstructured_random":
        if args.random_strings_pt is None:
            raise ValueError("--random_strings_pt is required for unstructured_random mode.")
        prefixes = sample_existing_random_strings(args.random_strings_pt, args.max_examples, args.seed)
    elif args.mode == "self_conditioned_random":
        if args.raw_vllm_file is None:
            raise ValueError("--raw_vllm_file is required for self_conditioned_random mode.")
        prefixes = build_model_generated(
            args.raw_vllm_file, tokenizer, args.prefix_tokens, args.max_examples, show_progress=show_progress)
    elif args.mode == "shuffled_self_conditioned_random":
        if args.raw_vllm_file is None:
            raise ValueError("--raw_vllm_file is required for shuffled_self_conditioned_random mode.")
        prefixes = build_shuffled_model_generated(
            args.raw_vllm_file, tokenizer, args.prefix_tokens, args.max_examples, args.seed,
            show_progress=show_progress)
    elif args.mode == "iid_vocab_random":
        prefixes = sample_iid_vocab_prefixes(
            tokenizer, args.max_examples, args.prefix_tokens, args.seed, show_progress=show_progress)
    elif args.mode == "original_prompts":
        if args.metadata_path is None:
            raise ValueError("--metadata_path is required for original_prompts mode.")
        prefixes = build_original_prompts(args.metadata_path, args.max_examples)
    elif args.mode == "natural_text_file":
        if args.text_file is None:
            raise ValueError("--text_file is required for natural_text_file mode.")
        prefixes = build_natural_text_file(
            args.text_file, args.max_examples, args.prefix_words, show_progress=show_progress)
    elif args.mode in {
        "structured_feedback_control",
        "structured_feedback_adversarial",
        "structured_feedback_random_noise",
    }:
        prefixes = build_structured_feedback(
            args.mode, args.max_examples, args.seed, args.noise_chars, show_progress=show_progress)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_pt)), exist_ok=True)
    torch.save(prefixes, args.output_pt)
    if args.output_jsonl is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
        write_jsonl(args.output_jsonl, prefixes, args.mode)

    print(json.dumps({
        "mode": args.mode,
        "num_prefixes": len(prefixes),
        "output_pt": args.output_pt,
        "output_jsonl": args.output_jsonl,
        "preview": prefixes[:3],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
