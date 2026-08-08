"""Multi-turn agentic Branching Factor (BF) harness.

This is the *real* step-wise experiment (option #5): instead of a single static
"feedback" prompt, it runs a generate -> environment-feedback -> regenerate loop and
measures BF at every interaction turn. For each feedback condition (control /
adversarial / random_noise) and each scenario it:

  1. renders the running conversation into a prompt,
  2. samples N continuations with vLLM (the model's "next action"),
  3. builds the standard distribution_profile and computes BF for that turn,
  4. appends the model's chosen action + the environment's next feedback, and loops.

It reuses the shared building blocks rather than re-implementing them:
  * uncertainty_quantification.manager.ForwardManager        (vLLM generation)
  * loglik_computation.build_distribution_profile_from_responses (entropy/loglik)
  * uncertainty_computation.compute_overall_bf_from_profile   (BF)

Outputs (under <output_dir>/<model>/):
  * agentic_multistep_bf.csv  -> model, condition, turn, bf, bf_std, n_prompts
  * trajectories.jsonl        -> the full conversation per scenario/condition
  * summary.json              -> run config + records
Plot with reviewer_mvhn_experiments/plot_agentic_stepwise_bf.py.

This script needs the cluster env (vLLM + torch); run it via
slurm/04_run_agentic_multistep_bf.sh.
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
for _p in (THIS_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer

from uncertainty_quantification.arg_utils import step1_forward_args
from uncertainty_quantification.manager import ForwardManager
from uncertainty_quantification.loglik_computation import build_distribution_profile_from_responses
from uncertainty_quantification.uncertainty_computation import compute_overall_bf_from_profile

from agentic_multistep_scenarios import (
    MULTI_TURN_SCENARIOS,
    CONDITIONS,
    SYSTEM_PROMPT,
    initial_user_message,
    feedback_text,
)


def render_prompt(model_name, tokenizer, messages):
    """Render a chat message list into a prompt string.

    Chat/instruct models use the tokenizer chat template; base models fall back to a
    plain labeled transcript ending with an 'Assistant:' cue.
    """
    is_chat = ("chat" in model_name.lower()) or ("instruct" in model_name.lower())
    if is_chat and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    lines = []
    for message in messages:
        if message["role"] == "assistant":
            lines.append("Assistant: " + message["content"])
        else:  # system / user
            lines.append(message["content"])
    lines.append("Assistant:")
    return "\n\n".join(lines)


def run_condition(condition, scenarios, manager, tokenizer, args, ckpt_dir):
    """Run the multi-turn loop for one feedback condition; return (records, trajectories)."""
    rng = random.Random(args.seed)
    convos = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_user_message(scenario)},
        ]
        for scenario in scenarios
    ]

    records = []
    for turn in range(args.num_turns):
        prompts = [render_prompt(args.model, tokenizer, convo) for convo in convos]
        ckpt = os.path.join(ckpt_dir, f"{condition}_turn{turn}.pt")
        response = manager.forward(
            prompts, ckpt,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        profile = build_distribution_profile_from_responses(
            response, prompts, args.top_p, show_progress=not args.no_progress)
        bf_values, mean_bf = compute_overall_bf_from_profile(
            profile, asymptotic_limit=args.asymptotic_limit)
        records.append({
            "model": args.model_basename,
            "condition": condition,
            "turn": turn,
            "bf": float(mean_bf),
            "bf_std": float(np.std(bf_values)) if len(bf_values) else 0.0,
            "n_prompts": int(len(bf_values)),
        })
        print(f"[{condition}] turn {turn}: BF={mean_bf:.3f} (n={len(bf_values)})")

        # Advance every scenario's conversation with the chosen action + next feedback.
        for i, convo in enumerate(convos):
            chosen = response[i].outputs[0].text.strip()
            convo.append({"role": "assistant", "content": chosen})
            if turn + 1 < args.num_turns:
                feedback = feedback_text(scenarios[i], turn + 1, condition, rng, args.noise_chars)
                convo.append({"role": "user", "content": f"Environment Feedback: {feedback}"})

    trajectories = [
        {"condition": condition, "scenario_index": i,
         "task": scenarios[i]["task"], "messages": convos[i]}
        for i in range(len(scenarios))
    ]
    return records, trajectories


def write_csv(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = ["model", "condition", "turn", "bf", "bf_std", "n_prompts"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model + sampling args shared with demo.py (defaults tuned smaller for a loop).
    parser = step1_forward_args(parser, sample_counts=20, max_tokens=200, top_p=0.9)
    parser.add_argument("--output_dir", default=os.path.join(THIS_DIR, "outputs", "agentic_multistep_bf"))
    parser.add_argument("--num_turns", type=int, default=5, help="Number of interaction turns.")
    parser.add_argument("--max_scenarios", type=int, default=None,
                        help="Limit the number of scenarios (default: all).")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--asymptotic_limit", type=int, default=50,
                        help="Entropy->loglik switch length for BF (see compute_bf_values).")
    parser.add_argument("--noise_chars", type=int, default=80, help="Length of random_noise feedback.")
    parser.add_argument("--max_num_seqs", type=int, default=128)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model:
        raise SystemExit("--model is required.")
    args.model_basename = os.path.basename(args.model.rstrip("/"))

    scenarios = MULTI_TURN_SCENARIOS
    if args.max_scenarios is not None:
        scenarios = scenarios[:args.max_scenarios]

    run_dir = os.path.join(args.output_dir, args.model_basename)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # ForwardManager keeps its temp/checkpoint store under output_root_dir.
    args.output_root_dir = run_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    manager = ForwardManager(args, ckpt_freq=args.ckpt_freq)
    manager.setup_model(max_num_seqs=args.max_num_seqs,
                        gpu_memory_utilization=args.gpu_memory_utilization)

    all_records = []
    all_trajectories = []
    for condition in args.conditions:
        records, trajectories = run_condition(
            condition, scenarios, manager, tokenizer, args, ckpt_dir)
        all_records.extend(records)
        all_trajectories.extend(trajectories)

    csv_path = os.path.join(run_dir, "agentic_multistep_bf.csv")
    write_csv(csv_path, all_records)
    write_jsonl(os.path.join(run_dir, "trajectories.jsonl"), all_trajectories)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_basename,
            "num_turns": args.num_turns,
            "conditions": args.conditions,
            "num_scenarios": len(scenarios),
            "sample_counts": args.sample_counts,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "asymptotic_limit": args.asymptotic_limit,
            "records": all_records,
        }, f, indent=2)
    print(f"Wrote {len(all_records)} per-turn BF rows to {csv_path}")


if __name__ == "__main__":
    main()
