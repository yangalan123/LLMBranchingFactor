import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from uncertainty_quantification.loglik_computation import get_logprob_per_token_from_vllm_outputs
import numpy as np
import traceback
import copy
import os
import torch

def visualize_per_token_stats(per_token_stats: Dict, prompt_boundaries: List[int],
                              save_dir: str, prefix: str = "") -> None:
    """
    Create visualizations for per-token statistics.

    Args:
        per_token_stats: Dictionary containing statistics for each token position
        prompt_boundaries: List of prompt lengths for different samples
        save_dir: Directory to save the visualizations
        prefix: Prefix for saved files (useful when comparing different models)
    """
    os.makedirs(save_dir, exist_ok=True)
    avg_prompt_len = np.mean(prompt_boundaries)

    # Plot 1: Prompt length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(prompt_boundaries, bins=30, color='blue', alpha=0.7)
    plt.axvline(x=avg_prompt_len, color='r', linestyle='--',
                label=f'Average length ({avg_prompt_len:.1f})')
    plt.xlabel('Prompt Length')
    plt.ylabel('Count')
    plt.title('Distribution of Prompt Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'{prefix}prompt_distribution.pdf'), bbox_inches='tight')
    plt.close()

    # Plot 2: Top-k overlap trends
    plt.figure(figsize=(15, 8))
    token_positions = sorted(per_token_stats.keys())
    ks_to_plot = [1, 5, 10]  # Adjust these k values as needed

    for k in ks_to_plot:
        means = []
        stds = []
        for pos in token_positions:
            if k in per_token_stats[pos]["w_prompt"]:
                values = per_token_stats[pos]["w_prompt"][k]
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(0)
                stds.append(0)

        plt.plot(token_positions, means, label=f'Top-{k} overlap', marker='o', markersize=2)
        plt.fill_between(token_positions,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.2)

    plt.axvline(x=avg_prompt_len, color='r', linestyle='--',
                label=f'Avg prompt length ({avg_prompt_len:.1f})')

    plt.xlabel('Token Position')
    plt.ylabel('Overlap Ratio')
    plt.title('Top-k Overlap Trends Across Token Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'{prefix}topk_trends.pdf'), bbox_inches='tight')
    plt.close()

    # Plot 3: Agreement types across positions
    plt.figure(figsize=(15, 8))
    metrics = ['top1_agree', 'top1_weakly_disagree', 'top1_disagree']
    colors = ['g', 'y', 'r']

    for metric, color in zip(metrics, colors):
        means = []
        stds = []
        for pos in token_positions:
            values = per_token_stats[pos][metric]
            means.append(np.mean(values))
            stds.append(np.std(values))

        plt.plot(token_positions, means, label=metric, color=color, marker='o', markersize=2)
        plt.fill_between(token_positions,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.2,
                         color=color)

    plt.axvline(x=avg_prompt_len, color='r', linestyle='--',
                label=f'Avg prompt length ({avg_prompt_len:.1f})')

    plt.xlabel('Token Position')
    plt.ylabel('Agreement Ratio')
    plt.title('Agreement Types Across Token Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'{prefix}agreement_trends.pdf'), bbox_inches='tight')
    plt.close()

    # Plot 4: Heatmap of top-k overlaps
    plt.figure(figsize=(15, 8))
    max_k = max(max(stats["w_prompt"].keys()) for stats in per_token_stats.values())
    heatmap_data = np.zeros((max_k, len(token_positions)))

    for j, pos in enumerate(token_positions):
        for k in range(1, max_k + 1):
            if k in per_token_stats[pos]["w_prompt"]:
                heatmap_data[k - 1, j] = np.mean(per_token_stats[pos]["w_prompt"][k])

    sns.heatmap(heatmap_data, cmap='YlOrRd',
                xticklabels=token_positions[::5],  # Show every 5th position
                yticklabels=range(1, max_k + 1))
    plt.axvline(x=np.where(np.array(token_positions) >= avg_prompt_len)[0][0],
                color='r', linestyle='--')

    plt.xlabel('Token Position')
    plt.ylabel('Top-k')
    plt.title('Top-k Overlap Heatmap Across Token Positions')
    plt.savefig(os.path.join(save_dir, f'{prefix}topk_heatmap.pdf'), bbox_inches='tight')
    plt.close()

def compare_original_response_and_patch_response(nudging_output_root_dir, original_responses, patch_data, patch_exist_flag, logger, args):
    stat_output_dir = os.path.join(nudging_output_root_dir, "stat")
    os.makedirs(stat_output_dir, exist_ok=True)
    stat_ckpt_filename = os.path.join(stat_output_dir, "top_k_overlapping_{}_vs_{}.pt".format(os.path.basename(args.model), os.path.basename(args.eval_model)))
    # corresponding to nudging paper, note, top1_agree should match w_prompt[1]
    all_nudging_flags = ['top1_agree', 'top1_weakly_disagree', 'top1_disagree']
    empty_token_stats = {
        "w_prompt": {},
        "wo_prompt": {},
        "origin_i": 0,
        "output_i": 0,
    }
    for _flag in all_nudging_flags:
        empty_token_stats[_flag] = []
    if os.path.exists(stat_ckpt_filename) and not args.force_recompute:
        counter, cross_token_stats, per_token_stats, prompt_boundaries = torch.load(stat_ckpt_filename)
    else:
        counter = 0
        # empty_token_stats = copy.deepcopy(cross_token_stats)
        cross_token_stats = copy.deepcopy(empty_token_stats)
        per_token_stats = dict()
        prompt_boundaries = []
    for origin_i, origin_response in enumerate(original_responses):
        if counter >= len(patch_data):
            logger.info(f"Run out of patch data: {counter} >= {len(patch_data)}")
            break
        if origin_i < cross_token_stats["origin_i"]:
            continue
        cross_token_stats["origin_i"] = origin_i
        prompt_token_ids = origin_response.prompt_token_ids
        prompt_logprobs = origin_response.prompt_logprobs
        prompt_boundaries.append(len(prompt_token_ids))
        for output_i, output in enumerate(origin_response.outputs):
            if counter >= len(patch_data):
                logger.info(f"Run out of patch data: {counter} >= {len(patch_data)}")
                break
            if output_i < cross_token_stats["output_i"]:
                counter += 1
                continue
            cross_token_stats["output_i"] = output_i
            output_token_ids = output.token_ids
            output_logprobs = output.logprobs
            original_all_token_ids = list(prompt_token_ids) + list(output_token_ids)
            # sometimes, the prompt_logprobs might be None (just-eval-instruct cases), so we need to skip them
            if prompt_logprobs is None:
                prompt_logprobs = [{} for _ in range(len(prompt_token_ids))]
            original_all_logprobs = list(prompt_logprobs) + list(output_logprobs)
            if patch_exist_flag:
                patch = patch_data[counter]
                patch_token_ids = patch.prompt_token_ids
                patch_all_logprobs = patch.prompt_logprobs
                assert len(prompt_token_ids) + len(output_token_ids) == len(patch_token_ids), "Token length mismatch: {} + {} != {}".format(len(prompt_token_ids), len(output_token_ids), len(patch_token_ids))
                assert original_all_token_ids == patch_token_ids, "Token mismatch: {} != {}".format(original_all_token_ids, patch_token_ids)
                assert len(original_all_logprobs) == len(patch_all_logprobs), "Logprob length mismatch: {} != {}".format(len(original_all_logprobs), len(patch_all_logprobs))
            else:
                # if using spectrum
                patch = patch_data[origin_i]
                patch_token_ids = list(patch.prompt_token_ids) + list(patch.outputs[output_i].token_ids)
                patch_all_logprobs = list(patch.prompt_logprobs) + list(patch.outputs[output_i].logprobs)
                assert len(prompt_token_ids) + len(output_token_ids) == len(patch_token_ids), "Token length mismatch: {} + {} != {}".format(len(prompt_token_ids), len(output_token_ids), len(patch_token_ids))
                assert original_all_token_ids == patch_token_ids, "Token mismatch: {} != {}".format(original_all_token_ids, patch_token_ids)
                assert len(original_all_logprobs) == len(patch_all_logprobs), "Logprob length mismatch: {} != {}".format(
                    len(original_all_logprobs), len(patch_all_logprobs))
            # token-0 in prompt_token_probs is often none, so start with 1
            for token_i in range(1, len(original_all_logprobs)):
                if len(original_all_logprobs[token_i]) == 0:
                    continue
                if token_i not in per_token_stats:
                    per_token_stats[token_i] = copy.deepcopy(empty_token_stats)
                original_logprobs = {k: get_logprob_per_token_from_vllm_outputs(v) for k, v in original_all_logprobs[token_i].items() if not np.isnan(get_logprob_per_token_from_vllm_outputs(v))}
                try:
                    patch_logprobs = {k: get_logprob_per_token_from_vllm_outputs(v) for k, v in patch_all_logprobs[token_i].items() if not np.isnan(get_logprob_per_token_from_vllm_outputs(v))}
                except Exception as e:
                    print("Error in patch logprobs: {}".format(token_i))
                    print("patch exist flag: {}".format(patch_exist_flag))
                    print(traceback.print_exc())
                    if token_i >= len(prompt_token_ids):
                        exit()
                    continue
                original_sorted = sorted(original_logprobs.items(), key=lambda x: x[1], reverse=True)
                patch_sorted = sorted(patch_logprobs.items(), key=lambda x: x[1], reverse=True)
                tmp_overlaps = []
                # Calculate overlaps for both cross-token and per-token stats
                for k in range(min(len(original_sorted), len(patch_sorted))):
                    original_topk = set([x[0] for x in original_sorted[:k + 1]])
                    patch_topk = set([x[0] for x in patch_sorted[:k + 1]])
                    overlap = len(original_topk.intersection(patch_topk))

                    # Update cross-token stats
                    if k + 1 not in cross_token_stats["w_prompt"]:
                        cross_token_stats["w_prompt"][k + 1] = []
                    cross_token_stats["w_prompt"][k + 1].append(overlap)

                    # Update per-token stats (always update w_prompt)
                    if k + 1 not in per_token_stats[token_i]["w_prompt"]:
                        per_token_stats[token_i]["w_prompt"][k + 1] = []
                    per_token_stats[token_i]["w_prompt"][k + 1].append(overlap)

                    tmp_overlaps.append((k, overlap))

                # Handle non-prompt tokens
                if token_i >= len(prompt_token_ids):
                    for k, overlap in tmp_overlaps:
                        # Update cross-token stats for non-prompt
                        if k + 1 not in cross_token_stats["wo_prompt"]:
                            cross_token_stats["wo_prompt"][k + 1] = []
                        cross_token_stats["wo_prompt"][k + 1].append(overlap)

                        # Update per-token stats for non-prompt
                        if k + 1 not in per_token_stats[token_i]["wo_prompt"]:
                            per_token_stats[token_i]["wo_prompt"][k + 1] = []
                        per_token_stats[token_i]["wo_prompt"][k + 1].append(overlap)

                # Update top-1 agreement statistics for both cross-token and per-token
                top_1_token_original = original_sorted[0][0]
                top_1_token_patch = patch_sorted[0][0]

                if top_1_token_original == top_1_token_patch:
                    # Update cross-token stats
                    flag = "top1_agree"
                else:
                    # Check for weak disagreement
                    top_3_token_patch = [x[0] for x in patch_sorted[:3]]
                    if top_1_token_original in top_3_token_patch:
                        flag = "top1_weakly_disagree"
                    else:
                        flag = "top1_disagree"
                for _flag in all_nudging_flags:
                    if _flag == flag:
                        cross_token_stats[_flag].append(1)
                        per_token_stats[token_i][_flag].append(1)
                    else:
                        cross_token_stats[_flag].append(0)
                        per_token_stats[token_i][_flag].append(0)


            counter += 1
    torch.save((counter, cross_token_stats), stat_ckpt_filename)
    # print averaged statistics
    logger.info("Base: {}, Eval: {}".format(os.path.basename(args.model), os.path.basename(args.eval_model)))
    logger.info("w_prompt top-k statistics:")
    for k in range(1, len(cross_token_stats["w_prompt"]) + 1):
        logger.info("Top-{}: {} ({}) [{}]".format(k,
                                                  np.mean(cross_token_stats["w_prompt"][k]),
                                                  np.std(cross_token_stats["w_prompt"][k]),
                                                  len(cross_token_stats["w_prompt"][k])))
    logger.info("wo_prompt top-k statistics:")
    for k in range(1, len(cross_token_stats["wo_prompt"]) + 1):
        logger.info("Top-{}: {} ({}) [{}]".format(k, np.mean(cross_token_stats["wo_prompt"][k]), np.std(cross_token_stats["wo_prompt"][k]), len(cross_token_stats["wo_prompt"][k])))

    logger.info('Nudging statistics: ')
    logger.info('Top-1 Agree: {} ({}) [{}]'.format(np.mean(cross_token_stats["top1_agree"]), np.std(cross_token_stats["top1_agree"]), len(cross_token_stats["top1_agree"])))
    logger.info('Top-1 Weakly Disagree: {} ({}) [{}]'.format(np.mean(cross_token_stats["top1_weakly_disagree"]), np.std(cross_token_stats["top1_weakly_disagree"]), len(cross_token_stats["top1_weakly_disagree"])))
    logger.info('Top-1 Disagree: {} ({}) [{}]'.format(np.mean(cross_token_stats["top1_disagree"]), np.std(cross_token_stats["top1_disagree"]), len(cross_token_stats["top1_disagree"])))

    viz_dir = os.path.join(stat_output_dir, "per_token_visualizations")
    visualize_per_token_stats(
        per_token_stats=per_token_stats,
        prompt_boundaries=prompt_boundaries,
        save_dir=viz_dir,
        prefix=f"{os.path.basename(args.model)}_{os.path.basename(args.eval_model)}_"
    )
    print("Visualization saved in: {}".format(viz_dir))

