Self-Narrowing and Unexpected-Context Controls
==============================================

This folder reproduces the camera-ready experiments that ask why BF decreases
and whether it can increase. The experiments separate three sources of context:

1. Unstructured random context: random-string prompts with no intended semantic
   task structure.
2. Self-conditioned random context: prefixes produced by the model while
   continuing random strings, testing whether autoregressive generation creates
   its own constraining structure.
3. Structured external perturbation: matched agentic/adversarial feedback,
   testing whether new external information can locally reopen the branching
   space.

The scripts are path-agnostic. Inputs and cluster settings are supplied through
arguments or environment variables; no private paths or credentials are needed.

Goals
-----

1. Diagnose existing random-string artifacts.
   Use `random_string_artifact_diagnostics.py` on one of:
   - a `*_bf.pt` file from `demo/demo.py`
   - an entropy-profile checkpoint from `entropy_profile_generation.py`
   - a raw vLLM response dump such as `*.pt` or `*.pt.update_full_spectrum`
   The same script also supports batch mode with recursive glob patterns.

2. Build generation-randomness control datasets.
   Use `build_generation_randomness_control_dataset.py` to create small `.pt`
   datasets with modes:
   - `unstructured_random`
   - `self_conditioned_random`
   - `shuffled_self_conditioned_random`
   - `iid_vocab_random`
   - `structured_feedback_control`
   - `structured_feedback_adversarial`
   - `structured_feedback_random_noise`

3. Reuse the existing `demo/demo.py` pipeline.
   The SLURM templates in `slurm/` show how to run the generated datasets
   through the existing BF computation without changing core experiment code.

4. Build exact model-token prefix datasets from existing random-string artifacts.
   Use `build_model_token_prefix_random_string_datasets.py` when comparing
   different constraint levels. It groups artifacts by model, loads that model's
   tokenizer, and converts the original random strings into prompts whose prefix
   length is exactly measured in that model's tokens.

Legacy SLURM workflow
---------------------

1. Build mixed control datasets:
   `slurm/legacy_build_prefix_source_datasets.sh`

   Required environment variables:
   `PROJECT_DIR`, `ENV_PATH`, `RANDOM_STRINGS_PT`, `RAW_VLLM_FILE`, `MODEL`.

2. Run BF on the generated controls:
   `slurm/02_run_prefix_source_bf_array.sh`

   Required environment variables:
   `PROJECT_DIR`, `ENV_PATH`, `MODEL`, `CONTROL_DATA_DIR`.

3. Summarize BF outputs:
   `slurm/03_summarize_prefix_source_results.sh`

   Required environment variables:
   `PROJECT_DIR`, `ENV_PATH`, `RESULTS_ROOT`.

Exact token-prefix workflow
---------------------------

This is the safer workflow for the multi-constraint random-string artifacts,
because the old `word_level_constraint_multiplier` naming came from string word
counts rather than model-token counts.

1. Build model-token-prefix datasets:
   `slurm/01a_build_model_token_prefix_datasets.sh`

   Required environment variables:
   `PROJECT_DIR`, `ENV_PATH`, `RANDOM_STRINGS_PT`, `OUTPUT_DATA_DIR`, and either
   `ARTIFACT_PATTERN` or `ARTIFACT_PATTERNS_FILE`.

   Optional variables:
   `MODEL_MAP_JSON`, `CONSTRAINTS`, `LOWEST_CONSTRAINT_LEVEL`,
   `TOKEN_MULTIPLIER`, `MAX_EXAMPLES`, `SEED`.

2. Submit the manifest array:
   `slurm/04_run_model_token_prefix_bf_array.sh`

   Set `MANIFEST` to the generated `manifest.csv` or `manifest.jsonl`. Update
   the SLURM array range to match the number of manifest rows. Each output root
   includes `model_token_prefix_tokens_<N>`, which downstream diagnostics use as
   the aligned x-axis offset.

Agentic feedback workflow
-------------------------

This branch is independent from the random-string prefix workflow. It is a small
smoke test for whether structured environment feedback can reopen the branching
space.

1. Build matched agentic-feedback datasets:
   `slurm/01b_build_agentic_feedback_datasets.sh`

   The three datasets share the same interaction format and differ only in the
   `Environment Feedback:` content:
   - `random_strings_agentic_feedback_control.pt`: progress update, plan remains mostly valid.
   - `random_strings_agentic_feedback_adversarial.pt`: environment changes invalidate part of
     the plan.
   - `random_strings_agentic_feedback_random_noise.pt`: same feedback slot filled by random
     text.

2. Run BF on the agentic datasets:
   `slurm/02b_run_agentic_feedback_bf_array.sh`

   Required environment variables:
   `PROJECT_DIR`, `ENV_PATH`, `MODEL`, `AGENTIC_DATA_DIR`.

Interpreting the controls
-------------------------

Start with diagnostics on existing random-string outputs. If BF decline
coincides with increasing repetition, top-1 concentration, copying, or EOS/short
continuation behavior, the random-string case likely reflects generic
autoregressive self-conditioning rather than semantic task commitment.

Then run the generation-randomness controls. If self-conditioned random prefixes
reduce BF more than iid or shuffled random prefixes, this supports the
self-conditioning explanation. If structured adversarial feedback causes a local
BF increase relative to its matched control, that provides a smoke-test example
where external context can reopen the branching space. If all noisy prefixes
reduce BF similarly, that suggests an OOD/noise-handling mechanism and should be
treated as an OOD/noise-handling limitation rather than a semantic effect.

Batch diagnostics
-----------------

Example: recursively diagnose BF files matching a model/prompt pattern:

```bash
python reviewer_mvhn_experiments/random_string_artifact_diagnostics.py \
  --artifact_pattern "/path/to/demo/response_random_strings/application_ctrlgen_multi_constraints_*/*_response_n_50_max_tokens_1024_log_probs_50_min_p_0_top_p_0.9_seed42_word_level_constraint_multiplier_15_bf.pt" \
  --output_dir reviewer_mvhn_experiments/outputs/random_string_bf_diag \
  --window_size 10 \
  --num_workers 1
```

Example: diagnose raw response dumps instead of BF files:

```bash
python reviewer_mvhn_experiments/random_string_artifact_diagnostics.py \
  --artifact_pattern "/path/to/language_modeling/response_random_strings/application_ctrlgen_multi_constraints_*/*_response_n_50_max_tokens_512_log_probs_50_min_p_0.1_top_p_0.9_seed42_word_level_constraint_multiplier_15.pt" \
  --artifact_kind raw_vllm \
  --output_dir reviewer_mvhn_experiments/outputs/random_string_raw_diag \
  --window_size 10 \
  --num_workers 1
```

Batch outputs:

- `artifact_index.csv`: one row per artifact, with parsed model, constraint,
  sample count, max tokens, top-p/min-p, seed, and status.
- `model_index.csv` and `model_index.json`: artifacts grouped by model and
  sorted by constraint level.
- `aggregate_window_summary.csv`: per-window diagnostics across all artifacts,
  with artifact metadata attached for later aggregation. This file also includes
  `prompt_offset_tokens`, `aligned_start_pos`, `aligned_end_pos`, and
  `aligned_mid_pos`. If an artifact path contains
  `model_token_prefix_tokens_<N>`, diagnostics use that exact model-token offset;
  otherwise they fall back to `constraint_level * word_level_constraint_multiplier`.
- `per_model/<model>/aligned_window_summary.csv`: model-specific aligned
  diagnostics sorted by constraint level.
- `per_model/<model>/aligned_*.png`: line plots over aligned output position for
  entropy/BF and any available token diagnostics.
- `per_artifact/.../window_summary.csv`: per-artifact window diagnostics.
- `per_artifact/.../record_summary.csv`: per-output intermediate summaries.
- `per_artifact/.../summary.json`: full metadata and error trace if loading
  fails.

Keep `--num_workers` small. Each worker loads a full artifact, so parallelism can
increase memory pressure and slow down shared filesystems.

If older files omit `word_level_constraint_multiplier`, pass
`--default_word_level_constraint_multiplier 15` to recover the random-string
offset convention used by the current scripts.

V2 stitched external-control plot
---------------------------------

This reads the v2 model-token-prefix `manifest` and the `*_bf.pt` profile in each
row's `output_root_dir`. No vLLM rerun is needed.

```bash
python reviewer_mvhn_experiments/plot_v2_external_control_stitched.py \
  --manifest reviewer_mvhn_experiments/generated_randomness_control_datasets/manifest.csv \
  --output_dir reviewer_mvhn_experiments/outputs/v2_stitched_external_control \
  --constraints 0,2,4 \
  --window_size 10 \
  --metric bf
```

The lowest-constraint curve is plotted as-is. Each higher-constraint curve copies
the previous stitched curve for `x < model_token_prefix_tokens` and then appends
the higher-constraint curve shifted by that exact model-token offset. So the `+2`
curve matches the `0` curve up to `2*multiplier`, and the `+4` curve matches the
stitched `+2` curve up to `4*multiplier`.

Published plotting data
-----------------------

The compact, path-free data used for the camera-ready figures is included:

- `stitched_plots__entropy_only/<model>/stitched_bf.csv` contains the aligned
  self-conditioned and external-random curves for each model.
- `agentic_plots/agentic_delta_from_control.csv` contains the one-turn BF shift
  from the matched control.

Re-render the figures without model inference:

```bash
python reviewer_mvhn_experiments/replot_stitched_from_csv.py
python reviewer_mvhn_experiments/plot_agentic_delta_from_control.py \
  --delta_csv reviewer_mvhn_experiments/agentic_plots/agentic_delta_from_control.csv \
  --output_dir reviewer_mvhn_experiments/agentic_plots
```

Portable cluster configuration
------------------------------

The SLURM files contain only generic resource requests. Submit them from the
repository root after creating the log directory:

```bash
mkdir -p reviewer_mvhn_experiments/slurm/logs
export ENV_PATH=/path/to/conda/environment  # optional if already activated
sbatch reviewer_mvhn_experiments/slurm/01b_build_agentic_feedback_datasets.sh
```

Set dataset/artifact paths through the variables documented at the top of each
script. Adjust GPU, memory, wall-time, account, and partition directives for your
own cluster when submitting; none are hard-coded by this repository.
