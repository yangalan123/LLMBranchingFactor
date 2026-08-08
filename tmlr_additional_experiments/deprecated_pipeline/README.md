Deprecated random-string BF pipeline notes
==========================================

The older 01/01a -> 02 scripts are kept in place for reproducibility, but the
recommended entry point for the reviewer random-string BF experiment is now:

* `tmlr_additional_experiments/run_random_string_bf_pipeline_v2.py`
* `tmlr_additional_experiments/slurm/run_random_string_bf_pipeline_v2.sh`

The v2 pipeline follows a simpler structure:

1. Index artifact patterns containing either `*_bf.pt` files or raw vLLM `.pt`
   dumps. For each model, choose the least non-empty constraint level
   (`constraint_level >= 1` by default) as the self-conditioned baseline and
   compute a BF-by-output-position curve.
2. Build external-randomness prompt datasets directly from the random-string
   `.pt` file using each model tokenizer and token-level offsets such as
   `+2 * multiplier` and `+4 * multiplier`.
3. Run `demo/demo.py` on those external datasets, compute BF-by-position curves,
   and plot them with the external curves shifted by their token-prefix offset.

This avoids the older manifest/legacy split and keeps the model as the Slurm
array unit.
