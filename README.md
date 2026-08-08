# LLM Branching Factor

Official code for **[LLM Probability Concentration: How Alignment Shrinks the Generative Horizon](https://openreview.net/forum?id=KotVuXj6CL)**, published in Transactions on Machine Learning Research (TMLR).

Branching Factor (BF) measures the effective number of plausible next steps during generation. The camera-ready experiments distinguish:

- **autoregressive self-narrowing**: BF usually falls as a model conditions on its growing output prefix, for base and aligned models alike;
- **alignment concentration**: alignment lowers the BF level and often steepens its early decline;
- **local reversibility**: externally sampled random prefixes and unexpected environment feedback can temporarily raise BF.

## Installation

```bash
conda create -p ./env --file requirements_conda.txt
conda activate ./env
pip install -e .
```

Optional prompt resources:

```bash
git clone https://github.com/chujiezheng/chat_templates.git
git clone https://github.com/FranxYao/chain-of-thought-hub.git
```

Paths are configured through environment variables; no source edits are required:

```bash
export BF_PROJECT_ROOT="$PWD"
export BF_CHAT_TEMPLATES_ROOT="$PWD/chat_templates"
export BF_COT_HUB_ROOT="$PWD/chain-of-thought-hub"
```

`BF_PROJECT_ROOT` defaults to the repository root. The other variables are only needed by experiments that use those external prompt collections.

## Usage

- `demo/demo.py`: end-to-end BF estimation for a new model or dataset.
- `mmlu/`, `cognac/`, `storytelling/`, `language_modeling/`: original paper experiments.
- `visualization/`: plotting utilities for the original analyses and the camera-ready post-training comparison (`plot_bf_histogram.py`).
- `reviewer_mvhn_experiments/`: camera-ready self-narrowing, random-prefix substitution, and unexpected-feedback controls, including sanitized SLURM templates and compact plotting data.

Start with [`reviewer_mvhn_experiments/README.md`](reviewer_mvhn_experiments/README.md) for the new intervention workflows. All scripts accept paths through arguments or environment variables and intentionally omit cluster accounts, partitions, usernames, and private filesystem locations.

## Citation

```bibtex
@article{yang2026alignment,
  title   = {LLM Probability Concentration: How Alignment Shrinks the Generative Horizon},
  author  = {Yang, Chenghao and Holtzman, Ari},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=KotVuXj6CL}
}
```
