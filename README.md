# LLMBranchingFactor

Codebase for the paper "How Alignment Shrinks the Generative Horizon"

Author: Chenghao Yang (yangalan1996@gmail.com)

## Dependencies

Create a conda environment, and then install the dependencies from `requirements_conda.txt` and install our package in editable mode:
```bash
conda create -p ./env --file requirements_conda.txt
conda activate ./env
pip install -e .
```

Download chat_templastes used to set up aligned model prompting (for backward compatibility to older version of HF):
```bash
git clone https://github.com/chujiezheng/chat_templates.git
```

Update: we recently find CoT hub provides performance-verified prompt templates for various tasks, which are more reliable than the ones in `chat_templates`. 
We recommend using them instead. That repo is released under MIT license. We have adjusted part of our codes to accomodate it.  

Download prompt template files from Chain-of-Thoughts hub:
```bash
git clone https://github.com/FranxYao/chain-of-thought-hub.git
```
But in general, you can put any prompt template files in the `prompt_templates` folder under `root_path` (see instruction below to set it up), and the code will automatically load them.

## Usage
1. Setup Environment Variable

    In order to run the code, you need to set the environment variable:

    In `uncertainty_quantification/consts.py`, set `root_path` to be the absolute path to the current project directory.

    Also change all `/path/to/project` in all shell scripts to run the codes.

2. Run shell scripts under `mmlu`, `cognac`, `storytelling`, `language_modeling` to replicate experiments in the paper. `stepX_xxx.sh` specify the ordering to run the scripts. `visualization` provide codes to create figures in the paper.

3. [For Application to New Dataset and New Models] Take a look at `demo/demo.py`. Owing to significant improvement of vLLM memory management, using my columnar IO storage, you can now using a single file to run the whole pipeline.

## Reference
If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):

```
@inproceedings{yang-2025-branching,
  author =      {Chenghao Yang and Ari Holtzman},
  title =       {How Alignment Shrinks the Generative Horizon},
  booktitle =   {Arxiv},
  year =        {2025}
}
```
