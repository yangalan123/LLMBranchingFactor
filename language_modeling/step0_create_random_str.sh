#!/bin/bash
echo $PATH
cd /path/to/your/project
conda activate ./env
cd language_modeling

python step0_random_string_generator.py | tee create_random_str_shared_llama.log
