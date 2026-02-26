#!/bin/bash

template_file=$1
example_file=$2

python3 kushal_sync_logic.py ${template_file} ${example_file}
./benchmark.sh glm_asr_triton_example
git checkout ${example_file}
echo "✨ Workspace restored to clean state."