#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_v1plusv0csynth_cleanbf16
#PBS -j oe

# Phase 2.1 post-train: clean_moshi with --model_dtype bfloat16 to make a
# moshi.server-ready bundle (~15 GB, matches the public llm-jp-moshi-v1
# format). Source is the already-consolidated step_2757_fp32.
#
# Output: output/v1_plus_v0csynth/step_2757_cleaned_bf16/ (~15 GB bf16, dep_q=8)

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export NO_TORCH_COMPILE=1

uv run -m tools.clean_moshi \
    --moshi_ft_dir output/v1_plus_v0csynth/step_2757_fp32 \
    --save_dir     output/v1_plus_v0csynth/step_2757_cleaned_bf16 \
    --model_dtype  bfloat16 \
    --remove_modules_for_user_stream

ls -la output/v1_plus_v0csynth/step_2757_cleaned_bf16/
echo "DONE"
