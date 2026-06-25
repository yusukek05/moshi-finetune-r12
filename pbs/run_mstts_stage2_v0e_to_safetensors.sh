#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_mstts_stage2_v0e_to_fp32
#PBS -j oe

# Consolidate mstts v0e Stage 2 final ZeRO shards (step_18001) into a single
# fp32 safetensors. Base for Stage 3 (Zoom1 finetune). Built on the properly
# trained mono v0e (full 0178 recipe).

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

run_dir="output/mstts_stage2_jchat_v0e"
step="step_18001"

uv run -m tools.zero_to_fp32 \
    "$run_dir/$step" \
    "$run_dir/${step}_fp32" \
    --moshi_lm_kwargs_path init_models/mstts_init_from_mono_v0e/moshi_lm_kwargs.json

echo "DONE: $run_dir/${step}_fp32 (mstts v0e Stage 2 final)"
