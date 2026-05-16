#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_v11_to_fp32
#PBS -j oe

# Consolidate v1.1 final stage (VisualBank -> Zoom1 7ep, step_9282) into a
# single fp32 safetensors so it can be evaluated / uploaded to HF.

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

run_dir="output/v1.2_reazonspeech_jchat_visualbank_zoom1"

uv run -m tools.zero_to_fp32 \
    "$run_dir/step_9282" \
    "$run_dir/step_9282_fp32" \
    --moshi_lm_kwargs_path output/v1.2_reazonspeech_jchat/step_8880_fp32/moshi_lm_kwargs.json

echo "DONE: $run_dir/step_9282_fp32 (v1.1 candidate release)"
