#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_mstts_s3_from18k_to_fp32
#PBS -j oe

# Consolidate mstts Stage 3 (from18000 lineage) final ZeRO shards into fp32
# safetensors. Stage 2 base was step_18000_fp32 (matches v0c 17,892 step count),
# Stage 3 is 2008 Zoom1 steps on top.

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

run_dir="output/mstts_stage3_zoom1_from18000"
step="step_2008"

uv run -m tools.zero_to_fp32 \
    "$run_dir/$step" \
    "$run_dir/${step}_fp32" \
    --moshi_lm_kwargs_path init_models/mstts_init_from_mono_step8001/moshi_lm_kwargs.json

echo "DONE: $run_dir/${step}_fp32 (Stage 3 from step_18000, v0d-match lineage)"
