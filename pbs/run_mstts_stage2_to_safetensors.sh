#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_mstts_stage2_to_fp32
#PBS -j oe

# Consolidate mstts Stage 2 final ZeRO shards (step_18000) into a single fp32
# safetensors. This is the v0c step-count matched LaboroTV-free mstts
# candidate — the next-step evaluation will compare it against v0c.
# (step_10001_fp32 was the prior 56%-step intermediate consolidation.)

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

run_dir="output/mstts_stage2_jchat"
step="step_18000"

uv run -m tools.zero_to_fp32 \
    "$run_dir/$step" \
    "$run_dir/${step}_fp32" \
    --moshi_lm_kwargs_path init_models/mstts_init_from_mono_step8001/moshi_lm_kwargs.json

echo "DONE: $run_dir/${step}_fp32 (LaboroTV-free mstts Stage 2 final)"
