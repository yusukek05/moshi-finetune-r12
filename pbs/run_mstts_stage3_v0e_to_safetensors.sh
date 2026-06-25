#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_mstts_s3_v0e_to_fp32
#PBS -j oe

# Consolidate mstts v0e Stage 3 (Zoom1 FT) final ZeRO shards into fp32
# safetensors. FINAL v0e mstts model — LaboroTV-free, commercial-clean,
# built on the full-0178-recipe mono v0e.

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

run_dir="output/mstts_stage3_zoom1_v0e"
step="step_2008"

uv run -m tools.zero_to_fp32 \
    "$run_dir/$step" \
    "$run_dir/${step}_fp32" \
    --moshi_lm_kwargs_path init_models/mstts_init_from_mono_v0e/moshi_lm_kwargs.json

echo "DONE: $run_dir/${step}_fp32 (FINAL v0e mstts — full 0178 recipe, LaboroTV-free)"
