#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_mono_v0d_to_fp32
#PBS -j oe

# Consolidate mono v0d Stage 1 final ZeRO shards (step_8001) into a single fp32
# safetensors so it can serve as the base for Stage 2 (mstts training).
# v0d = moshika base + ReazonSpeech + J-CHAT + ccaudio (LaboroTV-free).

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

run_dir="output/mono_jchat_reazon_ccaudio"

uv run -m tools.zero_to_fp32 \
    "$run_dir/step_8001" \
    "$run_dir/step_8001_fp32" \
    --moshi_lm_kwargs_path init_models/moshika-single_streams-float32/moshi_lm_kwargs.json

echo "DONE: $run_dir/step_8001_fp32 (mstts v0d rebuild Stage 1 final)"
