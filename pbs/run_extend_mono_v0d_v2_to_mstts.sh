#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_extend_mono_v0d_v2_to_mstts
#PBS -j oe

# v0d_v2 Stage 1 -> Stage 2 bridge: extend the mono-trained depth transformer
# for the user stream and save as the init for Stage 2 (mstts) training.
#   mono v0d_v2 = moshika base + ReazonSpeech + J-CHAT + ccaudio_v2_full
#                 (clean filter, 587 shard / 2,462 h, LaboroTV-free,
#                  commercial-clean).

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

uv run python -m tools.extend_mono_to_mstts \
    --mono_dir output/mono_jchat_reazon_ccaudio_v2/step_8001_fp32 \
    --save_dir init_models/mstts_init_from_mono_v0d_v2_step8001 \
    --model_dtype bfloat16

echo "DONE: init_models/mstts_init_from_mono_v0d_v2_step8001 (mstts v0d_v2 Stage 2 base)"
