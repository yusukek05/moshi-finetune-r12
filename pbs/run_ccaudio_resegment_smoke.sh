#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N 0162_ccaudio_resegment_smoke
#PBS -j oe

# Smoke test for the ccaudio re-segment + re-transcribe pipeline.
# Processes the first 10 episodes of slice 0: faster-whisper VAD splits each
# episode into short speech segments, re-transcribes, filters, tokenizes.
# Inspect shard-00000.transcripts.jsonl afterwards to judge transcript quality
# before committing to a full slice / all 100 slices.

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

uv run --with faster-whisper -m tools.ccaudio_resegment_transcribe \
    --slice_idx 0 \
    --output_dir processed_data/ccaudio_v2/rinna_gpt2-kyutai_mimi-q16 \
    --max_episodes 10 \
    --overwrite

echo "DONE: processed_data/ccaudio_v2/rinna_gpt2-kyutai_mimi-q16/shard-00000.parquet"
