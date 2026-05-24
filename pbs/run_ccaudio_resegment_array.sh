#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=10:00:00
#PBS -N 0162_ccaudio_resegment
#PBS -j oe

# Full ccaudio re-segment + re-transcribe, one slice per array task.
#   qsub pbs/run_ccaudio_resegment_array.sh            -> slice 0 only
#   qsub -J 0-99 pbs/run_ccaudio_resegment_array.sh    -> all 100 slices
# Each slice = 300 podcast episodes; faster-whisper VAD splits each into short
# speech segments, re-transcribes, filters, Mimi-q16 + rinna-SP tokenizes into
# the mono parquet schema {__key__, A_text, A_audio}.

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

SLICE="${PBS_ARRAY_INDEX:-0}"
echo "SLICE=$SLICE"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

export NO_TORCH_COMPILE=1

uv run --with faster-whisper -m tools.ccaudio_resegment_transcribe \
    --slice_idx "$SLICE" \
    --output_dir processed_data/ccaudio_v2/rinna_gpt2-kyutai_mimi-q16 \
    --overwrite

echo "DONE: processed_data/ccaudio_v2/rinna_gpt2-kyutai_mimi-q16/shard-$(printf '%05d' "$SLICE").parquet"
