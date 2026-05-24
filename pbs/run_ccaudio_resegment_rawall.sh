#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=04:00:00
#PBS -N 0162_ccaudio_rawall
#PBS -j oe

# Re-segment + re-transcribe the FULL ccaudio raw_all corpus (~23,685h).
# One recording.*.tar (~100 podcast episodes) per array task.
#   qsub -v IDX=0 pbs/run_ccaudio_resegment_rawall.sh   -> single tar (smoke)
#   qsub -J 0-593 pbs/run_ccaudio_resegment_rawall.sh   -> all 594 tars
# Task index -> line (IDX+1) of tar_manifest.txt -> that tar -> shard-IDX.parquet

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

IDX="${PBS_ARRAY_INDEX:-${IDX:-0}}"
MANIFEST=processed_data/ccaudio_v2_full/tar_manifest.txt
TAR_PATH=$(sed -n "$((IDX + 1))p" "$MANIFEST")
echo "IDX=$IDX TAR_PATH=$TAR_PATH"
[ -n "$TAR_PATH" ] || { echo "no tar for IDX=$IDX"; exit 1; }

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

export NO_TORCH_COMPILE=1

uv run --with faster-whisper -m tools.ccaudio_resegment_transcribe \
    --tar_path "$TAR_PATH" \
    --shard_id "$IDX" \
    --output_dir processed_data/ccaudio_v2_full/rinna_gpt2-kyutai_mimi-q16 \
    --overwrite

echo "DONE: processed_data/ccaudio_v2_full/rinna_gpt2-kyutai_mimi-q16/shard-$(printf '%05d' "$IDX").parquet"
