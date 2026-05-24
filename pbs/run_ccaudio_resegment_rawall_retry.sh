#!/bin/bash -l
#PBS -P gca50130
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N 0162_ccaudio_rawall_retry
#PBS -j oe

# Retry tar shards that 1791625 array tasks killed at the 4h walltime.
# Walltime 12h on rt_HG (1 GPU + 16 CPU, 168h walltime cap), so single-tar
# long-tail jobs always finish on the first try. Was R9920261000/rt_HF which
# wasted 7/8 of the H200 node per task.
#
# Project: gca50130 (not gcg51557) because gcg51557 is out of points and
# only its R9920261000 reserved queue is free; rt_HG needs a budgeted project.
#
# RETRY_LIST is one-idx-per-line at pbs/ccaudio_rawall_retry_idx.txt so we
# can keep appending newly-discovered walltime kills without editing this
# script. PBS_ARRAY_INDEX is 1-based (ABCI PBS rejects 0-indexed arrays)
# and is used directly as the line number in that file. The resolved IDX
# is fed to --shard_id, so shard-NNNNN.parquet keeps its original numbering.
#
#   qsub -J 1-$(wc -l < pbs/ccaudio_rawall_retry_idx.txt) \
#        pbs/run_ccaudio_resegment_rawall_retry.sh

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# RETRY_LIST can be overridden at submit time, e.g.:
#   qsub -v RETRY_LIST=pbs/ccaudio_rawall_retry_idx_held.txt -J 1-15 ...
RETRY_LIST="${RETRY_LIST:-pbs/ccaudio_rawall_retry_idx.txt}"
ARR_IDX="${PBS_ARRAY_INDEX:-1}"
IDX=$(sed -n "${ARR_IDX}p" "$RETRY_LIST")
[ -n "$IDX" ] || { echo "no retry idx at line $ARR_IDX of $RETRY_LIST"; exit 1; }
MANIFEST=processed_data/ccaudio_v2_full/tar_manifest.txt
TAR_PATH=$(sed -n "$((IDX + 1))p" "$MANIFEST")
echo "ARR_IDX=$ARR_IDX -> IDX=$IDX  TAR_PATH=$TAR_PATH"
[ -n "$TAR_PATH" ] || { echo "no tar for IDX=$IDX"; exit 1; }

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Force line-buffered stdout so the .o log shows per-episode progress and we
# can diagnose if a retry still walltimes. (Block buffering was why the
# original walltime kills had blank logs.)
export PYTHONUNBUFFERED=1

export NO_TORCH_COMPILE=1

uv run --with faster-whisper -m tools.ccaudio_resegment_transcribe \
    --tar_path "$TAR_PATH" \
    --shard_id "$IDX" \
    --output_dir processed_data/ccaudio_v2_full/rinna_gpt2-kyutai_mimi-q16 \
    --overwrite

echo "DONE: processed_data/ccaudio_v2_full/rinna_gpt2-kyutai_mimi-q16/shard-$(printf '%05d' "$IDX").parquet"
