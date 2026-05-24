#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=06:00:00
#PBS -N 0162_ccaudio_smoke
#PBS -j oe

# Smoke pass: ccaudio slice 0 → 1 parquet shard.
# Validates hallucination filter + Mimi/rinna tokenize + parquet shape on real
# data before launching the 100-slice array job.

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

OUT_DIR="$PWD/processed_data/ccaudio/rinna_gpt2-kyutai_mimi-q16"
mkdir -p "$OUT_DIR"

SLICE_IDX="${SLICE_IDX:-0}"

uv run python tools/ccaudio_to_parquet.py \
    --slice_idx "$SLICE_IDX" \
    --output_dir "$OUT_DIR" \
    --rep_ratio_thresh 0.05 \
    --non_ja_thresh 0.10 \
    --overwrite \
  > "$OUT_DIR/build_slice${SLICE_IDX}_${PBS_JOBID}.log" 2>&1

echo "DONE: $OUT_DIR/shard-$(printf '%05d' $SLICE_IDX).parquet"
