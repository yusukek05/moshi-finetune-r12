#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_cascade_to_v1_parquet_smoke
#PBS -j oe

# Smoke test: convert 5 existing ICASSP Phase 1 cascade dialogues into v1 parquet
# format. Validates the pipeline (Mimi tokenize + text alignment from turns.json)
# before scaling up to the 47K-dialogue full cascade synth corpus.

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

CASCADE_SRC=/home/acg17145sv/projects/icassp-2027-mstts/dialogues/cascade_output
OUT_DIR=output/cascade_to_v1_parquet_smoke
mkdir -p "$OUT_DIR"

uv run python -m mstts.data_prep.cascade_output_to_v1_parquet \
    --cascade-dir "$CASCADE_SRC" \
    --output-prefix "$OUT_DIR/synth" \
    --num-examples-per-parquet 10 \
    --limit 5

echo "DONE: $OUT_DIR"
ls -la "$OUT_DIR"
