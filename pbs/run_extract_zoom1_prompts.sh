#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:10:00
#PBS -N 0162_zoom1_prompts
#PBS -j oe

set -euxo pipefail
cd "$PBS_O_WORKDIR"
module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export NO_TORCH_COMPILE=1

PAPER=/home/acg17145sv/projects/icml-mlforaudio-2026
uv run python $PAPER/scripts/extract_zoom1_prompts.py \
    --parquet processed_data/llmjp-zoom1/test-001-of-001.parquet \
    --out_dir  $PAPER/docs/wavs/zoom1_reference \
    --targets 0:zoom1_ref_0 20:zoom1_ref_20
