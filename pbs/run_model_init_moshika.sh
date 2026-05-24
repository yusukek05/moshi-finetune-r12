#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N 0162_init_moshika_single
#PBS -j oe

# v0d Stage 1 base: moshika single-stream float32 init (mirrors the
# moshiko-single_streams-float32 recipe in run_model_init_system_only.sh,
# only the source repo differs).

set -euxo pipefail
echo "JOB_ID: $PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

uv run -m tools.init_moshi_for_ft \
    --moshi_lm_repo kyutai/moshika-pytorch-bf16 \
    --save_dir init_models/moshika-single_streams-float32 \
    --model_dtype float32 \
  > model_init_moshika_$PBS_JOBID.log 2>&1

echo "DONE: init_models/moshika-single_streams-float32"
