#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N 0162_mstts_v0a_decode
#PBS -j oe

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

INFER_DIR="$PWD/output/mstts_v0a_jchat-1of39_step500/inference_step500"
mkdir -p "$INFER_DIR/decoded_audio"

uv run -m tools.decode_tokens \
    --tokens_dir "$INFER_DIR/generated_tokens" \
    --output_dir "$INFER_DIR/decoded_audio" \
    --num_workers 1 \
  > "$INFER_DIR/decode_$PBS_JOBID.log" 2>&1
