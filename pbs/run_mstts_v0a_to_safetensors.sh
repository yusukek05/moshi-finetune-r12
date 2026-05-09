#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=8
#PBS -l walltime=01:00:00
#PBS -N 0162_mstts_v0a_to_fp32
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
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

run_dir="output/mstts_v0a_jchat-1of39_step500"

uv run -m tools.zero_to_fp32 \
    "$run_dir/step_500" \
    "$run_dir/step_500_fp32" \
    --moshi_lm_kwargs_path init_models/moshiko-both_streams-init_text_emb-float32/moshi_lm_kwargs.json
