#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=8
#PBS -l walltime=01:00:00
#PBS -N 0162_0178mstts_to_fp32
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

src_dir=/home/acg17145sv/experiments/0178_dialogue_tts/mstts-training/output/mstts-moshika-jchat-bs512-tlr3e-5-dlr1e-4-textpad0.5-warmlr500-sw100-aw1/step_17892
dst_dir=$PWD/init_models/0178_mstts-moshika-jchat_step17892_fp32
kwargs=/home/acg17145sv/experiments/0178_dialogue_tts/mstts-training/output/mono-moshika-jchat_reazon_laboro-bs512-tlr3e-4-dlr3e-4-textpad0.5-sw100-aw1/step_6000-fp32-mstts/moshi_lm_kwargs.json

uv run -m tools.zero_to_fp32 \
    "$src_dir" \
    "$dst_dir" \
    --moshi_lm_kwargs_path "$kwargs"
