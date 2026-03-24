#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=20:00:00
#PBS -N 0162_convert_v1.2_zoom1
#PBS -j oe

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
module load cuda/12.6/12.6.1

uv sync --python 3.12

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NO_TORCH_COMPILE=1
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

model_dir="output/v1.2_reazonspeech_jchat_zoom1"

uv run -m tools.zero_to_fp32 \
    $model_dir/step_9282 \
    $model_dir/step_9282_fp32 \
    --moshi_lm_kwargs_path init_models/moshiko-both_streams-init_text_emb-float32/moshi_lm_kwargs.json
