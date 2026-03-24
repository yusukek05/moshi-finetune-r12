#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=20:00:00
#PBS -N 0162_clean_v1.2
#PBS -j oe

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
module load cuda/12.6/12.6.1

uv sync --python 3.12

export NO_TORCH_COMPILE=1

uv run -m tools.clean_moshi \
    --moshi_ft_dir output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32 \
    --save_dir output/v1.2_reazonspeech_jchat_zoom1/step_9282_cleaned \
    --model_dtype bfloat16 \
    --remove_modules_for_user_stream
