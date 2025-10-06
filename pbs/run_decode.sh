#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:30:00
#PBS -j oe

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
module load cuda/12.6/12.6.1

uv sync --python 3.12

export NO_TORCH_COMPILE=1

model_dir="output/moshi_stage3_new_jchat_clean_tabidachi_csj/step_57_fp32"

uv run -m tools.decode_tokens \
    --tokens_dir "${model_dir}/continuation_jchat_full/generated_tokens" \
    --output_dir "${model_dir}/continuation_jchat_full/generated_wavs" 
