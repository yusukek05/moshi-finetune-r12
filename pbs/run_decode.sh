#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=02:00:00

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
module load cuda/12.6/12.6.1

uv sync --python 3.12

export NO_TORCH_COMPILE=1

model_dir="output/moshi-finetuned_podcast_train_callhome/step_15_fp32"

uv run -m tools.decode_tokens \
    --tokens_dir "${model_dir}/continuation/generated_tokens" \
    --output_dir "${model_dir}/continuation/generated_wavs"
