#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=4
#PBS -l walltime=02:00:00

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
module load cuda/12.6/12.6.1

uv sync --python 3.12

export NO_TORCH_COMPILE=1

model_dir="output/moshiko-finetuned/step_1_fp32"
uv run accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    generate.py \
        --output_dir "${model_dir}/continuation" \
        --model_dir "${model_dir}" \
        --eval_data_files "processed_data/spokenwoz_sample/train-*.parquet" \
        --prompt_length 125 \
        --generation_length 250 \
        --temperature 0.8 \
        