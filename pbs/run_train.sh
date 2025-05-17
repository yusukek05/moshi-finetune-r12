#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8

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

train_data_files="processed_data/spokenwoz_sample/train-*.parquet"

uv run accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/zero3-fp16-warmlr-act_ckpt.json \
    finetune.py \
        --launcher accelerate \
        --output_dir output/moshiko-finetuned \
        --train_data_files "${train_data_files}" \
        --model_dir init_models/moshiko-both_streams-float32 \
        --model_dtype float32 \
        --model_user_stream \
        --max_length 2048 \
        --min_length 128 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --num_warmup_steps 500 \
        --activation_checkpointing \
        --logging_steps 1 \
        --report_to wandb \
        --project_name moshi-finetuning \
        --save_steps 1000
