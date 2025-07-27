#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
module load cuda/12.6/12.6.1

uv sync --python 3.12

export NO_TORCH_COMPILE=1
export ACCELERATE_DISTRIBUTED_TYPE=gloo
export PYTORCH_USE_RDMA=0
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

model_dir="output/moshi-finetuned_train_ohashi/step_8880_fp32"
eval_data="processed_data/J-CHAT/podcast_train/podcast_train-001-of-039.parquet"  

uv run accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    generate.py \
        --output_dir "${model_dir}/continuation_jchat" \
        --model_dir "${model_dir}" \
        --eval_data_files "${eval_data}" \
        --prompt_length 125 \
        --generation_length 250 \
        --example_length 375 \
        --temperature 0.8
