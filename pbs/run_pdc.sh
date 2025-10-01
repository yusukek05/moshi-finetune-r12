#!/bin/bash
#PBS -P gca50130                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=4
#PBS -l walltime=01:00:00
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

model_dir="/home/acg17145sv/experiments/0215_audio_llm/moshi-finetune/output/20250929-0030+j-chat+j-chat-clean-tabidachi/step_6150_fp32"
eval_data="processed_data/J-CHAT/podcast_test_by_espnet_lower/podcast_test_by_espnet_lower-001-of-001.parquet"  

my_model_dir="output/20250929-0030+j-chat+j-chat-clean-tabidachi/step_6150_fp32"

uv run accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    generate.py \
        --output_dir "${my_model_dir}/continuation_jchat" \
        --model_dir "${model_dir}" \
        --eval_data_files "${eval_data}" \
        --prompt_length 125 \
        --generation_length 250 \
        --example_length 375 \
        --temperature 0.8
