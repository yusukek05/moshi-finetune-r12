#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=8:ncpus=8:ngpus=8
#PBS -l walltime=120:00:00
#PBS -N 0162_train
#PBS -j oe

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

# ── modules ─────────────────────────────────────────────────────
module purge
module load cuda/12.6/12.6.1
module load hpcx/2.20
module load python/3.12/3.12.9

# ── Python venv ─────────────────────────────────────────────────
uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# ── NCCL / CUDA env ─────────────────────────────────────────────
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0 
unset  NCCL_ASYNC_ERROR_HANDLING
export NCCL_IB_HCA=mlx5_0  
unset NCCL_IB_DISABLE
ulimit -l unlimited 

# ── MPI hostfile を動的生成 ────────────────────────────────────
GPUS_PER_NODE=8
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_mpi
NNODES=$(wc -l < hostfile_mpi)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

echo "HOSTFILE:"
cat hostfile_mpi
echo "WORLD_SIZE=${WORLD_SIZE}  ( ${GPUS_PER_NODE}x${NNODES} )"

# ── トレーニングデータ ────────────────────────────────────────
train_data="processed_data/J-CHAT/podcast_train_by_espnet/podcast_train_by_espnet-*.parquet \
processed_data/J-CHAT/youtube_train_by_espnet/youtube_train_by_espnet-*.parquet"

# ── mpirun ─────────────────────────────────────────────────────
mpirun \
  -n  ${WORLD_SIZE}            \
  --map-by ppr:${GPUS_PER_NODE}:node:PE=1 \
  --bind-to none \
  --oversubscribe \
  -x NCCL_NVLS_ENABLE \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING -x NCCL_IB_HCA -x NCCL_IB_DISABLE \
  -x LD_LIBRARY_PATH -x PATH \
  uv run finetune.py \
      --launcher mpi \
      --use_deepspeed \
      --deepspeed_config_file ds_configs/zero3-fp16-warmlr-act_ckpt.json \
      --output_dir  output/moshi_p1_stage2_jchat_dialog \
      --train_data_files ${train_data} \
      --model_dir   init_models/moshiko-both_streams-float32 \
      --model_dtype bfloat16 \
      --model_user_stream \
      --max_length 2048 \
      --min_length 128 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 18 \
      --gradient_accumulation_steps 2 \
      --tempformer_learning_rate 4.5e-4 \
      --depformer_learning_rate 4.5e-4 \
      --num_warmup_steps 500 \
      --activation_checkpointing \
      --logging_steps 1 \
      --report_to wandb \
      --project_name moshi_p1_stage2_jchat_dialog \
      --save_steps 1000 \
  > 0162_train.log 2>&1
