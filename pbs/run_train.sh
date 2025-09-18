#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=8:ncpus=8:ngpus=8
#PBS -l walltime=120:00:00
#PBS -N 0162_train_new_jchat
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

# ── ランタイム最適化（CPU/ログ/キャッシュ等） ────────────────
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# 共有キャッシュ（全ランク共通で読める場所推奨）
export HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$PWD/.cache/huggingface/transformers"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# ── NCCL / CUDA env ─────────────────────────────────────────────
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1          # ← 有効化
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1                 # ← ハングを早期に検知
export NCCL_NVLS_ENABLE=0
export NCCL_IB_HCA=mlx5_0
unset  NCCL_IB_DISABLE
ulimit -l unlimited

# ── MPI hostfile を動的生成 ────────────────────────────────────
GPUS_PER_NODE=8
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_mpi_new_jchat
NNODES=$(wc -l < hostfile_mpi_new_jchat)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

echo "HOSTFILE:"
cat hostfile_mpi_new_jchat
echo "WORLD_SIZE=${WORLD_SIZE}  ( ${GPUS_PER_NODE}x${NNODES} )"

# ── トレーニングデータ ────────────────────────────────────────
train_data="processed_data/J-CHAT/podcast_train_by_espnet_lower/podcast_train_by_espnet_lower-*.parquet \
processed_data/J-CHAT/youtube_train_by_espnet_lower/youtube_train_by_espnet_lower-*.parquet"

# ── mpirun ─────────────────────────────────────────────────────
mpirun \
  -n  ${WORLD_SIZE} \
  --map-by ppr:${GPUS_PER_NODE}:node:PE=1 \
  --bind-to none \
  --oversubscribe \
  -hostfile hostfile_mpi_new_jchat \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING -x TORCH_NCCL_ASYNC_ERROR_HANDLING -x NCCL_BLOCKING_WAIT \
  -x NCCL_NVLS_ENABLE -x NCCL_IB_HCA -x NCCL_IB_DISABLE \
  -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x NUMEXPR_NUM_THREADS -x TOKENIZERS_PARALLELISM \
  -x HF_DATASETS_CACHE -x TRANSFORMERS_CACHE \
  -x PYTHONUNBUFFERED \
  -x LD_LIBRARY_PATH -x PATH \
  uv run finetune.py \
      --launcher mpi \
      --use_deepspeed \
      --deepspeed_config_file ds_configs/zero3-fp16-warmlr-act_ckpt.json \
      --output_dir  output/moshi_stage2_new_jchat \
      --train_data_files ${train_data} \
      --model_dir   init_models/moshiko-both_streams-float32 \
      --model_dtype float32 \
      --model_user_stream \
      --max_length 2048 \
      --min_length 128 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 1 \
      --num_warmup_steps 500 \
      --activation_checkpointing \
      --logging_steps 1 \
      --report_to wandb \
      --project_name moshi_stage2_new_jchat \
      --save_steps 1000 \
  > 0162_train_new_jchat.log 2>&1
