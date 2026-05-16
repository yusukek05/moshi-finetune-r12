#!/bin/bash -l
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=02:00:00
#PBS -N 0162_train_v1.2_zoom1_1ep_baseline
#PBS -j oe

# Zoom1 1-epoch baseline for fair comparison against synth-phase1 (1 epoch on
# mstts-synthesised JMultiWOZ+RPC). Same base, same hyperparams, only training
# data differs.
#
#   - Base: v1.2 J-CHAT pretrain (step_8880_fp32) — identical to synthphase1
#   - Train data: processed_data/llmjp-zoom1/train-001-of-001.parquet
#                 (1,723 rows → ~108 steps at eff batch 16)
#   - 1 epoch only (Zoom1 rows split into ~1,326 optimization steps at eff
#     batch 16; ~55 min wall on 8 GPU). save_steps=500 keeps disk usage
#     bounded — each ZeRO ckpt is ~94 GB and the project quota only has
#     ~700 GB headroom (2026-05-13 incident: save_steps=25 hit quota at
#     step 325).
#
# Output: output/v1.2_reazonspeech_jchat_zoom1_1ep_baseline/
# Wandb project: v1.2_reazonspeech_jchat_zoom1_1ep_baseline

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load hpcx/2.20
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
unset NCCL_ASYNC_ERROR_HANDLING
export NCCL_IB_HCA=mlx5_0
unset NCCL_IB_DISABLE
ulimit -l unlimited

GPUS_PER_NODE=8
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_mpi_zoom1_1ep_$PBS_JOBID
NNODES=$(wc -l < hostfile_mpi_zoom1_1ep_$PBS_JOBID)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

echo "HOSTFILE:"
cat hostfile_mpi_zoom1_1ep_$PBS_JOBID
echo "WORLD_SIZE=${WORLD_SIZE} (${GPUS_PER_NODE}x${NNODES})"

mpirun \
  -n  ${WORLD_SIZE}            \
  --map-by ppr:${GPUS_PER_NODE}:node:PE=1 \
  --bind-to none \
  --oversubscribe \
  -x NCCL_NVLS_ENABLE \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING -x NCCL_IB_HCA -x NCCL_IB_DISABLE \
  -x LD_LIBRARY_PATH -x PATH \
  uv run finetune.py \
      --audio_loss_weight_when_text_pad 1.0 \
      --launcher mpi \
      --use_deepspeed \
      --tempformer_learning_rate 2e-6 \
      --depformer_learning_rate 4e-6 \
      --deepspeed_config_file ds_configs/zero3-fp16-act_ckpt.json \
      --output_dir output/v1.2_reazonspeech_jchat_zoom1_1ep_baseline \
      --train_data_files processed_data/llmjp-zoom1/train-001-of-001.parquet \
      --model_dir output/v1.2_reazonspeech_jchat/step_8880_fp32 \
      --model_dtype bfloat16 \
      --model_user_stream \
      --max_length 2048 \
      --min_length 128 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 2 \
      --num_warmup_steps 0 \
      --activation_checkpointing \
      --logging_steps 5 \
      --report_to wandb \
      --project_name v1.2_reazonspeech_jchat_zoom1_1ep_baseline \
      --save_steps 500
