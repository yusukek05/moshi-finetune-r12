#!/bin/bash -l
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=24:00:00
#PBS -N 0162_train_mono_v0d
#PBS -j oe

# v0d Stage 1 mono: moshika base + ReazonSpeech + J-CHAT-mono + ccaudio.
# Differences from run_train_mono.sh:
#   - base : init_models/moshika-single_streams-float32 (was moshiko)
#   - data : adds processed_data/ccaudio/rinna_gpt2-kyutai_mimi-q16/shard-*.parquet
#   - output_dir : output/mono_jchat_reazon_ccaudio (separate from v1.1 line)
# Identical hyper-params otherwise (eff batch 16, tlr=dlr=1e-4, 8000 steps).

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

export NO_TORCH_COMPILE=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
# 2h watchdog: rank 0 preprocesses the new ccaudio shards on first run;
# subsequent runs hit the HF datasets cache.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
unset  NCCL_ASYNC_ERROR_HANDLING
unset NCCL_IB_DISABLE
ulimit -l unlimited

GPUS_PER_NODE=8
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_mono_v0d_$PBS_JOBID
NNODES=$(wc -l < hostfile_mono_v0d_$PBS_JOBID)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
echo "HOSTFILE:"; cat hostfile_mono_v0d_$PBS_JOBID
echo "WORLD_SIZE=${WORLD_SIZE}  ( ${GPUS_PER_NODE}x${NNODES} )"

DP=/groups/gcg51557/experiments/0178_dialogue_tts/data/parquet
CC=processed_data/ccaudio/rinna_gpt2-kyutai_mimi-q16

mpirun \
  -n  ${WORLD_SIZE} \
  --map-by ppr:${GPUS_PER_NODE}:node:PE=1 \
  --bind-to none \
  --oversubscribe \
  -x NCCL_NVLS_ENABLE -x NO_TORCH_COMPILE -x TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING -x NCCL_IB_DISABLE \
  -x LD_LIBRARY_PATH -x PATH \
  uv run finetune_mono_text.py \
      --launcher mpi \
      --use_deepspeed \
      --deepspeed_config_file ds_configs/zero3-bf16-warmlr-act_ckpt.json \
      --output_dir output/mono_jchat_reazon_ccaudio \
      --dataset_processing_workers 32 \
      --train_data_files \
          "${DP}/J-CHAT-mono/rinna_gpt2-kyutai_mimi-q16/shard-*.parquet" \
          "${DP}/reazonspeech/rinna_gpt2-kyutai_mimi-q16/shard-000[0-2][0-9].parquet" \
          "${CC}/shard-*.parquet" \
      --model_dir init_models/moshika-single_streams-float32 \
      --model_dtype bfloat16 \
      --max_length 2048 \
      --num_train_epochs 2 \
      --max_train_steps 8000 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 2 \
      --audio_loss_weight_when_text_pad 1.0 \
      --text_padding_loss_weight 0.5 \
      --semantic_loss_weight 100 \
      --acoustic_loss_weight 1 \
      --parameters_to_finetune all \
      --tempformer_learning_rate 1e-4 \
      --depformer_learning_rate 1e-4 \
      --num_warmup_steps 500 \
      --weight_decay 0.1 \
      --activation_checkpointing \
      --logging_steps 10 \
      --report_to wandb \
      --project_name mono_jchat_reazon_ccaudio_v0d \
      --save_steps 1000 \
      --seed 42

echo "DONE. Checkpoints under output/mono_jchat_reazon_ccaudio/"
