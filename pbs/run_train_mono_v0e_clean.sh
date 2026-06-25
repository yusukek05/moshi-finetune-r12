#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=4:ncpus=192:ngpus=8
#PBS -l walltime=72:00:00
#PBS -N 0162_train_mono_v0e_clean
#PBS -j oe

# v0e Stage 1 mono — the commercial-clean candidate AND the diagnostic that
# isolates ccaudio's effect. Identical to run_train_mono_v0d_v3.sh (full 0178
# recipe: eff bs 512 = 4 node x 8 GPU x bs16, lr 3e-4, Reazon full, 3 ep) but
# with the ccaudio_v2 shards REMOVED — only ReazonSpeech + J-CHAT-mono.
#
# Why: the v0d/v2/v3 series all ADD ccaudio and all land 63-71% CER, worse than
# the ccaudio-free "from18000" baseline (57%) and far from v0c (36%, NC). So
# ccaudio appears to HURT. This run answers two things at once:
#   1. moshika + clean-only + full recipe = the best LaboroTV-free config we can
#      build from data already on disk (the v0e baseline to then improve with
#      Emilia-YODAS / NDL-Aozora and Sidon high-freq restoration).
#   2. vs v0d_v3 (same recipe + ccaudio), the delta isolates ccaudio's effect.
#
# Downstream (extend->mstts 18000->Zoom1 2008->CER) is set up after this lands.

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
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
unset  NCCL_ASYNC_ERROR_HANDLING
unset NCCL_IB_DISABLE
ulimit -l unlimited

GPUS_PER_NODE=8
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_mono_v0e_$PBS_JOBID
NNODES=$(wc -l < hostfile_mono_v0e_$PBS_JOBID)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
echo "HOSTFILE:"; cat hostfile_mono_v0e_$PBS_JOBID
echo "WORLD_SIZE=${WORLD_SIZE}  ( ${GPUS_PER_NODE}x${NNODES} )"

DP=/groups/gcg51557/experiments/0178_dialogue_tts/data/parquet

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
      --output_dir output/mono_jchat_reazon_moshika_v0e \
      --dataset_processing_workers 32 \
      --train_data_files \
          "${DP}/J-CHAT-mono/rinna_gpt2-kyutai_mimi-q16/shard-*.parquet" \
          "${DP}/reazonspeech/rinna_gpt2-kyutai_mimi-q16/shard-*.parquet" \
      --model_dir init_models/moshika-single_streams-float32 \
      --model_dtype bfloat16 \
      --max_length 2048 \
      --num_train_epochs 3 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 16 \
      --gradient_accumulation_steps 1 \
      --text_padding_loss_weight 0.5 \
      --semantic_loss_weight 100 \
      --acoustic_loss_weight 1 \
      --parameters_to_finetune all \
      --tempformer_learning_rate 3e-4 \
      --depformer_learning_rate 3e-4 \
      --num_warmup_steps 500 \
      --weight_decay 0.1 \
      --activation_checkpointing \
      --logging_steps 1 \
      --report_to wandb \
      --project_name mono_jchat_reazon_moshika_v0e \
      --save_steps 2000 \
      --seed 42

echo "DONE. Checkpoints under output/mono_jchat_reazon_moshika_v0e/"
