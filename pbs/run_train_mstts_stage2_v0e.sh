#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=24:00:00
#PBS -N 0162_train_mstts_stage2_v0e
#PBS -j oe

# mstts v0e Stage 2: train multi-stream dialogue TTS on J-CHAT (commercial-clean)
# starting from the properly-trained mono v0e base (Stage 1, full 0178 recipe).
#   - Base : init_models/mstts_init_from_mono_v0e
#           (moshika + Reazon full + J-CHAT + ccaudio_v2_full, eff bs 512, 3 ep)
#   - Data : 0178 data-jmoshi/jchat-podcast ({A,B} format)
#   - 1 node x 8 GPU, ZeRO-3, per_device 1, grad_accum 2 = eff bs 16.
#   - max_train_steps 18000 (match v0c's mstts pretrain count); save_steps 1000.

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
export DEEPSPEED_TIMEOUT=120
unset  NCCL_ASYNC_ERROR_HANDLING
unset NCCL_IB_DISABLE
ulimit -l unlimited

GPUS_PER_NODE=8
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_mstts_s2_v0e_$PBS_JOBID
NNODES=$(wc -l < hostfile_mstts_s2_v0e_$PBS_JOBID)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

DJ=/groups/gcg51557/experiments/0178_dialogue_tts/data-jmoshi
RUN_NAME=mstts_stage2_jchat_v0e
OUT_DIR="$PWD/output/$RUN_NAME"
mkdir -p "$OUT_DIR"

cd mstts
mpirun \
  -n  ${WORLD_SIZE} \
  --map-by ppr:${GPUS_PER_NODE}:node:PE=1 \
  --bind-to none \
  --oversubscribe \
  -x NCCL_NVLS_ENABLE -x NO_TORCH_COMPILE -x TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC \
  -x DEEPSPEED_TIMEOUT \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING -x NCCL_IB_DISABLE \
  -x LD_LIBRARY_PATH -x PATH \
  uv run python finetune_ms_tts.py \
      --launcher mpi \
      --use_deepspeed \
      --deepspeed_config_file ../ds_configs/zero3-bf16-warmlr-act_ckpt.json \
      --activation_checkpointing \
      --output_dir "$OUT_DIR" \
      --dataset_processing_workers 32 \
      --train_data_files "${DJ}/jchat-podcast/train-*.parquet" \
      --model_dir ../init_models/mstts_init_from_mono_v0e \
      --model_dtype bfloat16 \
      --model_user_stream \
      --max_length 2048 \
      --min_length 128 \
      --num_train_epochs 2 \
      --max_train_steps 18000 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 2 \
      --tempformer_learning_rate 1e-4 \
      --depformer_learning_rate 1e-4 \
      --num_warmup_steps 500 \
      --weight_decay 0.1 \
      --logging_steps 10 \
      --report_to wandb \
      --project_name mstts_stage2_jchat_v0e \
      --save_steps 1000 \
      --moshi_speakers A B \
      --main_speaker_bos_id 1 \
      --other_speaker_bos_id 2 \
      --process_group_timeout 7200 \
      --seed 42

echo "DONE. Checkpoints under $OUT_DIR/"
