#!/bin/bash -l
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=08:00:00
#PBS -N 0162_train_mstts_stage3_zoom1_v0d
#PBS -j oe

# mstts v0d Stage 3: Zoom1 finetune on top of the LaboroTV-free, v0c
# step-count-matched Stage 2 ckpt (step_18001_fp32). This produces the final
# v0d mstts model — commercially clean (moshika base + ReazonSpeech + J-CHAT +
# ccaudio + Zoom1, all commercial-OK lineage).
#   - Base : output/mstts_stage2_jchat_v0d/step_18001_fp32
#   - Data : processed_data/llmjp-zoom1/train-001-of-001.parquet
#   - 1 node x 8 GPU, ZeRO-3, per_device 1, grad_accum 4 = eff bs 32.
#   - max_train_steps 2000 (mirrors v0c: v0b 500 + v0c 1500).

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
unset  NCCL_IB_DISABLE
ulimit -l unlimited

GPUS_PER_NODE=8
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_mstts_s3_v0d_$PBS_JOBID
NNODES=$(wc -l < hostfile_mstts_s3_v0d_$PBS_JOBID)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

RUN_NAME=mstts_stage3_zoom1_v0d
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
      --train_data_files ../processed_data/llmjp-zoom1/train-001-of-001.parquet \
      --model_dir ../output/mstts_stage2_jchat_v0d/step_18001_fp32 \
      --model_dtype bfloat16 \
      --model_user_stream \
      --max_length 2048 \
      --min_length 128 \
      --num_train_epochs 10 \
      --max_train_steps 2000 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --tempformer_learning_rate 1e-5 \
      --depformer_learning_rate 3e-5 \
      --num_warmup_steps 50 \
      --logging_steps 10 \
      --report_to wandb \
      --project_name mstts_stage3_zoom1_v0d \
      --save_steps 500 \
      --moshi_speakers A B \
      --main_speaker_bos_id 1 \
      --other_speaker_bos_id 2 \
      --process_group_timeout 7200 \
      --seed 42

echo "DONE. Checkpoints under $OUT_DIR/"
