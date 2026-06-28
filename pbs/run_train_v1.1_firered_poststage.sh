#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=03:00:00
#PBS -N 0162_train_v1.1_fr_poststage
#PBS -j oe

# Spoken Dialogue Arena 向けドメイン適応:
#   v1.1 (= Reazon+J-CHAT -> Zoom1, step_9282_fp32) を base に、
#   FireRedTTS-2 合成(雑談対話 500件/12.4h) で "最終段" を軽く当てる。
# 狙い: Zoom1-CER ではなく Arena (60-120s 全二重雑談, FD-DMOS 人手) でのドメイン適応。
# 軽め (3 epoch, lr 2e-6/4e-6) で base 品質を壊さずに雑談スタイルへ寄せる。
# walltime kill 時は再投入で自動再開。

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
unset  NCCL_ASYNC_ERROR_HANDLING
export NCCL_IB_HCA=mlx5_0
unset NCCL_IB_DISABLE
ulimit -l unlimited

GPUS_PER_NODE=8
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_mpi_v1.1_fr_poststage_$PBS_JOBID
NNODES=$(wc -l < hostfile_mpi_v1.1_fr_poststage_$PBS_JOBID)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
echo "WORLD_SIZE=${WORLD_SIZE}"

OUT=output/v1.1_firered_poststage
MODEL_DIR="output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32"   # = v1.1 (final, dep_q=16)
TRAIN_DATA="processed_data/synth_dialogue/synth_dialogue-001-of-001.parquet"

# auto-resume from latest raw step_N if present
RESUME_ARGS=()
LATEST=$(ls -d "$OUT"/step_* 2>/dev/null | grep -oE 'step_[0-9]+$' | sort -t_ -k2 -n | tail -1 || true)
if [ -n "$LATEST" ] && [ -f "$OUT/$LATEST/latest" ]; then
    echo "Resuming from $OUT/$LATEST"
    RESUME_ARGS=(--resume_from_checkpoint "$OUT/$LATEST")
fi

mpirun \
  -n  ${WORLD_SIZE} \
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
      --output_dir  "$OUT" \
      --train_data_files "$TRAIN_DATA" \
      --model_dir   "$MODEL_DIR" \
      --model_dtype bfloat16 \
      --model_user_stream \
      --max_length 2048 \
      --min_length 128 \
      --num_train_epochs 3 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 2 \
      --num_warmup_steps 0 \
      --activation_checkpointing \
      --logging_steps 5 \
      --report_to wandb \
      --project_name v1.1_firered_poststage \
      --save_steps 30 \
      "${RESUME_ARGS[@]}"
