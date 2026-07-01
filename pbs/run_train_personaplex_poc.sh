#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=03:00:00
#PBS -N 0162_train_personaplex_poc
#PBS -j oe

# PersonaPlex 型 prompt-control PoC (v0):
#   v1.1 (= Reazon+J-CHAT -> Zoom1, step_9282_fp32) を base に、
#   persona-conditioned parquet (prompt_text_ids + 任意 prompt_audio) で
#   --system_prompt_conditioning 学習。prompt 区間は loss マスク(=zero_token_id)。
# 設計: docs/personaplex_poc_design.md。学習ループ本体は無改修(機構は前処理のみ)。
# レシピは v1.1/poststage を忠実ミラー(tlr2e-6/dlr4e-6/max_len2048/eff bs16)。
# walltime kill 時は再投入で自動再開。
#
# 注意(2026-06-29 実測): 既存コーパス(Zoom1/FireRed)は丁寧体にほぼ偏り。text-style 制御の
#   実証には causal 生成 (persona sample -> LLM-jp-3 -> FireRed) の persona parquet が要る。
#   TRAIN_DATA を causal 版に差し替えること。bootstrap parquet は plumbing smoke 用。

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
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_mpi_personaplex_poc_$PBS_JOBID
NNODES=$(wc -l < hostfile_mpi_personaplex_poc_$PBS_JOBID)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
echo "WORLD_SIZE=${WORLD_SIZE}"

OUT="${OUT:-output/personaplex_poc}"
MODEL_DIR="output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32"   # = v1.1 (final, dep_q=16)
# persona-conditioned parquet (prompt_text_ids [+ prompt_audio]).
# causal 版に差し替え推奨; bootstrap smoke は persona_poc/persona_synth*.parquet
TRAIN_DATA="${TRAIN_DATA:-processed_data/persona_poc/persona_train-001-of-001.parquet}"

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
      --system_prompt_conditioning \
      --num_main_audio 8 \
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
      --num_train_epochs "${EPOCHS:-3}" \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 2 \
      --num_warmup_steps 0 \
      --activation_checkpointing \
      --logging_steps 5 \
      --report_to wandb \
      --project_name personaplex_poc \
      --save_steps 50 \
      "${RESUME_ARGS[@]}"
