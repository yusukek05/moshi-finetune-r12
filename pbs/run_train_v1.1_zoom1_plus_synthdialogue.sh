#!/bin/bash -l
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=08:00:00
#PBS -N 0162_train_v1.1_zoom1_plus_synth
#PBS -j oe

# ─────────────────────────────────────────────────────────────────
# 実験: v1.1 を「実Zoom1 + 0386合成対話(drop, はじめまして+5話題)」の混合で学習。
#   対照(baseline)  = Zoom1 単独 (pbs/run_train_v1.2_visualbank_zoom1.sh など)
#   本スクリプト(exp) = Zoom1 + 合成500本 を連結
# ⚠ 実効混合比は「行数」でなく「総フレーム/窓数」で決まる(utils/data.py の
#   split_streams が各行を max_length=2048 窓に分割するため)。
#   zoom1 = 935h ≒ 21,198窓 / 合成 = 12.4h = 500窓 → 素の連結だと合成は全体の僅か 2.3%。
#   意味ある実験には upsample 必須:  ×1≈2.3% / ×11≈20%(既定) / ×21≈33% / ×42≈50%。
#   合成は重なり無し → 上げすぎると full-duplex が劣化しうる。20%前後を推奨。
#   ⚠ --model_dir は v1.1 の Zoom1 学習を開始する base ckpt に必ず合わせること。
# ─────────────────────────────────────────────────────────────────
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
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_mpi_v1.1_zoom1_synth_$PBS_JOBID
NNODES=$(wc -l < hostfile_mpi_v1.1_zoom1_synth_$PBS_JOBID)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
echo "WORLD_SIZE=${WORLD_SIZE} ( ${GPUS_PER_NODE}x${NNODES} )"

# ── トレーニングデータ: Zoom1 + 合成(同一parquetをN回並べて upsample) ──
# 同一パスの重複は load_dataset で行数ぶん効く(実測: 3x→1500行)。
ZOOM1="processed_data/llmjp-zoom1/train-001-of-001.parquet"
SYNTH_FILE="processed_data/synth_dialogue/synth_dialogue-001-of-001.parquet"
SYNTH_UPSAMPLE="${SYNTH_UPSAMPLE:-11}"   # 11≈20% / 21≈33% / 1≈2.3%
SYNTH=""; for _ in $(seq 1 "$SYNTH_UPSAMPLE"); do SYNTH="$SYNTH $SYNTH_FILE"; done
train_data="${ZOOM1}${SYNTH}"

# ── 開始 ckpt (⚠ v1.1 の Zoom1 学習 base に合わせて書き換え) ────
MODEL_DIR="output/v1.2_reazonspeech_jchat_visualbank/step_1389_fp32"

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
      --output_dir  output/v1.1_zoom1_plus_synthdialogue \
      --train_data_files ${train_data} \
      --model_dir   ${MODEL_DIR} \
      --model_dtype bfloat16 \
      --model_user_stream \
      --max_length 2048 \
      --min_length 128 \
      --num_train_epochs 7 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 2 \
      --num_warmup_steps 0 \
      --activation_checkpointing \
      --logging_steps 10 \
      --report_to wandb \
      --project_name v1.1_zoom1_plus_synthdialogue \
      --save_steps 1000
