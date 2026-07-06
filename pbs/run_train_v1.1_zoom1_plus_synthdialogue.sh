#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=12:00:00
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
# SYNTH_FILE env-overridable. NOTE: the 100h diverse corpus (~3590 dialogues ≈ 3600 windows)
# is already ~14.5% of zoom1 (21,198 windows) at upsample=1, so DON'T upsample it — instead
# point SYNTH_FILE at a pre-subsampled parquet (…_5pct=1118 / …_10pct=2364 dialogues) built by
# stratified nested sampling, and keep SYNTH_UPSAMPLE=1. (Old 500-dialogue corpus needed ×11.)
SYNTH_FILE="${SYNTH_FILE:-processed_data/synth_dialogue/synth_dialogue-001-of-001.parquet}"
SYNTH_UPSAMPLE="${SYNTH_UPSAMPLE:-11}"   # old-corpus default; for 100h pass SYNTH_UPSAMPLE=1
SYNTH=""; for _ in $(seq 1 "$SYNTH_UPSAMPLE"); do SYNTH="$SYNTH $SYNTH_FILE"; done
train_data="${ZOOM1}${SYNTH}"
# 比率別に出力先を分ける (×3/×5 スイープが衝突しないように)
OUT="${OUT:-output/v1.1_zoom1_plus_synth_x${SYNTH_UPSAMPLE}}"

# ── 開始 ckpt = v1.1 baseline と同一 (ReazonSpeech→J-CHAT 後, VB なし) ──
#   参照 pbs/run_train_v1.2_zoom1.sh と同じ base。これで
#     baseline = Zoom1 単独 (= 既評価の v1.1候補, 自己整合CER 0.62)
#     exp      = Zoom1 + FireRed合成
#   が同一 base からの apples-to-apples になる。
#   ⚠ VB入り (_visualbank/step_1389) は lineage評価で CER 0.62→0.91 と悪化要因なので使わない。
MODEL_DIR="output/v1.2_reazonspeech_jchat/step_8880_fp32"

# ── walltime kill 対策: 最新 raw step_N があれば自動再開 (再投入だけで継続) ──
RESUME_ARGS=()
LATEST=$(ls -d "$OUT"/step_* 2>/dev/null | grep -E 'step_[0-9]+$' | sort -t_ -k2 -n | tail -1 || true)
if [ -n "$LATEST" ] && [ -f "$LATEST/latest" ]; then
    echo "Resuming from $LATEST"
    RESUME_ARGS=(--resume_from_checkpoint "$LATEST")
fi

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
      --output_dir  "$OUT" \
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
      --project_name "$(basename "$OUT")" \
      --save_steps 1000 \
      "${RESUME_ARGS[@]}"
