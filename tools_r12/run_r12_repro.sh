#!/bin/bash
#PBS -q rt_HF
#PBS -N 0378_r12repro
#PBS -l select=1:ncpus=32:ngpus=8
#PBS -l walltime=9:00:00
#PBS -P gcg51557
#PBS -j oe
# 同僚 v1.1候補(lb512_epoch2_half_lr)の【プロンプト付き】完全再現。
# 同僚スクリプト run_moshi_jchat_fullsynth_zoom1_1ep_hf.sh / _continue_epoch2_hf.sh と
# 環境変数・引数まで揃える。差分はプロンプト条件づけONと、合成が detailed 修正版である点のみ。
#   STAGE=1: J-Chat step_8880 から lr 2e-6/4e-6・DATALOADER_SEED=1
#   STAGE=2: 1段目fp32(BASEDIR)から lr 1e-6/2e-6・DATALOADER_SEED=2
set -uxo pipefail
cd "$PBS_O_WORKDIR"
REPO=/groups/gcg51557/experiments/0378_spoken-dialogue-model/abe/moshi-finetune
W=/groups/gcg51557/experiments/0378_spoken-dialogue-model/personaplex_4000h
R=/groups/gcg51557/experiments/0378_spoken-dialogue-model/personaplex_role
WRAP="${WRAP:-/groups/gcg51557/experiments/0378_spoken-dialogue-model/share_r12_repro/run_finetune_with_subgroup_timeout.py}"
: "${STAGE:?STAGE=1|2}"
if [ "$STAGE" = "1" ]; then
  BASE="${BASE:?STAGE=1 には BASE(起点モデルのfp32ディレクトリ)を -v で渡す}"
  TLR=2e-6; DLR=4e-6; DSEED=1
else
  BASE="${BASEDIR:?STAGE=2 には BASEDIR(1段目のfp32)が必要}"
  TLR=1e-6; DLR=2e-6; DSEED=2
fi
OUT="${OUTDIR:?出力先OUTDIRを-vで渡す}/stage${STAGE}"
CACHE="${CACHE:-$OUTDIR/hf_cache}"
LIVE="$OUTDIR/s${STAGE}_${PBS_JOBID%%.*}.live.log"
mkdir -p "$OUT"
exec > >(tee -a "$LIVE") 2>&1
SYN=""; for i in 0 1 2 3 4 5 6 7; do SYN="$SYN $W/data/train_fix/train-00${i}-of-008.parquet"; done
# zoom1 は同僚と同じく複製せず1ファイルをそのまま渡す(ZOOM1_DUP=Nで複製も可)
Z="${ZOOM1_SRC:-$W/data/mix/z1nostyle.parquet}"
if [ "${ZOOM1_DUP:-1}" -gt 1 ]; then
  Z=""; for i in $(seq 1 "${ZOOM1_DUP}"); do Z="$Z $W/data/mix/z1c$i.parquet"; done
fi
module purge
module load cuda/12.6/12.6.1 cudnn/9.5/9.5.1 hpcx/2.20 python/3.12/3.12.9
source "${VENV:?学習用venvのパスを-vで渡す(moshi-finetuneのuv環境)}/bin/activate"
# --- 同僚と同一の分散/データローダ設定 -----------------------------------
export NO_TORCH_COMPILE=1 HF_HUB_OFFLINE=1 WANDB_MODE=offline
export NCCL_DEBUG=INFO NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export DEEPSPEED_TIMEOUT=3600 MOSHI_NCCL_SUBGROUP_TIMEOUT_SEC=3600
export MOSHI_LENGTH_BUCKET_SIZE=512 MOSHI_DATALOADER_SEED=$DSEED
unset NCCL_ASYNC_ERROR_HANDLING NCCL_IB_HCA MOSHI_BALANCE_PRIMARY_TOPICS
ulimit -l unlimited
T_START=$(date +%s); WALL_SEC=${WALL:-$((9*3600))}
RESUME=""; TRIES=0; FAST=0
while [ $TRIES -lt 10 ]; do
  TRIES=$((TRIES+1))
  LAST=$(ls -d "$OUT"/step_* 2>/dev/null | grep -v _fp32 | sed 's/.*step_//' | sort -n | tail -1)
  if [ -n "${LAST:-}" ]; then RESUME="--resume_from_checkpoint $OUT/step_$LAST"; fi
  T0=$(date +%s); LEFT=$(( WALL_SEC - (T0 - T_START) - 300 ))
  [ "$LEFT" -lt 600 ] && { echo "残り時間不足で打ち切り"; break; }
  echo "--- 試行 $TRIES / 持ち時間 ${LEFT}秒 / ${RESUME:-新規開始} ---"
  cd "$REPO"
  timeout --signal=TERM "$LEFT" mpirun -n 8 --map-by ppr:8:node:PE=1 --bind-to none --oversubscribe \
      -x NCCL_NVLS_ENABLE -x NCCL_DEBUG -x TORCH_NCCL_ASYNC_ERROR_HANDLING \
      -x TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC -x DEEPSPEED_TIMEOUT \
      -x MOSHI_NCCL_SUBGROUP_TIMEOUT_SEC -x MOSHI_LENGTH_BUCKET_SIZE -x MOSHI_DATALOADER_SEED \
      -x NCCL_IB_DISABLE -x LD_LIBRARY_PATH -x PATH -x VIRTUAL_ENV -x HF_HUB_OFFLINE \
      -x NO_TORCH_COMPILE -x WANDB_MODE \
      python "$WRAP" \
          --launcher mpi --use_deepspeed \
          --deepspeed_config_file ds_configs/zero3-fp16-act_ckpt.json \
          --output_dir "$OUT" --train_data_files $SYN $Z \
          --model_dir "$BASE" --model_dtype float16 --moshi_speakers A --model_user_stream \
          --system_prompt_conditioning --paper_prefix \
          --prefix_tokens_npz /groups/gcg51557/experiments/0378_spoken-dialogue-model/share_r12_repro/prefix_tokens.npz \
          --prefix_delimiter_id 5 --prefix_pause_frames 6 --num_main_audio 8 \
          --max_length 2048 --min_length 128 --dataset_processing_workers 16 \
          --process_group_timeout 3600 --activation_checkpointing \
          --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
          --tempformer_learning_rate $TLR --depformer_learning_rate $DLR --weight_decay 0.1 \
          --num_train_epochs 1 --num_warmup_steps 0 --seed 1 --logging_steps 10 --save_steps 1000 \
          --parameters_to_finetune all --text_padding_loss_weight 0.5 \
          --semantic_loss_weight 100.0 --acoustic_loss_weight 1.0 \
          --audio_loss_weight_when_text_pad 1.0 $RESUME \
          --report_to wandb --project_name r12repro_s${STAGE}
  RC=$?
  cd "$PBS_O_WORKDIR"
  echo "--- 試行 $TRIES rc=$RC (経過 $(( $(date +%s) - T0 ))秒) ---"
  [ "$RC" -eq 0 ] && { echo "=== 正常終了 ==="; break; }
  [ "$RC" -eq 124 ] && { echo "=== walltime到達で打ち切り ==="; break; }
  if [ $(( $(date +%s) - T0 )) -lt 300 ]; then
    FAST=$((FAST+1)); grep -E "Missmatch|Error|Traceback|Errno" "$LIVE" | tail -3
    [ "$FAST" -ge 2 ] && { echo "=== 即死が2回続いたので中止 ==="; break; }
  else FAST=0; fi
done
echo "=== 最終CK ==="; ls -d "$OUT"/step_* 2>/dev/null | grep -v _fp32 | sed 's/.*step_//' | sort -n | tr '\n' ' '; echo
echo "=== R1P STAGE${STAGE} DONE ==="
