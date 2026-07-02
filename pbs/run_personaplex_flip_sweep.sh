#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=03:00:00
#PBS -N 0162_personaplex_flip_sweep
#PBS -j oe
#PBS -o logs/
#
# Sweep flip eval across training checkpoints to find the best step (peak held-out
# flip before overfit). Consolidates each step -> fp32, evals on held-out, then
# DELETES the fp32 dir to bound disk (~30GB each).
#
# Submit (spot):
#   qsub -q rt_HF -v RTYPE=rt_HF,OUT=output/personaplex_poc_full,\
#        HELDOUT=processed_data/persona_poc/persona_full_heldout-001-of-001.parquet,\
#        STEPS="step_50 step_100 step_150 step_200 step_250" pbs/run_personaplex_flip_sweep.sh
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

OUT="${OUT:-output/personaplex_poc_full}"
HELDOUT="${HELDOUT:-processed_data/persona_poc/persona_full_heldout-001-of-001.parquet}"
STEPS="${STEPS:-step_50 step_100 step_150 step_200 step_250}"
PARAPHRASE_IDX="${PARAPHRASE_IDX:--1}"   # >=0 to eval with a held-out paraphrase
BASE_KWARGS="${BASE_KWARGS:-output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32/moshi_lm_kwargs.json}"
RINNA=/home/acg17145sv/.cache/huggingface/hub/models--rinna--japanese-gpt2-medium/snapshots/8ce2399c33e99013a593ea9389378fd86662b9c7/spiece.model

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
export NO_TORCH_COMPILE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

for STEP in $STEPS; do
  [ -d "$OUT/$STEP" ] || { echo "SKIP missing $OUT/$STEP"; continue; }
  FP32="$OUT/${STEP}_fp32"
  echo "############ SWEEP $STEP ############"
  if [ ! -f "$FP32/model.safetensors" ]; then
    uv run -m tools.zero_to_fp32 "$OUT/$STEP" "$FP32" --moshi_lm_kwargs_path "$BASE_KWARGS"
  fi
  echo ">>> flip eval @ $STEP"
  uv run python mstts/data_prep/persona_flip_eval.py \
      --model_dir "$FP32" --heldout "$HELDOUT" --tokenizer "$RINNA" \
      --paraphrase-idx "$PARAPHRASE_IDX" || echo "eval failed @ $STEP"
  # free disk: drop the consolidated fp32 (keep the raw DeepSpeed step for re-consolidation if needed)
  rm -rf "$FP32"
  echo "############ END $STEP ############"
done
echo "=== flip sweep done ==="
