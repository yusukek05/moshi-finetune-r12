#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=02:00:00
#PBS -N 0162_personaplex_flip_eval
#PBS -j oe
#PBS -o logs/
#
# PersonaPlex flip verification (Task #131):
#   1) consolidate the trained DeepSpeed ckpt -> fp32 (zero_to_fp32)
#   2) teacher-forced conditional NLL flip on held-out personas (matched vs swapped prompt)
# Text-only conditioning => tempformer forward only => dep_q=16 fp32 is fine (no clean needed).
#
# Submit (after training done):
#   qsub pbs/run_personaplex_flip_eval.sh                       (auto: latest step in OUT)
#   qsub -v STEP=step_150 pbs/run_personaplex_flip_eval.sh
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

OUT="${OUT:-output/personaplex_poc_smoke}"
HELDOUT="${HELDOUT:-processed_data/persona_poc/persona_smoke_heldout-001-of-001.parquet}"
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

# auto-detect latest raw step_N
STEP="${STEP:-$(ls -d "$OUT"/step_* 2>/dev/null | grep -oE 'step_[0-9]+$' | sort -t_ -k2 -n | tail -1)}"
[ -n "$STEP" ] || { echo "FATAL: no step_N in $OUT"; exit 1; }
FP32="$OUT/${STEP}_fp32"
echo "consolidating $OUT/$STEP -> $FP32"
if [ ! -f "$FP32/model.safetensors" ]; then
  uv run -m tools.zero_to_fp32 "$OUT/$STEP" "$FP32" --moshi_lm_kwargs_path "$BASE_KWARGS"
fi

echo "=== flip eval on held-out ==="
uv run python mstts/data_prep/persona_flip_eval.py \
    --model_dir "$FP32" \
    --heldout   "$HELDOUT" \
    --tokenizer "$RINNA"
echo "=== flip eval done ==="
