#!/bin/bash -l
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=03:00:00
#PBS -N 0162_personaplex_gen_flip_topic
#PBS -j oe
#PBS -o logs/
#
# TOPIC generation-flip (gold): prime topic prompt prefix, generate, classify generated
# text's topic (24-way, char-2gram sim to per-topic reference). Tests CONTENT control —
# the axis the user actually wants ("talk about X"), stronger than the failed formality axis.
#
# Submit: qsub -q rt_HF -v RTYPE=rt_HF,USE_SSH=1 pbs/run_personaplex_gen_flip_topic.sh
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

OUT="${OUT:-output/personaplex_topic_100h}"
STEP="${STEP:-$(ls -d "$OUT"/step_* 2>/dev/null | grep -oE 'step_[0-9]+$' | sort -t_ -k2 -n | tail -1)}"
CARRIERS="${CARRIERS:-processed_data/topic_100h/heldout_topic-001-of-001.parquet}"
SCRIPTS="${SCRIPTS:-/groups/gcg51557/experiments/0386_dialogue_model/data/dialogue_scripts/diverse_100h}"
SCRIPTS_JSONL="${SCRIPTS_JSONL:-}"
N="${N:-96}"
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

[ -n "$STEP" ] || { echo "FATAL: no step in $OUT"; exit 1; }
FP32="$OUT/${STEP}_fp32"
if [ ! -f "$FP32/model.safetensors" ]; then
  echo "consolidating $OUT/$STEP -> $FP32"
  uv run -m tools.zero_to_fp32 "$OUT/$STEP" "$FP32" --moshi_lm_kwargs_path "$BASE_KWARGS"
fi

# build a merged scripts jsonl for topic references if not provided
if [ -z "$SCRIPTS_JSONL" ]; then
  SCRIPTS_JSONL="$OUT/diverse_100h_scripts.jsonl"
  if [ ! -f "$SCRIPTS_JSONL" ]; then
    uv run --no-project --python 3.12 python - "$SCRIPTS" "$SCRIPTS_JSONL" <<'PY'
import sys, glob, json, os
srcdir, out = sys.argv[1], sys.argv[2]
with open(out, "w") as w:
    for p in sorted(glob.glob(f"{srcdir}/*.json")):
        d = json.load(open(p, encoding="utf-8")); d["dialogue_id"] = os.path.splitext(os.path.basename(p))[0]
        w.write(json.dumps(d, ensure_ascii=False) + "\n")
print("scripts jsonl ready")
PY
  fi
fi

uv run python mstts/data_prep/persona_gen_flip_topic.py \
    --model_dir "$FP32" --carriers "$CARRIERS" --scripts "$SCRIPTS_JSONL" \
    --tokenizer "$RINNA" --n "$N" --dump "$OUT/gen_flip_topic_${STEP}_n${N}.jsonl"
echo "=== topic gen flip done ($STEP, n=$N) ==="
