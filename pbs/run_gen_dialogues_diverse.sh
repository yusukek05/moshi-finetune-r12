#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=10:00:00
#PBS -N 0162_gen_dialogues_diverse
#PBS -j oe
#PBS -o logs/
#
# Diverse dialogue text generation for the ~100h FireRed corpus (dual-use: SFT + PersonaPlex).
# 24 topics x 2 styles x 2 openings x NPС dialogues. Topics sharded across 8 GPUs.
# Generator = LLM-jp-4-8b (confirmed best JA generator). Output feeds FireRed synth next.
#   NPC=42 -> ~4032 dialogues ~= 100h audio.  qsub -v ...,NPC=42
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"

REPO0162=/groups/gcg51557/experiments/0162_dialogue_model/moshi-finetune
REPO0386=/groups/gcg51557/experiments/0386_dialogue_model/FireRedTTS2
GEN="$REPO0162/mstts/data_prep/gen_dialogues_diverse.py"
OUT="${OUT:-/groups/gcg51557/experiments/0386_dialogue_model/data/dialogue_scripts/diverse_100h}"
NPC="${NPC:-42}"
mkdir -p "$OUT" "$REPO0162/logs"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
export VIRTUAL_ENV="$REPO0386/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
nvidia-smi || true
cd "$REPO0386"
uv run python -c "import torch,transformers;print('venv ok',transformers.__version__)"
PY_BIN="$VIRTUAL_ENV/bin/python"

# 24 topics sharded into 8 GPU groups (3 each).
# NOTE: do NOT name this GROUPS — that's a reserved bash array (user's gids).
TOPIC_GROUPS=(
  "hobby,weekend,food"   "media,place,travel"   "work,health,pet"     "family,sports,music"
  "study,season,hometown" "future,shopping,cooking" "game,tech,money"  "book,cafe,event"
)
pids=()
for i in "${!TOPIC_GROUPS[@]}"; do
  g="${TOPIC_GROUPS[$i]}"; seed=$((100 + i))
  log="$REPO0162/logs/gen_diverse_g${i}.log"
  echo "[launch] gpu=$i topics=$g seed=$seed -> $log"
  CUDA_VISIBLE_DEVICES=$i "$PY_BIN" "$GEN" \
      --out_dir "$OUT" --topics "$g" --styles polite,casual --openings greeting,midconv \
      --n_per_cell "$NPC" --seed "$seed" > "$log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p"; done

echo "=== gen diverse DONE -> $OUT ==="
echo "total dialogues: $(ls "$OUT"/*.json 2>/dev/null | wc -l)"
echo "--- by style/opening ---"
for st in polite casual; do for op in greeting midconv; do
  echo "$st/$op: $(ls "$OUT"/${st}_${op}_* 2>/dev/null | wc -l)"; done; done
