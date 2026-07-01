#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=04:00:00
#PBS -N 0162_gen_persona_full
#PBS -j oe
#
# Full persona-conditioned dialogue text generation for the PersonaPlex PoC (Task #131).
# De-risk smoke (Job 2000496) confirmed 100% formality separation, so scale up.
#
# Generates NPS dialogues per (style x topic). styles={polite,casual} x topics={5}
# = 10 combos. Default NPS=100 -> 1000 dialogues (500 polite / 500 casual), balanced.
# Parallelized across 8 GPUs by assigning combos round-robin (each combo pinned to one
# GPU). Output filenames are {style}_{topic}_{idx}.json => no cross-combo collision.
#
# Output feeds 0386 FireRedTTS2 infer_batch.py (text -> stereo wav) next.
#
# Submit:  qsub pbs/run_gen_persona_dialogues.sh                 (NPS=100 => 1000)
#          qsub -v NPS=50 pbs/run_gen_persona_dialogues.sh       (NPS=50  =>  500)
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"

REPO0162=/groups/gcg51557/experiments/0162_dialogue_model/moshi-finetune
ROOT0386=/groups/gcg51557/experiments/0386_dialogue_model
REPO0386=$ROOT0386/FireRedTTS2

NPS="${NPS:-100}"
OUT="${OUT:-$ROOT0386/data/dialogue_scripts/persona_gen}"
GEN="$REPO0162/mstts/data_prep/gen_persona_dialogues.py"
mkdir -p "$OUT"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
export VIRTUAL_ENV="$REPO0386/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
nvidia-smi || true
cd "$REPO0386"   # use FireRedTTS2 (proven torch+transformers) env

# Warm up / validate the venv once via uv (avoids 8 parallel `uv run` racing on the
# project lock); the parallel workers then call the venv python directly.
uv run python -c "import torch, transformers; print('venv ok', transformers.__version__)"
PY_BIN="$VIRTUAL_ENV/bin/python"

STYLES=(polite casual)
TOPICS=(hobby weekend food media place)
NGPU=8

# Build the combo list (style,topic) and launch one process per combo, pinned to a GPU
# round-robin. Each combo uses a distinct seed so parallel combos don't duplicate.
combos=()
for s in "${STYLES[@]}"; do for t in "${TOPICS[@]}"; do combos+=("$s:$t"); done; done

i=0
pids=()
for c in "${combos[@]}"; do
  s="${c%%:*}"; t="${c##*:}"
  gpu=$(( i % NGPU ))
  seed=$(( 100 + i ))
  log="$REPO0162/gen_persona_${s}_${t}.log"
  echo "[launch] combo=$c gpu=$gpu seed=$seed -> $log"
  CUDA_VISIBLE_DEVICES=$gpu "$PY_BIN" "$GEN" \
      --out_dir "$OUT" --styles "$s" --topics "$t" \
      --n_per_style "$NPS" --seed "$seed" > "$log" 2>&1 &
  pids+=("$!")
  i=$(( i + 1 ))
  # throttle: if we've launched NGPU jobs, wait for this wave before the next
  if (( i % NGPU == 0 )); then
    for p in "${pids[@]}"; do wait "$p"; done
    pids=()
  fi
done
# wait for any remaining
for p in "${pids[@]}"; do wait "$p"; done

echo "=== gen_persona full done -> $OUT ==="
ls "$OUT" | wc -l
echo "--- per-style counts ---"
echo "polite: $(ls "$OUT"/polite_* 2>/dev/null | wc -l)"
echo "casual: $(ls "$OUT"/casual_* 2>/dev/null | wc -l)"
echo "--- aggregate measured-formality match (should be ~100%) ---"
uv run python - <<PY
import glob, json
from collections import Counter
c = Counter()
for f in glob.glob("$OUT/*.json"):
    d = json.load(open(f))
    c[(d["style"], d["style_measured"])] += 1
for style in ("polite","casual"):
    tot = sum(v for (s,_),v in c.items() if s==style) or 1
    match = c.get((style,style),0)
    print(f"  {style:7s}: match {match}/{tot} = {100*match/tot:.0f}%   breakdown={[ (m,v) for (s,m),v in c.items() if s==style ]}")
PY
