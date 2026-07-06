#!/bin/bash -l
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=04:00:00
#PBS -N 0162_synth_cer_filter
#PBS -j oe
#PBS -o logs/
#
# 100h corpus stage 4: CER filter for the diverse FireRed synth wavs.
# Scores every stereo synth wav against its own generated script (whisper-large-v3, mono
# downmix, char CER) via 0386/scripts/eval_dialogue_cer.py (8 GPU shards, --skip_utmos).
# Then merges per-shard jsonl, prints the CER distribution, and writes keep-lists at several
# thresholds so we can pick the cut after seeing the data (past calibration bugs -> look first).
#
# Submit:
#   qsub -q rt_HF -v RTYPE=rt_HF,USE_SSH=1 pbs/run_synth_cer_filter.sh
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"

ROOT=/groups/gcg51557/experiments/0386_dialogue_model
REPO=$ROOT/FireRedTTS2
WAVS="${WAVS:-$ROOT/output/dialogue_arena/persona_diverse_100h_stereo}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$ROOT/data/dialogue_scripts/diverse_100h}"
EVAL_DIR="${EVAL_DIR:-${WAVS}_eval}"
NSHARD="${NSHARD:-8}"

cd "$REPO"
module purge; module load cuda/12.6/12.6.1; module load python/3.12/3.12.9
export VIRTUAL_ENV="$REPO/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
export HF_HUB_OFFLINE=1   # whisper-large-v3 already cached
nvidia-smi || true
mkdir -p "$EVAL_DIR" "$ROOT/logs"
echo "wavs=$(ls "$WAVS"/*.wav | wc -l)  scripts=$(ls "$SCRIPTS_DIR"/*.json | wc -l)  eval=$EVAL_DIR"
PY_BIN="$VIRTUAL_ENV/bin/python"

for i in $(seq 0 $((NSHARD - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY_BIN" "$ROOT/scripts/eval_dialogue_cer.py" \
      --pools "$WAVS" --scripts_dir "$SCRIPTS_DIR" \
      --num_shards "$NSHARD" --shard_idx "$i" --skip_utmos \
      --out "$EVAL_DIR/cer_sh${i}.jsonl" \
      > "$ROOT/logs/synth_cer_sh${i}.out" 2>&1 &
done
wait
echo "=== CER scoring done ==="

# ---- merge + distribution + keep-lists at several thresholds ----
"$PY_BIN" - "$EVAL_DIR" <<'PY'
import glob, json, os, sys, statistics as st
ev = sys.argv[1]
recs = []
for f in sorted(glob.glob(f"{ev}/cer_sh*.jsonl")):
    for line in open(f):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
merged = f"{ev}/cer_all.jsonl"
with open(merged, "w") as w:
    for r in recs:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")
cers = sorted(r["cer"] for r in recs)
n = len(cers)
print(f"[merge] {n} dialogues scored -> {merged}")
if n:
    def pct(p): return cers[min(n-1, int(p*n))]
    print(f"[dist] mean={sum(cers)/n:.4f} median={st.median(cers):.4f} "
          f"p10={pct(.10):.4f} p25={pct(.25):.4f} p75={pct(.75):.4f} "
          f"p90={pct(.90):.4f} p95={pct(.95):.4f} max={cers[-1]:.4f}")
    # by cell (style_opening_topic prefix): mean CER per style/opening
    from collections import defaultdict
    grp = defaultdict(list)
    for r in recs:
        parts = r["stem"].split("_")
        key = "_".join(parts[:2]) if len(parts) >= 2 else r["stem"]
        grp[key].append(r["cer"])
    print("[by style_opening] " + "  ".join(
        f"{k}={sum(v)/len(v):.3f}(n{len(v)})" for k, v in sorted(grp.items())))
    for thr in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        keep = [r["stem"] for r in recs if r["cer"] <= thr]
        with open(f"{ev}/keep_cer{int(thr*100):02d}.txt", "w") as w:
            w.write("\n".join(sorted(keep)) + ("\n" if keep else ""))
        print(f"[keep thr<={thr:.2f}] {len(keep)}/{n} ({100*len(keep)/n:.1f}%) -> keep_cer{int(thr*100):02d}.txt")
PY
echo "=== 100h CER filter DONE -> $EVAL_DIR ==="
