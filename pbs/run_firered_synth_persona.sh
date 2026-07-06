#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=24:00:00
#PBS -N 0162_firered_synth_persona
#PBS -j oe
#PBS -o logs/
#
# FireRedTTS2 synthesis of the persona-conditioned dialogue scripts (Task #131).
# Reads gen_persona_dialogues.py output (0386 data/dialogue_scripts/persona_gen),
# emits L=S1/R=S2 turn-separated stereo wav @ 24 kHz (Moshi v1 convention, L=A R=B).
# 8 GPU shards. FireRedTTS2 = Apache-2.0 (commercial-clean synth source).
# Mirrors 0386/pbs/100_synth_dialogue.sh but with --scripts_dir = persona_gen and a
# persona-specific out_dir (0386 untouched).
#
# Submit:  qsub pbs/run_firered_synth_persona.sh                 (all 1000)
#          qsub -v LIMIT=25 pbs/run_firered_synth_persona.sh     (smoke: 25/shard = ~200)
#   diverse 100h corpus (4032 dialogues, expanded 31-pair pool):
#     qsub -v SCRIPTS_DIR=<...>/diverse_100h,OUTDIR=<...>/persona_diverse_100h_stereo,\
#             VOICE_POOL_JSON=<...>/data/voice_pool_100h.json pbs/run_firered_synth_persona.sh
#   infer_batch skips existing wavs -> re-submit safely resumes after walltime/preemption.
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"

ROOT=/groups/gcg51557/experiments/0386_dialogue_model
REPO=$ROOT/FireRedTTS2
MODELN="${MODELN:-drop_full}"
VOICE="${VOICE:-pool}"
NSHARD="${NSHARD:-8}"
LIMIT="${LIMIT:-0}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$ROOT/data/dialogue_scripts/persona_gen}"
VOICE_POOL_JSON="${VOICE_POOL_JSON:-$ROOT/data/voice_pool.json}"
PDIR="$REPO/pretrained_models/finetuned_${MODELN}"
OUTDIR="${OUTDIR:-$ROOT/output/dialogue_arena/persona_${MODELN}_${VOICE}_stereo}"

cd "$REPO"
module purge; module load cuda/12.6/12.6.1; module load python/3.12/3.12.9
export VIRTUAL_ENV="$REPO/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
nvidia-smi || true
[ -d "$PDIR" ] || { echo "FATAL: no model dir $PDIR"; exit 1; }
[ -d "$SCRIPTS_DIR" ] || { echo "FATAL: no scripts dir $SCRIPTS_DIR"; exit 1; }
mkdir -p "$OUTDIR" "$ROOT/logs"
echo "scripts=$(ls "$SCRIPTS_DIR"/*.json | wc -l)  out=$OUTDIR  voice=$VOICE  pool=$VOICE_POOL_JSON  limit=$LIMIT"

# warm up venv once, then launch shards with the venv python directly (avoid uv lock race)
uv run python -c "import torch; print('venv ok')"
PY_BIN="$VIRTUAL_ENV/bin/python"

for i in $(seq 0 $((NSHARD - 1))); do
  echo "=== shard $i / $NSHARD on GPU $i ==="
  CUDA_VISIBLE_DEVICES=$i "$PY_BIN" "$ROOT/scripts/infer_batch.py" \
      --scripts_dir "$SCRIPTS_DIR" \
      --pretrained_dir "$PDIR" --out_dir "$OUTDIR" --voice "$VOICE" \
      --voice_pool_json "$VOICE_POOL_JSON" \
      --num_shards "$NSHARD" --shard_idx "$i" --limit "$LIMIT" --stereo \
      > "$ROOT/logs/synth_persona_${MODELN}_sh${i}.out" 2>&1 &
done
wait
echo "=== firered persona synth done ($MODELN/$VOICE) ==="
ls "$OUTDIR"/*.wav 2>/dev/null | wc -l