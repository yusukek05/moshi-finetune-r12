#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=01:30:00
#PBS -N 0162_gen_persona_smoke
#PBS -j oe
#
# De-risk smoke for the PersonaPlex causal-corpus path (Task #131).
# Question this answers: does LLM-jp-4-8b actually write *tameguchi* (casual) JA
# when given a casual few-shot, and *teineigo* (polite) when given a polite one?
# (The existing 0386 corpus collapsed to ~uniform polite -> text-style control was
#  un-trainable. Fix = style-matched few-shot in gen_persona_dialogues.py.)
#
# GO criterion: the printed "formality confusion" matrix shows requested polite ->
# mostly polite AND requested casual -> mostly casual (clear separation, not 469/500
# all-polite). If casual stays polite, the text-style axis is dead -> pivot axis.
#
# Submit:  qsub pbs/run_gen_persona_dialogues_smoke.sh            (NPS=4 => 40 dialogues)
#          qsub -v NPS=2 pbs/run_gen_persona_dialogues_smoke.sh   (quick, 20 dialogues)
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"

REPO0162=/groups/gcg51557/experiments/0162_dialogue_model/moshi-finetune
ROOT0386=/groups/gcg51557/experiments/0386_dialogue_model
REPO0386=$ROOT0386/FireRedTTS2

NPS="${NPS:-4}"
SEED="${SEED:-0}"
OUT="${OUT:-$ROOT0386/data/dialogue_scripts/persona_gen_smoke}"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
export VIRTUAL_ENV="$REPO0386/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
nvidia-smi || true

# Run from the FireRedTTS2 repo so `uv run` resolves its (proven) torch+transformers
# env; the generator finds persona_schema via its own absolute dirname.
cd "$REPO0386"
uv run python "$REPO0162/mstts/data_prep/gen_persona_dialogues.py" \
    --out_dir "$OUT" \
    --styles polite,casual \
    --n_per_style "$NPS" \
    --seed "$SEED"

echo "=== gen_persona smoke done -> $OUT ==="
ls "$OUT" | wc -l
