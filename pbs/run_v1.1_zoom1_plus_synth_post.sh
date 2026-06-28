#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N 0162_v1.1_synth_post
#PBS -j oe

# zero_to_fp32 for the v1.1 Zoom1+FireRed-synth run:
#   output/v1.1_zoom1_plus_synthdialogue/step_<final>  -> step_<final>_fp32
#
# Final step auto-detected (expected step_11683 = 7 epochs over zoom1+synth).
# moshi_lm_kwargs from the training base (Reazon->J-CHAT step_8880_fp32) — same arch.
# Skips if the fp32 dir already exists (safe to re-run).
# clean_moshi (dep_q=8) is NOT done here — head-to-head CER uses dep_q=16 fp32 directly,
# matching how the v1 lineage eval scored the baseline.
#
# Prereq: run_train_v1.1_zoom1_plus_synthdialogue.sh completed (step_11683).
# Next:   head-to-head CER vs baseline output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export NO_TORCH_COMPILE=1

RUN_DIR="output/v1.1_zoom1_plus_synthdialogue"
KWARGS="output/v1.2_reazonspeech_jchat/step_8880_fp32/moshi_lm_kwargs.json"

# NOTE: grep -o extracts the trailing "step_<n>" from the full path BEFORE sorting,
# otherwise sort -t_ -k2 keys on an underscore in the dir prefix (…_zoom1_plus_…)
# and mis-picks an earlier step. (_fp32 dirs are excluded by the trailing-digit anchor.)
STEP=$(ls -d "$RUN_DIR"/step_* 2>/dev/null | grep -oE 'step_[0-9]+$' | sort -t_ -k2 -n | tail -1)
if [ -z "$STEP" ]; then
    echo "ERROR: no step_* checkpoint in $RUN_DIR" >&2
    exit 1
fi

if [ -d "$RUN_DIR/${STEP}_fp32" ]; then
    echo "SKIP: $RUN_DIR/${STEP}_fp32 already exists"
else
    echo "===== zero_to_fp32: $RUN_DIR/$STEP ====="
    uv run -m tools.zero_to_fp32 \
        "$RUN_DIR/$STEP" \
        "$RUN_DIR/${STEP}_fp32" \
        --moshi_lm_kwargs_path "$KWARGS"
fi
ls -la "$RUN_DIR/${STEP}_fp32/"
echo "DONE: $RUN_DIR/${STEP}_fp32"
