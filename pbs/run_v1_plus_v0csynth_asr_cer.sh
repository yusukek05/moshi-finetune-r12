#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=03:00:00
#PBS -N 0162_v1plusv0csynth_asrcer
#PBS -j oe

# Phase 2.2 — self-consistency ASR-CER on all tags under
# output/v1_plus_v0csynth_eval/ (Zoom1 test continuation, 50 samples each;
# the eval script walks every subdir with generated_wavs/, so the Zoom1
# re-anchored tags are picked up automatically). faster-whisper JA ASR +
# jiwer CER vs the model's own text track.
#
# Two passes:
#   1. mix — downmix both speakers (legacy numbers; hyp contains the other
#            speaker's speech too, inflating CER as insertions)
#   2. ch1 — main channel only (matches the text-track speaker; cleaner
#            pairing, the headline number for the paper)
#
# Prereq: pbs/run_v1_plus_v0csynth_eval.sh must have completed

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

ICASSP=/home/acg17145sv/projects/icassp-2027-mstts
mkdir -p "$ICASSP/results"

UV_DEPS=(--with faster-whisper --with jiwer --with soundfile
         --with transformers --with sentencepiece --with pyarrow --with numpy
         --with tqdm)

ROOT="output/v1_plus_v0csynth_eval"
OUT_JSON="$ICASSP/results/asr_cer_v1_plus_v0csynth_eval.json"
OUT_JSON_CH1="$ICASSP/results/asr_cer_v1_plus_v0csynth_eval_mainch.json"

# Reuse the eval script from the ML4Audio paper repo (model-agnostic, walks
# {compare_root}/{tag}/generated_wavs/).
ASR_SCRIPT=/home/acg17145sv/projects/icml-mlforaudio-2026/scripts/run_asr_cer_eval.py

uv run --no-project --python 3.12 "${UV_DEPS[@]}" \
    python "$ASR_SCRIPT" \
        --compare_root "$ROOT" \
        --asr_backend whisper \
        --clip_seconds 0 \
        --asr_channel mix \
        --output_json "$OUT_JSON"

uv run --no-project --python 3.12 "${UV_DEPS[@]}" \
    python "$ASR_SCRIPT" \
        --compare_root "$ROOT" \
        --asr_backend whisper \
        --clip_seconds 0 \
        --asr_channel 1 \
        --output_json "$OUT_JSON_CH1"

echo "DONE: $OUT_JSON"
echo "      $OUT_JSON_CH1"
