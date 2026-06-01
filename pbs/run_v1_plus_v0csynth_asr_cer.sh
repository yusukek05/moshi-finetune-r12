#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:30:00
#PBS -N 0162_v1plusv0csynth_asrcer
#PBS -j oe

# Phase 2.2 — ASR-CER on v1 vs v1+v0csynth generated wavs (Zoom1 test
# continuation, 50 samples each). Uses faster-whisper for JA ASR + jiwer for
# CER vs the prompt-region text decoded from the held-out parquet.
#
# Prereq: pbs/run_v1_plus_v0csynth_eval.sh must have completed
#         (output/v1_plus_v0csynth_eval/{v1,v1_plus_v0csynth}/generated_wavs/)

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

# Reuse the eval script from the ML4Audio paper repo (model-agnostic, walks
# {compare_root}/{tag}/generated_wavs/).
ASR_SCRIPT=/home/acg17145sv/projects/icml-mlforaudio-2026/scripts/run_asr_cer_eval.py

uv run --no-project --python 3.12 "${UV_DEPS[@]}" \
    python "$ASR_SCRIPT" \
        --compare_root "$ROOT" \
        --asr_backend whisper \
        --clip_seconds 0 \
        --output_json "$OUT_JSON"

echo "DONE: $OUT_JSON"
