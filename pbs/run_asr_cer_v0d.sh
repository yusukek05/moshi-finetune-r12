#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N 0162_asr_cer_v0d
#PBS -j oe

# ASR-CER self-consistency for the v0d 3-way comparison.
# faster-whisper large-v3 transcribes each generated wav; CER is computed
# against the text-track (row 0 of generated_tokens/*.npy) the same model
# emitted. Low CER = the audio matches the model's own text plan.
#
# Conditions under output/v0d_eval_compare/:
#   v0c                — current production model (NC-licensed, LaboroTV lineage)
#   v0d                — new LaboroTV-free rebuild (moshika base + ccaudio)
#   from18000_moshiko  — prior LaboroTV-free attempt (moshiko base, no ccaudio)

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

PAPER=/home/acg17145sv/projects/icml-mlforaudio-2026
mkdir -p "$PAPER/results"

UV_DEPS=(--with faster-whisper --with jiwer --with soundfile
         --with transformers --with sentencepiece --with pyarrow --with numpy
         --with tqdm)

ROOT=output/v0d_eval_compare
OUT_JSON="$PAPER/results/asr_cer_v0d_eval_compare.json"
echo "=== $ROOT -> $OUT_JSON ==="
uv run --no-project --python 3.12 "${UV_DEPS[@]}" \
    python "$PAPER/scripts/run_asr_cer_eval.py" \
        --compare_root "$ROOT" \
        --asr_backend whisper \
        --clip_seconds 0 \
        --output_json "$OUT_JSON"

echo "DONE: $OUT_JSON"
