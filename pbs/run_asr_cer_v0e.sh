#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N 0162_asr_cer_v0e
#PBS -j oe

# Stage the FINAL v0e mstts smoke outputs into the shared compare_root, then run
# the 6-way self-consistency ASR-CER (v0c / from18000 / v0d / v0d_v2 / v0d_v3 / v0e).
# Staging: inference writes decoded_audio/1-000X.wav + generated_tokens/1-000X.npy;
# the eval expects <cond>/generated_wavs/X.wav + generated_tokens/X.npy.
#
# Prereq: pbs/run_mstts_stage3_v0e_inference.sh has produced
#         output/mstts_stage3_zoom1_v0e/inference_step2008/{decoded_audio,generated_tokens}/

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

INF=output/mstts_stage3_zoom1_v0e/inference_step2008
DST=output/v0d_eval_compare/v0e
mkdir -p "$DST/generated_wavs" "$DST/generated_tokens"

# 1-000X -> X (strip the "1-000" prefix, drop leading zeros)
for w in "$INF"/decoded_audio/*.wav; do
    base=$(basename "$w" .wav)          # e.g. 1-0001
    idx=$((10#${base#1-}))              # 0001 -> 1
    cp "$w" "$DST/generated_wavs/${idx}.wav"
done
for t in "$INF"/generated_tokens/*.npy; do
    base=$(basename "$t" .npy)
    idx=$((10#${base#1-}))
    cp "$t" "$DST/generated_tokens/${idx}.npy"
done
echo "staged $(ls $DST/generated_wavs/*.wav | wc -l) wavs into $DST"

PAPER=/home/acg17145sv/projects/icml-mlforaudio-2026
mkdir -p "$PAPER/results"
UV_DEPS=(--with faster-whisper --with jiwer --with soundfile
         --with transformers --with sentencepiece --with pyarrow --with numpy
         --with tqdm)

ROOT=output/v0d_eval_compare
OUT_JSON="$PAPER/results/asr_cer_v0e_eval_compare.json"
uv run --no-project --python 3.12 "${UV_DEPS[@]}" \
    python "$PAPER/scripts/run_asr_cer_eval.py" \
        --compare_root "$ROOT" \
        --asr_backend whisper \
        --clip_seconds 0 \
        --output_json "$OUT_JSON"

echo "DONE: $OUT_JSON"
