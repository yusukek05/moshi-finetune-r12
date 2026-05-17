#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=03:00:00
#PBS -N 0162_asr_cer
#PBS -j oe

# Run ReazonSpeech-NeMo-ASR on every condition under output/v11_eval_compare/
# and output/4way_eval_compare_50/, then compute CER vs the prompt-region text
# decoded from the held-out parquet (rinna/japanese-gpt2-medium tokenizer).
#
# Results JSON for the ICML ML4Audio submission lands in
# /home/acg17145sv/projects/icml-mlforaudio-2026/results/.

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

PAPER=/home/acg17145sv/projects/icml-mlforaudio-2026

mkdir -p "$PAPER/results"

# faster-whisper (CTranslate2) is the most reliable JA ASR with a clean
# PyPI install. `reazonspeech-nemo-asr` is not on PyPI (lives in the
# reazonspeech monorepo); switching to whisper avoids that install path.
UV_DEPS=(--with faster-whisper --with jiwer --with soundfile
         --with transformers --with sentencepiece --with pyarrow --with numpy
         --with tqdm)

for ROOT in output/v11_eval_compare output/4way_eval_compare_50; do
    [ -d "$ROOT" ] || continue
    OUT_JSON="$PAPER/results/asr_cer_$(basename $ROOT).json"
    echo "=== $ROOT -> $OUT_JSON ==="
    uv run --no-project --python 3.12 "${UV_DEPS[@]}" \
        python "$PAPER/scripts/run_asr_cer_eval.py" \
            --compare_root "$ROOT" \
            --asr_backend whisper \
            --clip_seconds 0 \
            --output_json "$OUT_JSON"
done

echo "DONE."
