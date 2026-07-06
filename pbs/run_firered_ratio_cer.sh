#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=02:00:00
#PBS -N 0162_firered_ratio_cer
#PBS -j oe
#PBS -o logs/
#
# Self-consistency CER (ASR(wav) vs model text-track) over v1_lineage_eval, now
# including firered_x3 / firered_x5, for the leaderboard CER column. Reuses the ICML
# run_asr_cer_eval.py (ephemeral uv env; does NOT touch the eval-tool .venv).
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"; mkdir -p logs
module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
export CUDA_VISIBLE_DEVICES=0

PAPER=/home/acg17145sv/projects/icml-mlforaudio-2026
OUT_JSON="$PAPER/results/asr_cer_v1_lineage_eval_ratio.json"
uv run --no-project --python 3.12 \
    --with faster-whisper --with jiwer --with soundfile \
    --with transformers --with sentencepiece --with pyarrow --with numpy --with tqdm \
    python "$PAPER/scripts/run_asr_cer_eval.py" \
        --compare_root output/v1_lineage_eval \
        --asr_backend whisper --clip_seconds 0 \
        --output_json "$OUT_JSON"
echo "=== CER DONE -> $OUT_JSON ==="
python3 -c "import json;d=json.load(open('$OUT_JSON'));print({k:round(v.get('cer',0),3) for k,v in d.items()})" 2>/dev/null || true
