#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=02:00:00
#PBS -N 0162_leaderboard_gt
#PBS -j oe
#PBS -o logs/
#
# Ground-Truth (natural) reference row for the leaderboard:
#   Stage 1 (moshi-finetune venv): Mimi-decode real continuation (frames 125:375) of the
#           first 50 Zoom1-test dialogues -> stereo GT wavs (L=A/R=B), 0.wav..49.wav.
#   Stage 2 (eval venv): score GT wavs with llm-jp-moshi-eval (UTMOS + LLM-Judge).
# CER is intentionally NOT computed for GT (self-consistency needs a model text-track;
# GT has none) -> leaderboard marks GT CER as N/A. GT is a reference, not ranked.
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"; mkdir -p logs

REPO=$PBS_O_WORKDIR
EVAL=/home/acg17145sv/projects/llm-jp-moshi-eval
GTWAV=$EVAL/results/leaderboard_gt_wavs
TESTPQ=processed_data/llmjp-zoom1/test-001-of-001.parquet

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
export CUDA_VISIBLE_DEVICES=0
export NO_TORCH_COMPILE=1

# --- Stage 1: decode GT continuations (moshi-finetune .venv has moshi/Mimi) ---
cd "$REPO"
uv sync --python 3.12
"$REPO/.venv/bin/python" tools/decode_gt_continuations.py \
    --parquet "$TESTPQ" --out_dir "$GTWAV" --n 50 --prompt_len 125 --example_len 375
echo "GT wavs: $(ls "$GTWAV"/*.wav 2>/dev/null | wc -l)"

# --- Stage 2: score with the eval tool (its own .venv) ---
cd "$EVAL"
uv venv --python 3.12 .venv 2>/dev/null || true
export VIRTUAL_ENV="$EVAL/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
uv pip install -e ".[utmos]"
python -m llm_jp_moshi_eval.cli run \
    --input_dir "$GTWAV" --metrics utmos,judge \
    --judge_model llm-jp/llm-jp-3.1-13b-instruct4 \
    --out_dir "$EVAL/results/leaderboard/ground_truth"

echo "=== GT SCORING DONE ==="
cat "$EVAL/results/leaderboard/ground_truth/summary.json"
