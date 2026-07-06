#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:40:00
#PBS -N 0162_dpo_selftest
#PBS -j oe
#PBS -o logs/

# DPO stage 3 self-test: validate the data path (prompt/continuation alignment,
# prompt-region masking) and the sequence log-prob BEFORE the full deepspeed run.
# At init policy==ref, so DPO loss must == log2 and logits == 0; we also print
# per-pair log-prob margins and scored-token counts.
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
export HF_HUB_OFFLINE=1

POLICY_DIR="output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32"
EVAL_DATA="processed_data/llmjp-zoom1/test-001-of-001.parquet"
PAIRS="output/dpo_pilot/dpo_pairs.jsonl"

uv run python dpo.py --selftest --num_selftest 8 \
    --model_dir "${POLICY_DIR}" \
    --eval_data_files "${EVAL_DATA}" \
    --pairs_jsonl "${PAIRS}" \
    --prompt_length 125 --example_length 375 --beta 0.1
echo "DONE"
