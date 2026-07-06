#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:30:00
#PBS -N 0162_dpo_score_mean
#PBS -j oe
#PBS -o logs/

# DPO stage 2b: text-based meaningfulness reward. Judge each continuation's
# noise-free inner-monologue transcript with a local JP instruct LLM
# (llm-jp-3.1-13b-instruct4), 1-5. Fixes the acoustic evaluator's fluent-nonsense
# bias (docs 2026-07-01 §6). Writes <GEN_ROOT>/meaningfulness.json {npy: score}.
#
# Params (override at submit: qsub -v RTYPE=rt_HF,GEN_ROOT=...,SEEDS="42 43").
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

GEN_ROOT="${GEN_ROOT:-output/dpo_scaled}"
SEEDS="${SEEDS:-42 43 44 45 46 47}"
OUT="${OUT:-${GEN_ROOT}/meaningfulness.json}"

echo "GEN_ROOT=${GEN_ROOT} SEEDS=${SEEDS} OUT=${OUT}"
uv run python tools/score_meaningfulness_llm.py \
    --gen_root "${GEN_ROOT}" --seeds ${SEEDS} --out_json "${OUT}" \
    --model llm-jp/llm-jp-3.1-13b-instruct4 --batch_size 16
echo "DONE"
