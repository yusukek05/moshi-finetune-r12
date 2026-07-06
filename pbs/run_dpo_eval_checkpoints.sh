#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N 0162_dpo_eval_ckpts
#PBS -j oe
#PBS -o logs/

# DPO stage 3 checkpoint selection: generate continuations from each saved DPO
# checkpoint (bf16 inference format, no conversion needed) on the same contexts/
# seed as the base, decode the inner-monologue text, so we can pick the best
# checkpoint = learned (margin > base) AND not collapsed (distinct_ratio healthy).
# Base for comparison = output/dpo_scaled/seed42 (the policy BEFORE DPO, seed 42).
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

EVAL_DATA="processed_data/llmjp-zoom1/test-001-of-001.parquet"
RUN="output/dpo_scaled_run"
TEXT_REPO="rinna/japanese-gpt2-medium"; TEXT_NAME="spiece.model"

for CK in step_8 step_16 step_24 step_27; do
    MODEL="${RUN}/${CK}"
    OUT="${RUN}/gen_${CK}_seed42"
    [ -f "${MODEL}/model.safetensors" ] || { echo "skip ${CK} (no model)"; continue; }
    echo "===== generate ${CK} ====="
    uv run python generate.py \
        --output_dir "${OUT}" --model_dir "${MODEL}" \
        --eval_data_files "${EVAL_DATA}" \
        --prompt_length 125 --generation_length 250 --example_length 375 \
        --temperature 0.8 --num_examples 8 --seed 42
    uv run -m tools.decode_tokens \
        --tokens_dir "${OUT}/generated_tokens" \
        --output_dir "${OUT}/generated_wavs" \
        --text_output_dir "${OUT}/generated_text" \
        --text_tokenizer_repo "${TEXT_REPO}" --text_tokenizer_name "${TEXT_NAME}"
done
echo "DONE. base=output/dpo_scaled/seed42  dpo=${RUN}/gen_step*_seed42"
