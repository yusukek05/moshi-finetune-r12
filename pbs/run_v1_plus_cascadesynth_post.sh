#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:30:00
#PBS -N 0162_v1pluscascade_post
#PBS -j oe

# v1+cascadesynth post-training (one-shot):
#   1. zero_to_fp32   step_2714 (Zero3 shards) → step_2714_fp32 (~33 GB)
#   2. clean_moshi    step_2714_fp32 → step_2714_cleaned (dep_q=8 fp32, ~30 GB)
#   3. clean_moshi    step_2714_fp32 → step_2714_cleaned_bf16 (dep_q=8 bf16, ~15 GB, HF release)

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

RUN_DIR=output/v1_plus_cascadesynth
STEP=step_2714
# moshi_lm_kwargs come from the v1 base ckpt (same model architecture)
KWARGS=output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_7epochs_1node_exp/step_9282_fp32/moshi_lm_kwargs.json

echo "===== [1/3] zero_to_fp32 ====="
uv run -m tools.zero_to_fp32 \
    "$RUN_DIR/$STEP" \
    "$RUN_DIR/${STEP}_fp32" \
    --moshi_lm_kwargs_path "$KWARGS"
ls -la "$RUN_DIR/${STEP}_fp32/"

echo "===== [2/3] clean_moshi (fp32, remove user-stream) ====="
uv run -m tools.clean_moshi \
    --moshi_ft_dir "$RUN_DIR/${STEP}_fp32" \
    --save_dir     "$RUN_DIR/${STEP}_cleaned" \
    --model_dtype  float32 \
    --remove_modules_for_user_stream
ls -la "$RUN_DIR/${STEP}_cleaned/"

echo "===== [3/3] clean_moshi (bf16, HF release bundle) ====="
uv run -m tools.clean_moshi \
    --moshi_ft_dir "$RUN_DIR/${STEP}_fp32" \
    --save_dir     "$RUN_DIR/${STEP}_cleaned_bf16" \
    --model_dtype  bfloat16 \
    --remove_modules_for_user_stream
ls -la "$RUN_DIR/${STEP}_cleaned_bf16/"

echo "DONE"
