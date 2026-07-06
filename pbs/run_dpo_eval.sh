#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:40:00
#PBS -N 0162_dpo_eval
#PBS -j oe
#PBS -o logs/

# DPO stage 3 eval-prep: consolidate the DPO checkpoint to a single fp32
# safetensors (inference format), then generate a few continuations from the
# DPO'd policy on the same contexts/seed as the base, so we can A/B base vs DPO.
# (Base = output/dpo_pilot/seed42, i.e. the same policy BEFORE DPO, seed 42.)
# NOTE: the pilot overfit 46 pairs x 8ep, so treat this as a sanity peek, not a
# verdict — success is judged on a scaled run by human A/B.
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

CKPT="output/dpo_pilot_run/step_24"
FP32="output/dpo_pilot_run/step_24_fp32"
KWARGS="output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32/moshi_lm_kwargs.json"
EVAL_DATA="processed_data/llmjp-zoom1/test-001-of-001.parquet"
OUT="output/dpo_pilot_run/gen_dpo_seed42"
TEXT_REPO="rinna/japanese-gpt2-medium"; TEXT_NAME="spiece.model"

# --- (1) consolidate ZeRO shards -> fp32 safetensors (inference format) ---
if [ -f "${FP32}/model.safetensors" ]; then
  echo "===== convert: reuse ${FP32} ====="
else
  echo "===== convert ${CKPT} -> ${FP32} ====="
  uv run -m tools.zero_to_fp32 "${CKPT}" "${FP32}" --moshi_lm_kwargs_path "${KWARGS}"
fi

# --- (2) generate DPO'd continuations (same contexts/seed as base seed42) ---
echo "===== generate (DPO policy) ====="
uv run python generate.py \
    --output_dir "${OUT}" \
    --model_dir "${FP32}" \
    --eval_data_files "${EVAL_DATA}" \
    --prompt_length 125 --generation_length 250 --example_length 375 \
    --temperature 0.8 --num_examples 8 --seed 42

uv run -m tools.decode_tokens \
    --tokens_dir "${OUT}/generated_tokens" \
    --output_dir "${OUT}/generated_wavs" \
    --text_output_dir "${OUT}/generated_text" \
    --text_tokenizer_repo "${TEXT_REPO}" --text_tokenizer_name "${TEXT_NAME}"
echo "DONE. base=output/dpo_pilot/seed42/{generated_wavs,generated_text}/{0..7}  dpo=${OUT}/..."
