#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q rt_HF
#PBS -l select=1:ncpus=8:ngpus=4
#PBS -l walltime=02:00:00
#PBS -N 0162_dpo_sample_gen
#PBS -j oe
#PBS -o logs/

# DPO stage 1: on-policy N-sample generation for preference-pair mining.
# Sample the CURRENT-BEST policy (v1.1_candidate = Zoom1-only continued) N times
# per context (different seeds -> different samples at temp 0.8), so each context
# yields N candidate continuations to rank by reward and pair up (chosen/rejected).
#
# Per seed we save: generated_tokens/<i>.npy (full 17-stream incl. text),
# generated_wavs/<i>.wav (2ch, for the acoustic naturalness reward), and
# generated_text/<i>.txt (noise-free inner-monologue transcript, rinna spm, for
# the text-based meaningfulness / anti-repetition reward).
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
export ACCELERATE_DISTRIBUTED_TYPE=gloo
export PYTORCH_USE_RDMA=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

# --- policy under improvement (best current v1.1) ---
POLICY_DIR="output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32"
EVAL_DATA="processed_data/llmjp-zoom1/test-001-of-001.parquet"
OUT_ROOT="output/dpo_pilot"

# rinna text tokenizer for the v1/v1.1/v1.2 lineage (NOT kyutai spm)
TEXT_REPO="rinna/japanese-gpt2-medium"
TEXT_NAME="spiece.model"

# N on-policy samples per context (one generate.py run per seed)
SEEDS=(42 43 44 45)

for S in "${SEEDS[@]}"; do
    OUT_DIR="${OUT_ROOT}/seed${S}"
    echo "===== seed ${S} -> ${OUT_DIR} ====="

    uv run accelerate launch \
        --num_machines 1 \
        --num_processes 4 \
        generate.py \
            --output_dir "${OUT_DIR}" \
            --model_dir "${POLICY_DIR}" \
            --eval_data_files "${EVAL_DATA}" \
            --prompt_length 125 \
            --generation_length 250 \
            --example_length 375 \
            --temperature 0.8 \
            --num_examples 50 \
            --seed "${S}"

    # audio for the acoustic naturalness reward
    uv run -m tools.decode_tokens \
        --tokens_dir "${OUT_DIR}/generated_tokens" \
        --output_dir "${OUT_DIR}/generated_wavs" \
        --text_output_dir "${OUT_DIR}/generated_text" \
        --text_tokenizer_repo "${TEXT_REPO}" \
        --text_tokenizer_name "${TEXT_NAME}"
done

echo "DONE. ${#SEEDS[@]} on-policy samples/context under ${OUT_ROOT}/seed*/"
