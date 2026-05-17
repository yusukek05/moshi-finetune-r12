#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q R9920261000
#PBS -l select=1:ncpus=8:ngpus=4
#PBS -l walltime=00:30:00
#PBS -N 0162_v1_public_eval
#PBS -j oe

# Generate Zoom1-test continuations for public v1 (llm-jp/llm-jp-moshi-v1)
# under the same prompt/temperature/seed protocol as the existing
# output/v11_eval_compare/ runs, so the 4 conditions sit side by side and
# share a single ASR-CER pass.
#
#   (A) baseline_step8880        = J-CHAT pretrain (no Zoom1 anywhere)
#   (B) v1_public                = public LLM-jp v1 (J-CHAT -> Zoom1)               [NEW]
#   (C) v12_official_step9282    = ReazonSpeech + J-CHAT -> Zoom1   (= our v1.1)
#   (D) v11_candidate_step9282   = ReazonSpeech + J-CHAT -> VB -> Zoom1 (= v1.1+VB)
#
# Followup: re-run pbs/run_asr_cer_eval.sh (writes asr_cer_v11_eval_compare.json
# with all 4 entries).

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

EVAL_DATA="processed_data/llmjp-zoom1/test-001-of-001.parquet"
MODEL_DIR="init_models/llm-jp-moshi-v1"
OUT_DIR="output/v11_eval_compare/v1_public"

uv run accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    generate.py \
        --output_dir "${OUT_DIR}" \
        --model_dir "${MODEL_DIR}" \
        --eval_data_files "${EVAL_DATA}" \
        --prompt_length 125 \
        --generation_length 250 \
        --example_length 375 \
        --temperature 0.8 \
        --num_examples 50 \
        --seed 42

uv run -m tools.decode_tokens \
    --tokens_dir "${OUT_DIR}/generated_tokens" \
    --output_dir "${OUT_DIR}/generated_wavs"

echo "DONE. Wavs under ${OUT_DIR}/generated_wavs"
