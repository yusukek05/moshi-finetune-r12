#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q R9920261000
#PBS -l select=1:ncpus=8:ngpus=4
#PBS -l walltime=01:00:00
#PBS -N 0162_v11_eval_compare
#PBS -j oe

# 3-way listening comparison to decide v1.1 release-readiness:
#   (A) baseline             = v1.2 J-CHAT pretrain step_8880 (Zoom1 unseen)
#   (B) v1.2 official        = baseline + Zoom1 7ep (released v1.2)
#   (C) v1.1 candidate       = baseline + VisualBank 3ep + Zoom1 7ep
# If C >= B in naturalness/turn-taking, VB stage justified -> publish v1.1.

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
BASELINE_DIR="output/v1.2_reazonspeech_jchat/step_8880_fp32"
V12_OFFICIAL_DIR="output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32"
V11_CANDIDATE_DIR="output/v1.2_reazonspeech_jchat_visualbank_zoom1/step_9282_fp32"
COMPARE_ROOT="output/v11_eval_compare"

for tag_dir in \
    "baseline_step8880:${BASELINE_DIR}" \
    "v12_official_step9282:${V12_OFFICIAL_DIR}" \
    "v11_candidate_step9282:${V11_CANDIDATE_DIR}"
do
    TAG="${tag_dir%%:*}"
    MODEL_DIR="${tag_dir##*:}"
    OUT_DIR="${COMPARE_ROOT}/${TAG}"

    echo "===== ${TAG} ====="
    echo "model_dir=${MODEL_DIR}"
    echo "out_dir=${OUT_DIR}"

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
done

echo "DONE. Wavs under ${COMPARE_ROOT}/{baseline_step8880,v12_official_step9282,v11_candidate_step9282}/generated_wavs"
