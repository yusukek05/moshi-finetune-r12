#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ncpus=8:ngpus=4
#PBS -l walltime=01:00:00
#PBS -N 0162_4way_eval_compare_50
#PBS -j oe

# Larger-sample (50 examples) 4-way compare. Same models / seed as the 8-example
# run; goal is to confirm whether the 8-example listening result was statistically
# weak (no difference) or genuine (difference too small for 8 to show).

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
SYNTHPHASE1_DIR="output/v1.2_reazonspeech_jchat_synthphase1/step_2757_fp32"
ZOOM1_1EP_DIR="output/v1.2_reazonspeech_jchat_zoom1_1ep_baseline/step_1326_fp32"
SYNTH_THEN_ZOOM1_DIR="output/v1.2_synthphase1_then_zoom1_1ep/step_1326_fp32"
COMPARE_ROOT="output/4way_eval_compare_50"

for tag_dir in \
    "baseline_step8880:${BASELINE_DIR}" \
    "synthphase1_step2757:${SYNTHPHASE1_DIR}" \
    "zoom1_1ep_step1326:${ZOOM1_1EP_DIR}" \
    "synth_then_zoom1_step1326:${SYNTH_THEN_ZOOM1_DIR}"
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

echo "DONE. Wavs under ${COMPARE_ROOT}/{baseline,synthphase1,zoom1_1ep,synth_then_zoom1}/generated_wavs (50 each)"
