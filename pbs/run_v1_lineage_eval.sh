#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q R9920261000
#PBS -l select=1:ncpus=8:ngpus=4
#PBS -l walltime=02:00:00
#PBS -N 0162_v1_lineage_eval
#PBS -j oe

# Phase 0: 3-way head-to-head on Zoom1 test using two-stream (dep_q=16) fp32
# checkpoints directly — generate.py / moshi_for_generation.py assume
# n_q == dep_q, which is satisfied by the un-cleaned _fp32 ckpts.
#
#   (A) v1            = J-CHAT -> Zoom1                                       (the local source for HF llm-jp/llm-jp-moshi-v1 release after clean)
#   (B) v1.1_candidate = ReazonSpeech + J-CHAT -> Zoom1                       (= our roadmap v1.1)
#   (C) v1.2_candidate = ReazonSpeech + J-CHAT + VisualBank -> Zoom1          (= our roadmap v1.2, gated on Phase A eval)
#
# clean_moshi.py is for the kyutai moshi.server runtime path, NOT for this
# codebase's generate.py — so we evaluate at the dep_q=16 layer directly.

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
COMPARE_ROOT="output/v1_lineage_eval"

V1_DIR="output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_7epochs_1node_exp_textpad1/step_9282_fp32"
V1_1_DIR="output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32"
V1_2_DIR="output/v1.2_reazonspeech_jchat_visualbank_zoom1/step_9282_fp32"

for tag_dir in \
    "v1:${V1_DIR}" \
    "v1.1_candidate:${V1_1_DIR}" \
    "v1.2_candidate:${V1_2_DIR}"
do
    TAG="${tag_dir%%:*}"
    MODEL_DIR="${tag_dir##*:}"
    OUT_DIR="${COMPARE_ROOT}/${TAG}"

    echo "===== ${TAG} ====="
    echo "model_dir=${MODEL_DIR}"
    echo "out_dir=${OUT_DIR}"

    if [ -d "${OUT_DIR}/generated_wavs" ] && [ "$(ls -A ${OUT_DIR}/generated_wavs 2>/dev/null)" ]; then
        echo "Already generated, skipping."
        continue
    fi

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

echo "DONE. 3-way wavs under ${COMPARE_ROOT}/{v1,v1.1_candidate,v1.2_candidate}/generated_wavs"
echo "Next: qsub pbs/run_asr_cer_eval.sh"
