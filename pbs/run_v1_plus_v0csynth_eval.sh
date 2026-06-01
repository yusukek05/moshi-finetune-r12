#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q R9920261000
#PBS -l select=1:ncpus=8:ngpus=4
#PBS -l walltime=02:00:00
#PBS -N 0162_v1plusv0csynth_eval
#PBS -j oe

# Phase 2.2: Downstream eval — does v0c synth corpus continue-FT help on
# spoken dialogue continuation? Compare baseline v1 (vanilla = source of HF
# llm-jp-moshi-v1 release) vs v1_plus_v0csynth (this PBS's Phase 2.1 output)
# on the SAME Zoom1 test parquet as the existing v1_lineage_eval pipeline.
#
#   (A) v1                = vanilla 1node_exp J-CHAT -> Zoom1 (HF v1 source, dep_q=16 fp32)
#   (B) v1_plus_v0csynth  = v1 base + v0c mstts synth corpus FT (step_2757_fp32, dep_q=16)
#
# We use the un-cleaned (dep_q=16) fp32 ckpt because generate.py /
# moshi_for_generation.py assume n_q == dep_q. cleaned ckpts are only for
# moshi.server / HF release.
#
# Output: output/v1_plus_v0csynth_eval/{v1,v1_plus_v0csynth}/generated_wavs/
# Next:   qsub pbs/run_v1_plus_v0csynth_asr_cer.sh

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
COMPARE_ROOT="output/v1_plus_v0csynth_eval"

V1_DIR="output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_7epochs_1node_exp/step_9282_fp32"
V1_PLUS_DIR="output/v1_plus_v0csynth/step_2757_fp32"

for tag_dir in \
    "v1:${V1_DIR}" \
    "v1_plus_v0csynth:${V1_PLUS_DIR}"
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

echo "DONE: wavs under ${COMPARE_ROOT}/{v1,v1_plus_v0csynth}/generated_wavs/"
echo "Next: qsub pbs/run_v1_plus_v0csynth_asr_cer.sh"
