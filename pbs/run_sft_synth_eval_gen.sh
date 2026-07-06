#!/bin/bash -l
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ncpus=192:ngpus=4
#PBS -l walltime=04:00:00
#PBS -N 0162_sft_synth_eval_gen
#PBS -j oe
#PBS -o logs/
#
# PRIMARY-GOAL eval: does blending FireRed synth dialogue into the Zoom1 stage
# improve semantic coherence over the v1.1 baseline? Generate the SAME 50-question
# Zoom1-test continuations used by the v1-lineage leaderboard (seed 42, dep_q=16 fp32
# directly — clean_moshi is only for the kyutai runtime, not generate.py) so scores
# are head-to-head comparable with v1.1_candidate (coherence 4.74 / overall 4.88).
#
# Two SFT tracks (both full 7 epochs on v1.1 step_9282_fp32 base + Zoom1 + synth blend):
#   sft_synth5pct  = Zoom1 + 5%  synth (CER<=0.20)  step_9772
#   sft_synth10pct = Zoom1 + 10% synth (CER<=0.20)  step_10318
#
# Submit: qsub pbs/run_sft_synth_eval_gen.sh
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"; mkdir -p logs

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"

export NO_TORCH_COMPILE=1
export ACCELERATE_DISTRIBUTED_TYPE=gloo
export PYTORCH_USE_RDMA=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

EVAL_DATA="processed_data/llmjp-zoom1/test-001-of-001.parquet"
COMPARE_ROOT="output/v1_lineage_eval"
BASE_KWARGS="output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32/moshi_lm_kwargs.json"

for tag_step_dir in \
    "sft_synth5pct:output/v1.1_zoom1_plus_synth100h_5pct/step_9772" \
    "sft_synth10pct:output/v1.1_zoom1_plus_synth100h_10pct/step_10318"
do
    TAG="${tag_step_dir%%:*}"
    STEP_DIR="${tag_step_dir##*:}"
    FP32="${STEP_DIR}_fp32"
    OUT_DIR="${COMPARE_ROOT}/${TAG}"

    echo "===== ${TAG} (step_dir=${STEP_DIR}) ====="

    if [ ! -f "${FP32}/model.safetensors" ]; then
        echo "consolidating ${STEP_DIR} -> ${FP32}"
        uv run -m tools.zero_to_fp32 "${STEP_DIR}" "${FP32}" --moshi_lm_kwargs_path "${BASE_KWARGS}"
    fi

    if [ -d "${OUT_DIR}/generated_wavs" ] && [ "$(ls -A ${OUT_DIR}/generated_wavs 2>/dev/null)" ]; then
        echo "Already generated, skipping ${TAG}."
        continue
    fi

    uv run accelerate launch \
        --num_machines 1 \
        --num_processes 4 \
        generate.py \
            --output_dir "${OUT_DIR}" \
            --model_dir "${FP32}" \
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

echo "DONE. wavs under ${COMPARE_ROOT}/{sft_synth5pct,sft_synth10pct}/generated_wavs"
echo "Next: qsub -v MODELS=sft_synth5pct:sft_synth10pct pbs/run_leaderboard_score.sh"
