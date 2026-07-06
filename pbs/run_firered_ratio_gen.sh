#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=03:00:00
#PBS -N 0162_firered_ratio_gen
#PBS -j oe
#PBS -o logs/
#
# FireRed-synth ratio sweep -> leaderboard: consolidate x3(≈5%)/x5(≈10%) final ckpts to
# fp32, then generate 50 dialogue continuations each on the SAME Zoom1-test set / params
# as v1_lineage_eval (prompt125/gen250, seed42, temp0.8), decode to stereo wavs.
# Outputs land under output/v1_lineage_eval/{firered_x3,firered_x5} so the existing
# leaderboard scoring picks them up.
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
export PYTORCH_USE_RDMA=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo

BASE_KWARGS="output/v1.2_reazonspeech_jchat/step_8880_fp32/moshi_lm_kwargs.json"
EVAL_DATA="processed_data/llmjp-zoom1/test-001-of-001.parquet"
COMPARE_ROOT="output/v1_lineage_eval"

# tag : run_dir : final_step
for spec in \
    "firered_x3:output/v1.1_zoom1_plus_synth_x3:step_9933" \
    "firered_x5:output/v1.1_zoom1_plus_synth_x5:step_10374"
do
    TAG="${spec%%:*}"; rest="${spec#*:}"; RUN="${rest%%:*}"; STEP="${rest##*:}"
    FP32="$RUN/${STEP}_fp32"
    OUT_DIR="$COMPARE_ROOT/$TAG"
    echo "===== $TAG (from $RUN/$STEP) ====="

    if [ ! -f "$FP32/model.safetensors" ]; then
        uv run -m tools.zero_to_fp32 "$RUN/$STEP" "$FP32" --moshi_lm_kwargs_path "$BASE_KWARGS"
    fi

    if [ -d "$OUT_DIR/generated_wavs" ] && [ "$(ls -A "$OUT_DIR/generated_wavs" 2>/dev/null)" ]; then
        echo "already generated, skip"; continue
    fi
    uv run accelerate launch --num_machines 1 --num_processes 4 \
        generate.py \
            --output_dir "$OUT_DIR" --model_dir "$FP32" \
            --eval_data_files "$EVAL_DATA" \
            --prompt_length 125 --generation_length 250 --example_length 375 \
            --temperature 0.8 --num_examples 50 --seed 42
    uv run -m tools.decode_tokens \
        --tokens_dir "$OUT_DIR/generated_tokens" --output_dir "$OUT_DIR/generated_wavs"
    echo "done $TAG : $(ls "$OUT_DIR/generated_wavs"/*.wav 2>/dev/null | wc -l) wavs"
done
echo "=== FIRERED RATIO GEN DONE ==="
