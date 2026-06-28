#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q R9920261000
#PBS -l select=1:ncpus=8:ngpus=4
#PBS -l walltime=03:00:00
#PBS -N 0162_v1.1_firered_eval
#PBS -j oe

# Head-to-head: add the FireRed-synth-augmented run as a new condition to the
# existing v1 lineage eval, on the SAME Zoom1 test / seed / gen params as the
# baseline (v1.1_candidate = Zoom1-alone, self-consistency CER 0.62), then
# re-score every condition with ASR-CER.
#
#   exp  = v1.1_firered_synth = ReazonSpeech+J-CHAT -> Zoom1 + FireRed合成(×11)
#   base = v1.1_candidate     = ReazonSpeech+J-CHAT -> Zoom1   (already CER 0.62)
#   昇格判定: exp CER < 0.62 (有意) なら v1.1 として採用。
#
# Two-stream (dep_q=16) fp32 ckpt は generate.py がそのまま使える。
# Prereq: step_11683_fp32 (run_v1.1_zoom1_plus_synth_post.sh / afterok で連鎖)。

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

TAG="v1.1_firered_synth"
MODEL_DIR="output/v1.1_zoom1_plus_synthdialogue/step_11683_fp32"
OUT_DIR="${COMPARE_ROOT}/${TAG}"

# ── (1) generate the new condition (same params/seed as the lineage baseline) ──
if [ -d "${OUT_DIR}/generated_wavs" ] && [ "$(ls -A ${OUT_DIR}/generated_wavs 2>/dev/null)" ]; then
    echo "Already generated, skipping generation."
else
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
fi

# ── (2) ASR-CER over the whole lineage root (re-scores all conditions) ──
PAPER=/home/acg17145sv/projects/icml-mlforaudio-2026
mkdir -p "$PAPER/results"
UV_DEPS=(--with faster-whisper --with jiwer --with soundfile
         --with transformers --with sentencepiece --with pyarrow --with numpy
         --with tqdm)

uv run --no-project --python 3.12 "${UV_DEPS[@]}" \
    python "$PAPER/scripts/run_asr_cer_eval.py" \
        --compare_root "$COMPARE_ROOT" \
        --asr_backend whisper \
        --clip_seconds 0 \
        --output_json "$PAPER/results/asr_cer_v1_lineage_eval.json"

echo "DONE. CER json: $PAPER/results/asr_cer_v1_lineage_eval.json"
echo "  compare exp(${TAG}) vs base(v1.1_candidate=0.62)."
