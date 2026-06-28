#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q R9920261000
#PBS -l select=1:ncpus=8:ngpus=4
#PBS -l walltime=03:00:00
#PBS -N 0162_all_staged_eval
#PBS -j oe

# Add v1.1-all-staged as a new condition to the v1 lineage eval, same Zoom1 test
# / seed / gen params as v1 (73.7%) and v1.1 (62.0%), then re-score with ASR-CER.
#
#   all_staged = J-CHAT(step_8880) -> 全部盛り data_stage_3(step_942) -> Zoom1(step_9282)
#                = HF abePclWaseda/llm-jp-moshi-v1.1-all-staged の dep_q=16 学習版
#   問い: v1(74%) より良いか? (高道流の "実コーパス盛り合わせ" が効いたか)

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
TAG="all_staged"
MODEL_DIR="output/moshi-finetuned_init_text_emb_train_ohashi_data_stage_3_3epochs_llmjp-zoom1_7epochs/step_9282_fp32"
OUT_DIR="${COMPARE_ROOT}/${TAG}"

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

echo "DONE. compare all_staged vs v1(73.7%) / v1.1(62.0%) in asr_cer_v1_lineage_eval.json"
