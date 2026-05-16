#!/bin/bash -l
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N 0162_process_mono
#PBS -j oe

# Offline preprocessing for mono training: builds the HF datasets cache so the
# main training run skips preprocessing entirely (no NCCL watchdog timeout).
# CPU-bound; reuses 32 workers across the 192-core node.

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

DP=/groups/gcg51557/experiments/0178_dialogue_tts/data/parquet

uv run process_mono.py \
    --train_data_files \
        "${DP}/J-CHAT-mono/rinna_gpt2-kyutai_mimi-q16/shard-*.parquet" \
        "${DP}/reazonspeech/rinna_gpt2-kyutai_mimi-q16/shard-000[0-2][0-9].parquet" \
    --model_dir init_models/moshiko-single_streams-float32 \
    --model_dtype bfloat16 \
    --max_length 2048 \
    --max_audio_delay 10 \
    --dataset_processing_workers 32

echo "DONE: HF datasets cache populated."
