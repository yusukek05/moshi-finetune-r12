#!/bin/bash -l
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N 0162_process_mstts
#PBS -j oe

# Offline preprocessing of J-CHAT multi-stream parquet (~5M rows) so the
# Stage 2 training job's main_process_first() map() is an instant cache hit
# (avoids the NCCL collective-op watchdog timeout we hit in mono).

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

DJ=/groups/gcg51557/experiments/0178_dialogue_tts/data-jmoshi

uv run python -m tools.process_mstts \
    --train_data_files \
        "${DJ}/jchat-podcast/train-*.parquet" \
    --model_dir init_models/mstts_init_from_mono_step8001 \
    --model_dtype bfloat16 \
    --max_length 2048 \
    --min_length 128 \
    --moshi_speakers A B \
    --main_speaker_bos_id 1 \
    --other_speaker_bos_id 2 \
    --dataset_processing_workers 32

echo "DONE: HF datasets cache populated for mstts Stage 2."
