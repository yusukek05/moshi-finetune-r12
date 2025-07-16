#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=20:00:00
#PBS -N 0162_tokenize_audio

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
uv sync --python 3.12

uv run -m tools.tokenize_audio \
    --audio_dir /home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/separated/podcast_train \
    --output_dir data/J-CHAT/tokenized_audio/podcast_train \
    --num_workers 8 \
    --resume
