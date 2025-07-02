#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1 
#PBS -l walltime=02:00:00

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
uv sync --python 3.12

uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/J-CHAT/tokenized_text/youtube_valid \
    --tokenized_audio_dir data/J-CHAT/tokenized_audio/youtube_valid \
    --output_prefix processed_data/J-CHAT/youtube_valid
