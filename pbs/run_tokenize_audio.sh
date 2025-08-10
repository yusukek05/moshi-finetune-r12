#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=60:00:00
#PBS -N 0162_tokenize_audio
#PBS -j oe

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
uv sync --python 3.12

uv run -m tools.tokenize_audio \
    --audio_dir /home/acg17145sv/experiments/0162_dialogue_model/CSJ/audio/noncore \
    --output_dir data/CSJ/tokenized_audio/noncore \
    --num_workers 8 \
    --resume