#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=100:00:00
#PBS -N 0162_prepare_dataset
#PBS -j oe

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
uv sync --python 3.12

uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/data_stage_3/Tabidachi/test/tokenized_text \
    --tokenized_audio_dir data/data_stage_3/Tabidachi/test/tokenized_audio \
    --output_prefix processed_data/data_stage_3/test/Tabidachi

