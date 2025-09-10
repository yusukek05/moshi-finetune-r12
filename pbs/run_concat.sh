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

# ===== CallHome =====
uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/data_stage_3/CallHome/tokenized_text \
    --tokenized_audio_dir data/data_stage_3/CallHome/tokenized_audio \
    --output_prefix processed_data/data_stage_3/CallHome

# ===== Chiba3Party =====
uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/data_stage_3/Chiba3Party/tokenized_text \
    --tokenized_audio_dir data/data_stage_3/Chiba3Party/tokenized_audio \
    --output_prefix processed_data/data_stage_3/Chiba3Party

# ===== CSJ =====
uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/data_stage_3/CSJ/tokenized_text \
    --tokenized_audio_dir data/data_stage_3/CSJ/tokenized_audio \
    --output_prefix processed_data/data_stage_3/CSJ

# ===== eldery_listen_corpus_0.2 =====
uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/data_stage_3/eldery_listen_corpus_0.2/tokenized_text \
    --tokenized_audio_dir data/data_stage_3/eldery_listen_corpus_0.2/tokenized_audio \
    --output_prefix processed_data/data_stage_3/eldery_listen_corpus_0.2

# ===== MapTask-Mie =====
uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/data_stage_3/MapTask-Mie/tokenized_text \
    --tokenized_audio_dir data/data_stage_3/MapTask-Mie/tokenized_audio \
    --output_prefix processed_data/data_stage_3/MapTask-Mie

# ===== PASD =====
uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/data_stage_3/PASD/tokenized_text \
    --tokenized_audio_dir data/data_stage_3/PASD/tokenized_audio \
    --output_prefix processed_data/data_stage_3/PASD

# ===== RWCP-SP96 =====
uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/data_stage_3/RWCP-SP96/tokenized_text \
    --tokenized_audio_dir data/data_stage_3/RWCP-SP96/tokenized_audio \
    --output_prefix processed_data/data_stage_3/RWCP-SP96

# ===== RWCP-SP97 =====
uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/data_stage_3/RWCP-SP97/tokenized_text \
    --tokenized_audio_dir data/data_stage_3/RWCP-SP97/tokenized_audio \
    --output_prefix processed_data/data_stage_3/RWCP-SP97

# ===== UUDB =====
uv run -m tools.prepare_dataset \
    --tokenized_text_dir data/data_stage_3/UUDB/tokenized_text \
    --tokenized_audio_dir data/data_stage_3/UUDB/tokenized_audio \
    --output_prefix processed_data/data_stage_3/UUDB
