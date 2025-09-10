#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=160:00:00
#PBS -N 0162_tokenize_audio
#PBS -j oe

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
uv sync --python 3.12

# ===== CallHome =====
uv run -m tools.tokenize_audio \
    --audio_dir /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/CallHome/audio \
    --output_dir data/data_stage_3/CallHome/tokenized_audio \
    --num_workers 8 \
    --resume

# ===== Chiba3Party =====
uv run -m tools.tokenize_audio \
    --audio_dir /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/Chiba3Party/audio \
    --output_dir data/data_stage_3/Chiba3Party/tokenized_audio \
    --num_workers 8 \
    --resume

# ===== CSJ =====
uv run -m tools.tokenize_audio \
    --audio_dir /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/CSJ/audio \
    --output_dir data/data_stage_3/CSJ/tokenized_audio \
    --num_workers 8 \
    --resume

# ===== eldery_listen_corpus_0.2 =====
uv run -m tools.tokenize_audio \
    --audio_dir /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/eldery_listen_corpus_0.2/audio \
    --output_dir data/data_stage_3/eldery_listen_corpus_0.2/tokenized_audio \
    --num_workers 8 \
    --resume

# ===== MapTask-Mie =====
uv run -m tools.tokenize_audio \
    --audio_dir /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/MapTask-Mie/audio \
    --output_dir data/data_stage_3/MapTask-Mie/tokenized_audio \
    --num_workers 8 \
    --resume

# ===== PASD =====
uv run -m tools.tokenize_audio \
    --audio_dir /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/PASD/audio \
    --output_dir data/data_stage_3/PASD/tokenized_audio \
    --num_workers 8 \
    --resume

# ===== RWCP-SP96 =====
uv run -m tools.tokenize_audio \
    --audio_dir /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/RWCP-SP96/audio \
    --output_dir data/data_stage_3/RWCP-SP96/tokenized_audio \
    --num_workers 8 \
    --resume

# ===== RWCP-SP97 =====
uv run -m tools.tokenize_audio \
    --audio_dir /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/RWCP-SP97/audio \
    --output_dir data/data_stage_3/RWCP-SP97/tokenized_audio \
    --num_workers 8 \
    --resume

# ===== UUDB =====
uv run -m tools.tokenize_audio \
    --audio_dir /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/UUDB/audio \
    --output_dir data/data_stage_3/UUDB/tokenized_audio \
    --num_workers 8 \
    --resume
