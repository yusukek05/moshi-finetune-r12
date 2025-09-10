#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=100:00:00
#PBS -N 0162_tokenize_text
#PBS -j oe

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
uv sync --python 3.12

# ===== CallHome =====
uv run -m tools.tokenize_text \
    --word_transcript /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/CallHome/text \
    --output_dir data/data_stage_3/CallHome/tokenized_text \
    --text_tokenizer_repo rinna/japanese-gpt2-medium \
    --text_tokenizer_name spiece.model \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --no_whitespace_before_word \
    --num_workers 8 \
    --resume

# ===== Chiba3Party =====
uv run -m tools.tokenize_text \
    --word_transcript /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/Chiba3Party/text \
    --output_dir data/data_stage_3/Chiba3Party/tokenized_text \
    --text_tokenizer_repo rinna/japanese-gpt2-medium \
    --text_tokenizer_name spiece.model \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --no_whitespace_before_word \
    --num_workers 8 \
    --resume

# ===== CSJ =====
uv run -m tools.tokenize_text \
    --word_transcript /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/CSJ/text \
    --output_dir data/data_stage_3/CSJ/tokenized_text \
    --text_tokenizer_repo rinna/japanese-gpt2-medium \
    --text_tokenizer_name spiece.model \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --no_whitespace_before_word \
    --num_workers 8 \
    --resume

# ===== eldery_listen_corpus_0.2 =====
uv run -m tools.tokenize_text \
    --word_transcript /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/eldery_listen_corpus_0.2/text \
    --output_dir data/data_stage_3/eldery_listen_corpus_0.2/tokenized_text \
    --text_tokenizer_repo rinna/japanese-gpt2-medium \
    --text_tokenizer_name spiece.model \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --no_whitespace_before_word \
    --num_workers 8 \
    --resume

# ===== MapTask-Mie =====
uv run -m tools.tokenize_text \
    --word_transcript /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/MapTask-Mie/text \
    --output_dir data/data_stage_3/MapTask-Mie/tokenized_text \
    --text_tokenizer_repo rinna/japanese-gpt2-medium \
    --text_tokenizer_name spiece.model \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --no_whitespace_before_word \
    --num_workers 8 \
    --resume

# ===== PASD =====
uv run -m tools.tokenize_text \
    --word_transcript /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/PASD/text \
    --output_dir data/data_stage_3/PASD/tokenized_text \
    --text_tokenizer_repo rinna/japanese-gpt2-medium \
    --text_tokenizer_name spiece.model \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --no_whitespace_before_word \
    --num_workers 8 \
    --resume

# ===== RWCP-SP96 =====
uv run -m tools.tokenize_text \
    --word_transcript /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/RWCP-SP96/text \
    --output_dir data/data_stage_3/RWCP-SP96/tokenized_text \
    --text_tokenizer_repo rinna/japanese-gpt2-medium \
    --text_tokenizer_name spiece.model \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --no_whitespace_before_word \
    --num_workers 8 \
    --resume

# ===== RWCP-SP97 =====
uv run -m tools.tokenize_text \
    --word_transcript /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/RWCP-SP97/text \
    --output_dir data/data_stage_3/RWCP-SP97/tokenized_text \
    --text_tokenizer_repo rinna/japanese-gpt2-medium \
    --text_tokenizer_name spiece.model \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --no_whitespace_before_word \
    --num_workers 8 \
    --resume

# ===== UUDB =====
uv run -m tools.tokenize_text \
    --word_transcript /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/UUDB/text \
    --output_dir data/data_stage_3/UUDB/tokenized_text \
    --text_tokenizer_repo rinna/japanese-gpt2-medium \
    --text_tokenizer_name spiece.model \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --no_whitespace_before_word \
    --num_workers 8 \
    --resume
