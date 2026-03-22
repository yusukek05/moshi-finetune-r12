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

# uv run -m tools.tokenize_text_from_dir \
#     --word_transcript /home/acg17145sv/experiments/0178_dialogue_tts/data/archive/J-CHAT \
#     --output_dir data/J-CHAT_ohashi/tokenized_text \
#     --text_tokenizer_repo rinna/japanese-gpt2-medium \
#     --text_tokenizer_name spiece.model \
#     --text_padding_id 3 \
#     --end_of_text_padding_id 0 \
#     --no_whitespace_before_word \
#     --num_workers 8 \
#     --resume

uv run -m tools.tokenize_text \
    --word_transcript /home/acg17145sv/experiments/0162_dialogue_model/AsReX/data/llmjp-zoom1/test/text \
    --output_dir data/llmjp-zoom1/test/tokenized_text \
    --text_tokenizer_repo rinna/japanese-gpt2-medium \
    --text_tokenizer_name spiece.model \
    --text_padding_id 3 \
    --end_of_text_padding_id 0 \
    --no_whitespace_before_word \
    --num_workers 8 \
    --resume

# uv run -m tools.tokenize_text \
#     --word_transcript /home/acg17145sv/experiments/0162_dialogue_model/AsReX/data/VisualBank/text \
#     --output_dir data/VisualBank/tokenized_text \
#     --text_tokenizer_repo rinna/japanese-gpt2-medium \
#     --text_tokenizer_name spiece.model \
#     --text_padding_id 3 \
#     --end_of_text_padding_id 0 \
#     --no_whitespace_before_word \
#     --num_workers 8 \
#     --resume