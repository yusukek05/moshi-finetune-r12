#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:40:00
#PBS -N 0162_dpo_ab_gen
#PBS -j oe
#PBS -o logs/
# Generate the selected DPO checkpoint (step_16) on 30 contexts (seed 42) for a
# human A/B vs base (= output/dpo_scaled/seed42, same contexts/seed, pre-DPO).
set -euxo pipefail
cd "$PBS_O_WORKDIR"
module purge; module load cuda/12.6/12.6.1; module load python/3.12/3.12.9
uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
export NO_TORCH_COMPILE=1 HF_HUB_OFFLINE=1
MODEL="output/dpo_scaled_run/step_16"
OUT="output/dpo_scaled_run/gen_step_16_seed42_ab"
uv run python generate.py --output_dir "$OUT" --model_dir "$MODEL" \
    --eval_data_files processed_data/llmjp-zoom1/test-001-of-001.parquet \
    --prompt_length 125 --generation_length 250 --example_length 375 \
    --temperature 0.8 --num_examples 30 --seed 42
uv run -m tools.decode_tokens --tokens_dir "$OUT/generated_tokens" \
    --output_dir "$OUT/generated_wavs" \
    --text_output_dir "$OUT/generated_text" \
    --text_tokenizer_repo rinna/japanese-gpt2-medium --text_tokenizer_name spiece.model
echo "DONE $OUT"
