#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HG,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:30:00
#PBS -j oe

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
module load cuda/12.6/12.6.1

uv sync --python 3.12

export NO_TORCH_COMPILE=1

model_dir="output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_mix_visualbank_mix_alagin_7epochs_1node_exp/step_13041_fp32/continuation_CallHome_test"

uv run -m tools.decode_tokens \
    --tokens_dir "${model_dir}/generated_tokens" \
    --output_dir "${model_dir}/generated_texts"
    # --text_tokenizer_repo rinna/japanese-gpt2-medium \
    # --text_tokenizer_name spiece.model
