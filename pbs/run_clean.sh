#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
module load cuda/12.6/12.6.1

uv sync --python 3.12

export NO_TORCH_COMPILE=1

uv run -m tools.clean_moshi \
    --moshi_ft_dir output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_and_VisualBank_7epochs_1node_exp_textpad0.1/step_12516_fp32 \
    --save_dir output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_and_VisualBank_7epochs_1node_exp_textpad0.1/step_12516_cleaned \
    --model_dtype bfloat16 \
    --remove_modules_for_user_stream
