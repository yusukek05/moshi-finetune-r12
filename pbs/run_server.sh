#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=02:00:00

echo "JOB_ID: $PBS_JOBID"
cd /home/acg17145sv/experiments/0162_dialogue_model/moshi-finetune/

module load python/3.12/3.12.9
module load cuda/12.6/12.6.1

uv sync --python 3.12

uv run -m moshi.server \
    --moshi-weight output/moshi_p1_stage2_jchat_dialog/step_1968_cleaned/model.safetensors \
    --host 0.0.0.0 \
    --port 8998 \
    --tokenizer  data/tokenizer/spiece.model \
    --static /home/acg17145sv/experiments/0162_dialogue_model/moshi-finetune/client/dist