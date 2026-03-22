#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1 
#PBS -l walltime=02:00:00

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
uv sync --python 3.12

uv run -m tools.init_moshi_llmjp_ft \
    --llama_repo llm-jp/llm-jp-3-7.2b-instruct3 \
    --save_dir init_models/moshi_llmjp_instruct_user_ft \
    --model_user_stream \
    > model_init.log 2>&1
