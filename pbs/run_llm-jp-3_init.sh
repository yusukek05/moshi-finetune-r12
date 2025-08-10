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
uv sync --python 3.12

uv run -m tools.init_llm-jp-3_for_ft \
    --save_dir init_models/llm-jp-3-both_streams-float32 \
    --model_dtype float32 \
    --extend_modules_for_user_stream \
    --use_llmjp_tempformer \
    --src_lm_repo llm-jp/llm-jp-3-7.2b
