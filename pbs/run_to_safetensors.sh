#!/bin/bash
#PBS -P gcg51557                
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8 
#PBS -l walltime=20:00:00
#PBS -j oe

echo "JOB_ID: $PBS_JOBID"
cd $PBS_O_WORKDIR

module load python/3.12/3.12.9
module load cuda/12.6/12.6.1

uv sync --python 3.12

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NO_TORCH_COMPILE=1
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

uv run -m tools.zero_to_fp32 \
    output/moshi-finetuned_podcast_train_callhome/step_15 \
    output/moshi-finetuned_podcast_train_callhome/step_15_fp32 \
    --moshi_lm_kwargs_path init_models/moshiko-both_streams-float32/moshi_lm_kwargs.json