#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N 0162_mstts_v0c_infer_custom
#PBS -j oe

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load hpcx/2.20
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

export NO_TORCH_COMPILE=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
unset NCCL_ASYNC_ERROR_HANDLING

OUT_DIR="$PWD/output/mstts_v0c_zoom1_finetune_step1500/inference_custom"
mkdir -p "$OUT_DIR"

cd mstts
mpirun \
  -n 1 \
  --map-by ppr:1:node:PE=1 \
  --bind-to none \
  --oversubscribe \
  -x NCCL_NVLS_ENABLE \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING \
  -x NO_TORCH_COMPILE \
  -x LD_LIBRARY_PATH -x PATH \
  uv run python run_ms_tts.py \
      --launcher mpi \
      --output_dir "$OUT_DIR" \
      --text_chat_data_dir inference_inputs/text_chat_custom \
      --model_dir ../output/mstts_v0c_zoom1_finetune_step1500/step_1500_fp32 \
      --model_dtype bfloat16 \
      --text_tokenizer_repo rinna/japanese-gpt2-medium \
      --text_tokenizer_name spiece.model \
      --prompt_streams_path inference_inputs/prompt_streams/standing_text_padded.npy \
      --per_device_batch_size 1 \
      --max_generation_length 1000 \
      --use_sampling \
      --text_temperature 0.55 \
      --audio_temperature 0.6 \
      --top_k 0 \
      --seed 1 \
  > "$OUT_DIR/infer_$PBS_JOBID.log" 2>&1
