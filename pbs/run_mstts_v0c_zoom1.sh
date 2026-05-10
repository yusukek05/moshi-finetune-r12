#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=8
#PBS -l walltime=06:00:00
#PBS -N 0162_mstts_v0c_zoom1
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
unset  NCCL_ASYNC_ERROR_HANDLING
export NCCL_IB_HCA=mlx5_0
unset NCCL_IB_DISABLE
ulimit -l unlimited

RUN_NAME=mstts_v0c_zoom1_finetune_step1500
OUT_DIR="$PWD/output/$RUN_NAME"
mkdir -p "$OUT_DIR"

cd mstts
mpirun \
  -n 8 \
  --map-by ppr:8:node:PE=1 \
  --bind-to none \
  --oversubscribe \
  -x NCCL_NVLS_ENABLE \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING -x NCCL_IB_HCA -x NCCL_IB_DISABLE \
  -x NO_TORCH_COMPILE \
  -x LD_LIBRARY_PATH -x PATH \
  uv run python finetune_ms_tts.py \
      --launcher mpi \
      --use_deepspeed \
      --deepspeed_config_file ds_configs/zero3-bf16-warmlr-act_ckpt.json \
      --activation_checkpointing \
      --output_dir "$OUT_DIR" \
      --train_data_files ../processed_data/llmjp-zoom1/train-001-of-001.parquet \
      --model_dir ../output/mstts_v0b_zoom1_finetune_step500/step_500_fp32 \
      --model_dtype bfloat16 \
      --model_user_stream \
      --max_length 2048 \
      --min_length 128 \
      --num_train_epochs 5 \
      --max_train_steps 1500 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --tempformer_learning_rate 1e-5 \
      --depformer_learning_rate 3e-5 \
      --num_warmup_steps 50 \
      --logging_steps 10 \
      --save_steps 500 \
      --moshi_speakers A B \
      --main_speaker_bos_id 1 \
      --other_speaker_bos_id 2 \
  > "$OUT_DIR/train_$PBS_JOBID.log" 2>&1
