#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=8
#PBS -l walltime=01:00:00
#PBS -N 0162_mstts_smoke
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

TOY_DIR="$PWD/output/mstts_smoke_$PBS_JOBID"
mkdir -p "$TOY_DIR"
TOY_PARQUET="$TOY_DIR/toy.parquet"
uv run python -c "
import pyarrow.parquet as pq
src = 'processed_data/VisualBank/train-001-of-001.parquet'
pq.write_table(pq.read_table(src).slice(0, 4), '$TOY_PARQUET')
print('toy parquet ready')
"

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
      --deepspeed_config_file ds_configs/zero3-bf16-act_ckpt.json \
      --activation_checkpointing \
      --output_dir "$TOY_DIR" \
      --train_data_files "$TOY_PARQUET" \
      --model_dir ../init_models/moshiko-both_streams-float32 \
      --model_dtype bfloat16 \
      --model_user_stream \
      --max_length 512 \
      --min_length 64 \
      --num_train_epochs 1 \
      --max_train_steps 2 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --tempformer_learning_rate 3e-5 \
      --depformer_learning_rate 3e-5 \
      --num_warmup_steps 0 \
      --logging_steps 1 \
      --save_steps 100000 \
      --moshi_speakers A B \
      --main_speaker_bos_id 1 \
      --other_speaker_bos_id 2 \
  > "$TOY_DIR/smoke_$PBS_JOBID.log" 2>&1
