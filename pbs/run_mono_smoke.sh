#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ncpus=8:ngpus=8
#PBS -l walltime=00:30:00
#PBS -N 0162_mono_smoke
#PBS -j oe

# Smoke test for the ported mono training (finetune_mono_text.py).
# Validates: code port, base model load, 0178 data read, 1 train step + save.
# Full run is run_train_mono.sh once this passes.
#   - Base: init_models/moshiko-single_streams-float32 (commercial-clean, CC-BY-4.0)
#   - Data: 0178 data/parquet/{J-CHAT-mono,reazonspeech}/rinna_gpt2-kyutai_mimi-q16
#           (rinna GPT2 tokenizer, A_text fits moshiko 32000 vocab; NO LaboroTV, no kana)

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

GPUS_PER_NODE=8
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_mono_smoke_$PBS_JOBID
NNODES=$(wc -l < hostfile_mono_smoke_$PBS_JOBID)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
echo "WORLD_SIZE=${WORLD_SIZE}"

D0178=/groups/gcg51557/experiments/0178_dialogue_tts/data/parquet

mpirun \
  -n  ${WORLD_SIZE} \
  --map-by ppr:${GPUS_PER_NODE}:node:PE=1 \
  --bind-to none \
  --oversubscribe \
  -x NCCL_NVLS_ENABLE -x NO_TORCH_COMPILE \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING -x NCCL_IB_HCA -x NCCL_IB_DISABLE \
  -x LD_LIBRARY_PATH -x PATH \
  uv run finetune_mono_text.py \
      --launcher mpi \
      --use_deepspeed \
      --deepspeed_config_file ds_configs/zero0-bf16-warmlr-act_ckpt.json \
      --output_dir output/mono_smoke \
      --train_data_files \
          "${D0178}/J-CHAT-mono/rinna_gpt2-kyutai_mimi-q16/shard-00000.parquet" \
          "${D0178}/reazonspeech/rinna_gpt2-kyutai_mimi-q16/shard-00000.parquet" \
      --model_dir init_models/moshiko-single_streams-float32 \
      --model_dtype bfloat16 \
      --max_length 2048 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 1 \
      --audio_loss_weight_when_text_pad 1.0 \
      --text_padding_loss_weight 0.5 \
      --semantic_loss_weight 100 \
      --acoustic_loss_weight 1 \
      --parameters_to_finetune all \
      --tempformer_learning_rate 3e-4 \
      --depformer_learning_rate 3e-4 \
      --num_warmup_steps 500 \
      --weight_decay 0.1 \
      --activation_checkpointing \
      --logging_steps 1 \
      --save_steps 5 \
      --max_train_steps 5 \
      --seed 42

echo "SMOKE DONE. Check output/mono_smoke/step_5/"
