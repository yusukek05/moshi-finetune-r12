#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=8
#PBS -l walltime=06:00:00
#PBS -N 0162_train_v1_plus_v0csynth
#PBS -j oe

# Phase 2.1 (ICASSP paper, downstream utility):
# Continue-train llm-jp-moshi-v1 (public, CC-BY-4.0) on the v0c mstts synth
# corpus (46,266 dialogues, 527 h, JMultiWOZ + RealPersonaChat seeds) to test
# whether single-LM joint speech synthesis output can usefully augment a
# spoken-dialogue LM's training data.
#
#   - Base: output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_7epochs_1node_exp/step_9282_fp32
#           (= the actual local fp32 ckpt that was published as HF abePclWaseda/llm-jp-moshi-v1
#            bf16, per README curriculum: J-CHAT → Zoom1 7ep, text_padding=0.5)
#   - Add data: output/mstts_v0c_synth_parquet/synth_*.parquet (~2,891 steps/epoch @ eff bs 16)
#   - 1 node × 8 GPU, eff bs 16, 1 epoch (~3-4 h walltime)

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

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
unset NCCL_ASYNC_ERROR_HANDLING
export NCCL_IB_HCA=mlx5_0
unset NCCL_IB_DISABLE
ulimit -l unlimited

GPUS_PER_NODE=8
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_v1synth_$PBS_JOBID
NNODES=$(wc -l < hostfile_v1synth_$PBS_JOBID)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

mpirun \
  -n  ${WORLD_SIZE} \
  --map-by ppr:${GPUS_PER_NODE}:node:PE=1 \
  --bind-to none \
  --oversubscribe \
  -x NCCL_NVLS_ENABLE \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING -x NCCL_IB_HCA -x NCCL_IB_DISABLE \
  -x LD_LIBRARY_PATH -x PATH \
  uv run finetune.py \
      --audio_loss_weight_when_text_pad 1.0 \
      --launcher mpi \
      --use_deepspeed \
      --tempformer_learning_rate 2e-6 \
      --depformer_learning_rate 4e-6 \
      --deepspeed_config_file ds_configs/zero3-fp16-act_ckpt.json \
      --output_dir output/v1_plus_v0csynth \
      --train_data_files "output/mstts_v0c_synth_parquet/synth_*.parquet" \
      --model_dir output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_7epochs_1node_exp/step_9282_fp32 \
      --model_dtype bfloat16 \
      --model_user_stream \
      --max_length 2048 \
      --min_length 128 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 2 \
      --num_warmup_steps 0 \
      --activation_checkpointing \
      --logging_steps 10 \
      --report_to wandb \
      --project_name v1_plus_v0csynth \
      --save_steps 1000

echo "DONE: checkpoints under output/v1_plus_v0csynth/"
