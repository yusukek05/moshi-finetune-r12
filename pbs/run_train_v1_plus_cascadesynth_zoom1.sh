#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=8
#PBS -l walltime=08:00:00
#PBS -N 0162_train_v1pluscascade_zoom1
#PBS -j oe

# Phase 2 follow-up: re-anchor v1+cascadesynth on Zoom1 (control arm of
# run_train_v1_plus_v0csynth_zoom1.sh — see rationale there; cascade CER was
# 116.5%, worst of the 3-way, with written-style text artifacts on top of the
# domain shift).
#
#   v1 (J-CHAT -> Zoom1) -> cascadesynth (step_2714) -> Zoom1 7ep  <- this job
#
# Recipe mirrors the original v1 Zoom1 FT exactly (tlr 2e-6, dlr 4e-6,
# eff bs 16 = 8 GPU x bs1 x ga2). Zoom1 train = 21,201 examples, so
# 7 epochs = 9,282 steps (~6.1h at 2.35 s/it). Auto-resumes from the latest
# raw step_N checkpoint if one exists, so a walltime kill just needs a
# resubmit.
#
# Next: qsub -W depend=afterok:<this> pbs/run_v1_plus_synth_zoom1_post.sh

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
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > hostfile_cascade_zoom1_$PBS_JOBID
NNODES=$(wc -l < hostfile_cascade_zoom1_$PBS_JOBID)
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

RESUME_ARGS=()
LATEST=$(ls -d output/v1_plus_cascadesynth_zoom1/step_* 2>/dev/null | grep -E 'step_[0-9]+$' | sort -t_ -k2 -n | tail -1 || true)
if [ -n "$LATEST" ] && [ -f "$LATEST/latest" ]; then
    echo "Resuming from $LATEST"
    RESUME_ARGS=(--resume_from_checkpoint "$LATEST")
fi

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
      --output_dir output/v1_plus_cascadesynth_zoom1 \
      --train_data_files processed_data/llmjp-zoom1/train-001-of-001.parquet \
      --model_dir output/v1_plus_cascadesynth/step_2714_fp32 \
      --model_dtype bfloat16 \
      --model_user_stream \
      --max_length 2048 \
      --min_length 128 \
      --num_train_epochs 7 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 2 \
      --num_warmup_steps 0 \
      --activation_checkpointing \
      --logging_steps 10 \
      --report_to wandb \
      --project_name v1_plus_cascadesynth_zoom1 \
      --save_steps 1000 \
      "${RESUME_ARGS[@]}"

echo "DONE: checkpoints under output/v1_plus_cascadesynth_zoom1/"
