#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=4
#PBS -l walltime=02:00:00
#PBS -N 0162_dpo_train
#PBS -j oe
#PBS -o logs/

# DPO stage 3 pilot: (1) cache reference log-probs from the initial policy,
# (2) run the deepspeed ZeRO-3 DPO training loop on the mined preference pairs.
# Small by design — validate the loop runs and the loss/accuracy move sanely.
# SUCCESS is judged later by human A/B, never by this reward (anti-hacking).
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
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
unset NCCL_ASYNC_ERROR_HANDLING
unset NCCL_IB_DISABLE
ulimit -l unlimited

POLICY_DIR="output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32"
EVAL_DATA="processed_data/llmjp-zoom1/test-001-of-001.parquet"
PAIRS="output/dpo_scaled/dpo_pairs.jsonl"
OUT="output/dpo_scaled_run"
REF="${OUT}/ref_logprobs.json"
DS_CONF="ds_configs/zero3-bf16-warmlr-act_ckpt.json"
mkdir -p "${OUT}"

# --- (1) cache reference log-probs (single GPU, no training); skip if cached ---
if [ -f "${REF}" ]; then
  echo "===== dump_ref: reuse cached ${REF} ====="
else
  echo "===== dump_ref ====="
  uv run python dpo.py --dump_ref --ref_out "${REF}" \
      --model_dir "${POLICY_DIR}" --eval_data_files "${EVAL_DATA}" \
      --pairs_jsonl "${PAIRS}" --prompt_length 125 --example_length 375
fi

# --- (2) deepspeed DPO training ---
GPUS_PER_NODE=4
HOSTFILE="/tmp/hostfile_dpo_$PBS_JOBID"   # node-local; avoids writes to a full /groups
uniq "$PBS_NODEFILE" | awk -v s=$GPUS_PER_NODE '{print $0" slots="s}' > "$HOSTFILE"
NNODES=$(wc -l < "$HOSTFILE")
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
echo "WORLD_SIZE=${WORLD_SIZE}"

echo "===== dpo train ====="
mpirun \
  -n ${WORLD_SIZE} \
  --map-by ppr:${GPUS_PER_NODE}:node:PE=1 \
  --bind-to none \
  --oversubscribe \
  -x NCCL_NVLS_ENABLE -x NO_TORCH_COMPILE -x TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC \
  -x NCCL_DEBUG -x HF_HUB_OFFLINE -x LD_LIBRARY_PATH -x PATH \
  uv run python dpo.py \
      --launcher mpi --use_deepspeed --deepspeed_config_file "${DS_CONF}" \
      --model_dir "${POLICY_DIR}" --eval_data_files "${EVAL_DATA}" \
      --pairs_jsonl "${PAIRS}" --ref_out "${REF}" --output_dir "${OUT}" \
      --prompt_length 125 --example_length 375 \
      --model_dtype bfloat16 --beta 0.3 \
      --parameters_to_finetune all \
      --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
      --num_train_epochs 3 --save_steps 8 \
      --tempformer_learning_rate 1e-6 --depformer_learning_rate 1e-6 \
      --num_warmup_steps 3 --activation_checkpointing --logging_steps 1
echo "DONE"
