#!/bin/bash -l
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=03:00:00
#PBS -N 0162_personaplex_gen_flip
#PBS -j oe
#PBS -o logs/
#
# PersonaPlex GENERATION-based flip (gold metric): prime prompt prefix only, generate a
# fresh dialogue, classify the agent text track's formality. Uses a SEEN paraphrase (idx2)
# since unseen phrasings do not generalise (see NLL diag). fp32 must already exist.
#
# Submit smoke: qsub -q rt_HF -v RTYPE=rt_HF,USE_SSH=1,N=6  pbs/run_personaplex_gen_flip.sh
# Submit full : qsub -q rt_HF -v RTYPE=rt_HF,USE_SSH=1,N=100 pbs/run_personaplex_gen_flip.sh
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

OUT="${OUT:-output/personaplex_100h}"
STEP="${STEP:-step_675}"
CARRIERS="${CARRIERS:-processed_data/persona_100h/heldout_indist-001-of-001.parquet}"
N="${N:-100}"
PARAPHRASE_IDX="${PARAPHRASE_IDX:-2}"
BASE_KWARGS="${BASE_KWARGS:-output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32/moshi_lm_kwargs.json}"
RINNA=/home/acg17145sv/.cache/huggingface/hub/models--rinna--japanese-gpt2-medium/snapshots/8ce2399c33e99013a593ea9389378fd86662b9c7/spiece.model

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
export NO_TORCH_COMPILE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

FP32="$OUT/${STEP}_fp32"
if [ ! -f "$FP32/model.safetensors" ]; then
  echo "consolidating $OUT/$STEP -> $FP32"
  uv run -m tools.zero_to_fp32 "$OUT/$STEP" "$FP32" --moshi_lm_kwargs_path "$BASE_KWARGS"
fi

uv run python mstts/data_prep/persona_gen_flip.py \
    --model_dir "$FP32" --carriers "$CARRIERS" --tokenizer "$RINNA" \
    --n "$N" --paraphrase-idx "$PARAPHRASE_IDX" \
    --dump "$OUT/gen_flip_${STEP}_n${N}.jsonl"
echo "=== gen flip done ($STEP, n=$N) ==="
