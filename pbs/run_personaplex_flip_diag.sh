#!/bin/bash -l
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=02:00:00
#PBS -N 0162_personaplex_flip_diag
#PBS -j oe
#PBS -o logs/
#
# PersonaPlex 100h flip DIAGNOSTIC: disentangle "did scale fix generalization?" from confounds.
#   The sweep used an OOD held-out (persona_gen: 5 topics, greeting-only) and idx0 (unseen prompt),
#   giving ~chance. This job consolidates STEP once, then evals several (held-out, paraphrase) combos:
#     1) in-dist held-out (442 CER-dropped diverse dialogues), idx0  = FAIR held-out-prompt test
#     2) in-dist held-out, idx2 (a SEEN training paraphrase)         = isolates prompt-phrasing gen
#     3) OOD held-out (persona_full_heldout), idx2 (SEEN paraphrase) = isolates distribution shift
#   Reading: (1)>>50 => in-dist generalizes (OOD was the confound). (1)~50 & (2)>>50 => prompt-phrasing
#   memorization. (1)&(2)~50 => the mechanism memorizes dialogues, not concept (scale won't fix).
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

OUT="${OUT:-output/personaplex_100h}"
STEP="${STEP:-step_675}"
INDIST="${INDIST:-processed_data/persona_100h/heldout_indist-001-of-001.parquet}"
OOD="${OOD:-processed_data/persona_poc/persona_full_heldout-001-of-001.parquet}"
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

run() { # heldout idx label
  echo "############ DIAG: $3 (heldout=$(basename "$1") idx=$2) ############"
  uv run python mstts/data_prep/persona_flip_eval.py \
      --model_dir "$FP32" --heldout "$1" --tokenizer "$RINNA" --paraphrase-idx "$2" \
      || echo "eval failed: $3"
}
run "$INDIST" 0 "in-dist_heldout_prompt"
run "$INDIST" 2 "in-dist_SEEN_prompt"
run "$OOD"    2 "OOD_SEEN_prompt"
echo "=== flip diag done ($STEP) ==="
