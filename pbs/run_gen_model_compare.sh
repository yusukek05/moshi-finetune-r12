#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=03:00:00
#PBS -N 0162_gen_model_compare
#PBS -j oe
#PBS -o logs/
#
# Compare dialogue-TEXT generators for naturalness (for future FireRed-synth corpus):
# LLM-jp-4-8b (current) vs Gemma-2-9b-it vs GLM-4-9B-0414. Same style-matched few-shot
# prompts (gen_persona_dialogues.py, --model swappable). Small run for a naturalness
# spot-check + formality-control confusion matrix per model.
#   出力: <OUT>/<tag>/*.json + 各モデルの formality confusion。目視 + classify で自然さ比較。
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"

REPO0162=/groups/gcg51557/experiments/0162_dialogue_model/moshi-finetune
REPO0386=/groups/gcg51557/experiments/0386_dialogue_model/FireRedTTS2
GEN="$REPO0162/mstts/data_prep/gen_persona_dialogues.py"
OUTROOT="${OUTROOT:-$REPO0162/output/gen_model_compare}"
NPS="${NPS:-5}"; TOPICS="${TOPICS:-hobby,food}"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
export VIRTUAL_ENV="$REPO0386/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
nvidia-smi || true
mkdir -p "$OUTROOT" "$REPO0162/logs"

# tag:model_id  (GLM/Gemma は native transformers; Gemma は gated=要ライセンス承諾済トークン)
declare -A MODELS=(
  [llmjp4]="llm-jp/llm-jp-4-8b-instruct"
  [gemma2]="google/gemma-2-9b-it"
  [glm4]="zai-org/GLM-4-9B-0414"
)
for tag in llmjp4 gemma2 glm4; do
  mid="${MODELS[$tag]}"
  echo "############ generate with $tag = $mid ############"
  uv run python "$GEN" --model "$mid" --out_dir "$OUTROOT/$tag" \
      --styles polite,casual --n_per_style "$NPS" --topics "$TOPICS" --seed 0 \
      2>&1 | tail -40 || echo "MODEL FAILED: $tag ($mid)"
done

echo "=== COMPARE DONE ==="
for tag in llmjp4 gemma2 glm4; do
  echo "--- $tag : $(ls "$OUTROOT/$tag"/*.json 2>/dev/null | wc -l) dialogues ---"
done