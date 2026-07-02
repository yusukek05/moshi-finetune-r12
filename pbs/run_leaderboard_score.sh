#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=03:00:00
#PBS -N 0162_leaderboard_score
#PBS -j oe
#PBS -o logs/
#
# Score the v1-lineage continuation models with llm-jp-moshi-eval (UTMOS + LLM-Judge)
# for the GitHub Pages leaderboard. CER is taken from the prior validated lineage json.
# Common 50-dialogue set; per-model generated_wavs are flat (0.wav, 10.wav, ...).
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"; mkdir -p logs

EVAL=/home/acg17145sv/projects/llm-jp-moshi-eval
LIN=output/v1_lineage_eval
OUTROOT=$EVAL/results/leaderboard
# MODELS overridable via `qsub -v ...,MODELS=a:b` (colon-separated; commas are taken by
# qsub's own -v parser). Default = full 5-model lineage.
MODELS="${MODELS:-v1 v1.1_candidate v1.2_candidate v1.1_firered_synth all_staged}"
MODELS="$(echo "$MODELS" | tr ':,' '  ')"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
export CUDA_VISIBLE_DEVICES=0

cd "$EVAL"
uv venv --python 3.12 .venv 2>/dev/null || true
export VIRTUAL_ENV="$EVAL/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
uv pip install -e ".[utmos]"
nvidia-smi || true

cd "$PBS_O_WORKDIR"
for m in $MODELS; do
  WD="$LIN/$m/generated_wavs"
  [ -d "$WD" ] || { echo "SKIP missing $WD"; continue; }
  echo "############ scoring $m ($(ls "$WD"/*.wav | wc -l) wavs) ############"
  python -m llm_jp_moshi_eval.cli run \
      --input_dir "$WD" \
      --metrics utmos,judge \
      --judge_model llm-jp/llm-jp-3.1-13b-instruct4 \
      --out_dir "$OUTROOT/$m" || echo "score failed: $m"
done
echo "=== LEADERBOARD SCORING DONE -> $OUTROOT ==="
for m in $MODELS; do echo "--- $m ---"; cat "$OUTROOT/$m/summary.json" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);r=d['results'];print('utmos',r.get('utmos',{}).get('overall',{}).get('mean'),'judge_overall',r.get('judge',{}).get('axes',{}).get('overall',{}).get('mean'))" 2>/dev/null || true; done