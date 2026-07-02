#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=02:00:00
#PBS -N 0162_llm_jp_moshi_eval_smoke
#PBS -j oe
#PBS -o logs/
#
# End-to-end smoke of the new llm-jp-moshi-eval tool on real data:
# 20 persona FireRed stereo wavs (10 polite / 10 casual) -> utmos + asr_cer + judge.
# Validates the packaged tool + yields real UTMOS/CER/judge numbers on our synth.
# (NISQA skipped: needs NISQA_DIR; set it and add nisqa to --metrics to include.)
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"; mkdir -p logs

EVAL=/home/acg17145sv/projects/llm-jp-moshi-eval
SUB=/groups/gcg51557/experiments/0386_dialogue_model/output/dialogue_arena/persona_eval_smoke20
REFS=/groups/gcg51557/experiments/0386_dialogue_model/output/dialogue_arena/persona_eval_smoke20_refs.jsonl
OUT=$EVAL/results/persona_smoke20

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9
export CUDA_VISIBLE_DEVICES=0

EXTRAS="${EXTRAS:-}"          # e.g. "[utmos]" once utmos deps resolve on py3.12
METRICS="${METRICS:-asr_cer,judge}"
cd "$EVAL"
uv venv --python 3.12 .venv 2>/dev/null || true
export VIRTUAL_ENV="$EVAL/.venv"; export PATH="$VIRTUAL_ENV/bin:$PATH"
uv pip install -e ".${EXTRAS}"
nvidia-smi || true

python -m llm_jp_moshi_eval.cli run \
    --input_dir "$SUB" \
    --reference "$REFS" \
    --metrics "$METRICS" \
    --asr_model_size large-v3 \
    --judge_model llm-jp/llm-jp-3.1-13b-instruct4 \
    --out_dir "$OUT"

echo "=== SMOKE DONE -> $OUT ==="
echo "--- summary.json ---"; cat "$OUT/summary.json"