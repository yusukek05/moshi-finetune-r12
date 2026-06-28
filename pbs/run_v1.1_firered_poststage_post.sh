#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N 0162_v1.1_fr_poststage_post
#PBS -j oe

# Post-process the FireRed post-stage run for the Arena:
#   raw step_N -> step_N_fp32 (dep_q=16) -> step_N_cleaned (dep_q=8, moshi.server用)
#   + mimi / text tokenizer を同梱 (他8モデルと同形式)。
# 完了後ログインノードで upload_to_hf.py で abePclWaseda/llm-jp-moshi-v1.1-firered-poststage へ。

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export NO_TORCH_COMPILE=1

RUN="output/v1.1_firered_poststage"
KWARGS="output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32/moshi_lm_kwargs.json"

STEP=$(ls -d "$RUN"/step_* 2>/dev/null | grep -oE 'step_[0-9]+$' | sort -t_ -k2 -n | tail -1 || true)
[ -n "$STEP" ] || { echo "ERROR: no step_* in $RUN" >&2; exit 1; }
echo "final step = $STEP"

FP32="$RUN/${STEP}_fp32"
CLEAN="$RUN/${STEP}_cleaned"

# 1) fp32 consolidation (dep_q=16)
if [ ! -f "$FP32/model.safetensors" ]; then
    uv run -m tools.zero_to_fp32 "$RUN/$STEP" "$FP32" --moshi_lm_kwargs_path "$KWARGS"
fi

# 2) clean -> dep_q=8 (moshi.server / Arena 用)
if [ ! -f "$CLEAN/model.safetensors" ]; then
    uv run -m tools.clean_moshi \
        --moshi_ft_dir "$FP32" \
        --save_dir "$CLEAN" \
        --model_dtype bfloat16 \
        --remove_modules_for_user_stream
fi

# 3) bundle mimi + text tokenizer
for f in tokenizer-e351c8d8-checkpoint125.safetensors tokenizer_spm_32k_3.model; do
    [ -f "$CLEAN/$f" ] || cp -v "output/v1.1_release/$f" "$CLEAN/$f"
done

echo "=== cleaned bundle ==="
ls -la "$CLEAN/"
echo "DONE: $CLEAN (dep_q=8). Next: upload_to_hf.py on login node."
