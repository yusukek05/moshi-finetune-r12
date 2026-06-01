#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_mono_v0d_v3_to_fp32
#PBS -j oe

# Consolidate mono v0d_v3 Stage 1 FINAL ZeRO shards into a single fp32
# safetensors. v0d_v3 = full 0178 recipe (eff bs 512, lr 3e-4, Reazon full,
# 3 epochs) with LaboroTV swapped for ccaudio_v2. The final step number is
# epoch-derived (~14334/14335), so auto-detect the highest numeric step_N dir
# and emit a stable step_final_fp32 the downstream scripts can reference.

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

run_dir="output/mono_jchat_reazon_ccaudio_v2_v3"

# Highest-numbered step_N dir that is a real shard dir (has pytorch_model/), not an *_fp32.
# NOTE: strip everything up to the trailing "step_" before sorting numerically —
# the run_dir name itself contains underscores, so `sort -t_ -kN` on full paths
# splits on the wrong field (earlier bug picked step_8000 instead of step_14334).
num=$(ls -d "$run_dir"/step_* 2>/dev/null \
       | grep -E 'step_[0-9]+$' \
       | sed 's#.*/step_##' \
       | sort -n \
       | tail -1)
step="step_${num}"
echo "Detected final Stage 1 checkpoint: $step"

uv run -m tools.zero_to_fp32 \
    "$run_dir/$step" \
    "$run_dir/step_final_fp32" \
    --moshi_lm_kwargs_path init_models/moshika-single_streams-float32/moshi_lm_kwargs.json

echo "DONE: $run_dir/step_final_fp32 (mono v0d_v3 Stage 1 final, from $step)"
