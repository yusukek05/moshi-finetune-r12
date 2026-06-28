#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N 0162_clean_v1.1_firered
#PBS -j oe

# clean_moshi for the FireRed-synth v1.1 experiment so it is servable by
# `moshi.server` (dep_q 16 -> 8, user-stream depformer modules removed).
#   in : output/v1.1_zoom1_plus_synthdialogue/step_11683_fp32   (training, dep_q=16)
#   out: output/v1.1_zoom1_plus_synthdialogue/step_11683_cleaned (serve,    dep_q=8)
# Then copies the mimi + text tokenizer next to it so upload_to_hf.py bundles
# the full moshi.server payload.

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

uv sync --python 3.12
export NO_TORCH_COMPILE=1

FT_DIR="output/v1.1_zoom1_plus_synthdialogue/step_11683_fp32"
CLEAN_DIR="output/v1.1_zoom1_plus_synthdialogue/step_11683_cleaned"

uv run -m tools.clean_moshi \
    --moshi_ft_dir "$FT_DIR" \
    --save_dir "$CLEAN_DIR" \
    --model_dtype bfloat16 \
    --remove_modules_for_user_stream

# bundle the mimi codec + text tokenizer (copy from the existing v1.1 release)
for f in tokenizer-e351c8d8-checkpoint125.safetensors tokenizer_spm_32k_3.model; do
    [ -f "$CLEAN_DIR/$f" ] || cp -v "output/v1.1_release/$f" "$CLEAN_DIR/$f"
done

echo "=== cleaned bundle ==="
ls -la "$CLEAN_DIR/"
echo "DONE: $CLEAN_DIR (dep_q=8). Next: upload_to_hf.py on the login node."
