#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_mstts_v0c_redecode_smoke
#PBS -j oe

# Re-decode existing smoke tokens after the L=A R=B channel-swap fix in
# tools/decode_tokens.py:decode_audio. The 50 .npy tokens already exist;
# this just regenerates the wavs with the corrected channel ordering.

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

OUT_DIR="$PWD/output/mstts_v0c_synth/0_50"
# Save old wavs for comparison, then re-decode fresh.
if [[ -d "$OUT_DIR/decoded_audio" ]]; then
    mv "$OUT_DIR/decoded_audio" "$OUT_DIR/decoded_audio_pre_swap"
fi
mkdir -p "$OUT_DIR/decoded_audio"

uv run -m tools.decode_tokens \
    --tokens_dir "$OUT_DIR/generated_tokens" \
    --output_dir "$OUT_DIR/decoded_audio" \
    --num_workers 1 \
  > "$OUT_DIR/redecode_$PBS_JOBID.log" 2>&1

echo "DONE. Fresh wavs (L=A, R=B) in $OUT_DIR/decoded_audio"
echo "Old wavs (L=B, R=A) preserved in $OUT_DIR/decoded_audio_pre_swap for comparison"
