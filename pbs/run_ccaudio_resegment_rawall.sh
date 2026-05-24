#!/bin/bash -l
#PBS -P gca50130
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N 0162_ccaudio_rawall
#PBS -j oe

# Re-segment + re-transcribe the FULL ccaudio raw_all corpus (~23,685h).
# One recording.*.tar (~100 podcast episodes) per array task.
#   qsub -v IDX=0 pbs/run_ccaudio_resegment_rawall.sh   -> single tar (smoke)
#   qsub -J 0-593 pbs/run_ccaudio_resegment_rawall.sh   -> all 594 tars
# Task index -> line (IDX+1) of tar_manifest.txt -> that tar -> shard-IDX.parquet
#
# Queue: rt_HG (1 GPU + 16 CPU, 168h walltime). Previously R9920261000 / rt_HF
# but that booked an entire 8-GPU H200 node per single-GPU task (7/8 wasted),
# and the reserved-queue 4h walltime kept killing long-tail tars. rt_HG is
# sized to exactly what this pipeline uses.
#
# Project: gca50130 (not gcg51557). gcg51557 ran out of points so non-reserved
# queues (rt_HG/rt_HC/rt_HF) reject its qsubs; gca50130 still has budget.

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

# Make the GPU visible (rt_HG hands it via cgroups but Plotly-side libs sometimes
# need the explicit env var). Harmless on rt_HF too.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

IDX="${PBS_ARRAY_INDEX:-${IDX:-0}}"
MANIFEST=processed_data/ccaudio_v2_full/tar_manifest.txt
TAR_PATH=$(sed -n "$((IDX + 1))p" "$MANIFEST")
echo "IDX=$IDX TAR_PATH=$TAR_PATH"
[ -n "$TAR_PATH" ] || { echo "no tar for IDX=$IDX"; exit 1; }

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

export NO_TORCH_COMPILE=1

uv run --with faster-whisper -m tools.ccaudio_resegment_transcribe \
    --tar_path "$TAR_PATH" \
    --shard_id "$IDX" \
    --output_dir processed_data/ccaudio_v2_full/rinna_gpt2-kyutai_mimi-q16 \
    --overwrite

echo "DONE: processed_data/ccaudio_v2_full/rinna_gpt2-kyutai_mimi-q16/shard-$(printf '%05d' "$IDX").parquet"
