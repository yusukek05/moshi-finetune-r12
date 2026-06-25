#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=04:00:00
#PBS -N 0162_cascade_synth_full
#PBS -j oe
#PBS -J 1-10

# Cascade synthesis at scale: process all 46,266 dialogues across 10 slots
# (1 PBS array job per slot). Mirrors the v0c synth slot scheme so that
# v1+cascadesynth can be compared one-to-one with v1+v0csynth.
#
# Input per slot: output/text_corpora/slices/${SLOT}/text_chat/dialogue_*.json
# Output per slot: output/cascade_synth_full/${SLOT}/<dialogue_id>/{audio.wav, turns.json, timing.json}
#
# Pipeline: Style-Bert-VITS2 (jvnv-llmjp_zoom1_0001_{l,r}) per turn + 200ms heuristic merge.
# Expected: ~1.9h per slot at RTF~=0.018, single GPU.
#
# After this completes, run pbs/run_cascade_to_v1_parquet_array.sh to tokenize → parquet.

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID  ARRAY_IDX=$PBS_ARRAY_INDEX"
cd "$PBS_O_WORKDIR"

SLOTS=(0_5000 5000_10000 10000_15000 15000_20000 20000_25000 25000_30000 30000_35000 35000_40000 40000_45000 45000_50000)
SLOT="${SLOTS[$((PBS_ARRAY_INDEX - 1))]}"
echo "SLOT=$SLOT"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export NO_TORCH_COMPILE=1

# Paths
SLOT_INPUT_DIR="output/text_corpora/slices/${SLOT}/text_chat"
TMP_JSONL=".tmp/cascade_synth/${SLOT}/dialogues.jsonl"
OUT_DIR="output/cascade_synth_full/${SLOT}"

# 1) Convert per-file JSONs to jsonl
mkdir -p "$(dirname "$TMP_JSONL")"
uv run python tools/text_chat_dir_to_jsonl.py \
    --input-dir "$SLOT_INPUT_DIR" \
    --output-jsonl "$TMP_JSONL"

# 2) Cascade synth (Style-Bert-VITS2 + heuristic merge)
mkdir -p "$OUT_DIR"
CASCADE_SCRIPT=/home/acg17145sv/projects/icassp-2027-mstts/scripts/cascade_synth.py
CASCADE_DEPS=/home/acg17145sv/projects/icassp-2027-mstts/scripts/deps_cascade.txt

uv run --no-project --python 3.12 --with-requirements "$CASCADE_DEPS" \
    python "$CASCADE_SCRIPT" \
        --input_jsonl "$TMP_JSONL" \
        --output_dir "$OUT_DIR" \
        --merge heuristic \
        --gap_ms 200 \
        --device cuda \
        --seed 42

# 3) Summary
N_DIRS=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
N_WAVS=$(find "$OUT_DIR" -mindepth 2 -name "audio.wav" | wc -l)
echo "DONE: SLOT=$SLOT, dirs=$N_DIRS, wavs=$N_WAVS"
du -sh "$OUT_DIR"
