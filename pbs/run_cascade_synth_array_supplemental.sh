#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=06:00:00
#PBS -N 0162_cascade_synth_sup
#PBS -j oe
#PBS -J 1-10

# Supplemental run: complete any dialogues that the first array
# (1816324[]/1816328[]) skipped because of walltime timeout or the early
# glob bug. Uses --skip-if-audio-exists-under so already-done dialogues
# are NOT redone.
#
# Walltime extended to 6h (vs the original 4h) to safely fit a full
# 5000-dialogue slot at observed pace (~5:10 wall).
#
# Output dir is shared with the original array: output/cascade_synth_full/<slot>/

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

SLOT_INPUT_DIR="output/text_corpora/slices/${SLOT}/text_chat"
TMP_JSONL=".tmp/cascade_synth_sup/${SLOT}/dialogues.jsonl"
OUT_DIR="output/cascade_synth_full/${SLOT}"

mkdir -p "$(dirname "$TMP_JSONL")" "$OUT_DIR"

# 1) Build a jsonl that EXCLUDES dialogues already having audio.wav
uv run python tools/text_chat_dir_to_jsonl.py \
    --input-dir "$SLOT_INPUT_DIR" \
    --output-jsonl "$TMP_JSONL" \
    --skip-if-audio-exists-under "$OUT_DIR"

# Short-circuit if nothing to do
N_TODO=$(wc -l < "$TMP_JSONL")
echo "remaining dialogues for slot $SLOT: $N_TODO"
if [ "$N_TODO" -eq 0 ]; then
    echo "slot already complete; nothing to do"
    exit 0
fi

# 2) Cascade synth on the remaining rows
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
