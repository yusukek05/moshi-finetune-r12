#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=03:00:00
#PBS -N 0162_cascade_to_v1_parquet
#PBS -j oe
#PBS -J 1-10

# Tokenize cascade synthesis output (audio.wav + turns.json) into v1 parquet
# training format. One PBS array job per slot, matching cascade_synth_array.
#
# Input:  output/cascade_synth_full/<slot>/<dialogue_id>/{audio.wav, turns.json}
# Output: output/cascade_synth_v1_parquet/synth_<slot>-001-of-NNN.parquet
#
# At ~5000 dialogues per slot, Mimi tokenize @ ~5-10 d/s on H100 ≈ 10-20 min.
# Walltime 3h is generous.

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

CASCADE_DIR="output/cascade_synth_full/${SLOT}"
OUT_DIR="output/cascade_synth_v1_parquet"
mkdir -p "$OUT_DIR"

if [ ! -d "$CASCADE_DIR" ]; then
    echo "FATAL: cascade dir missing: $CASCADE_DIR"
    exit 1
fi

uv run python -m mstts.data_prep.cascade_output_to_v1_parquet \
    --cascade-dir "$CASCADE_DIR" \
    --output-prefix "${OUT_DIR}/synth_${SLOT}" \
    --num-examples-per-parquet 5000

echo "DONE: SLOT=$SLOT"
ls -la "$OUT_DIR/synth_${SLOT}"* 2>/dev/null
