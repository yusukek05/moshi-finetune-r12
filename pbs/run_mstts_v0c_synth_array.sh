#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=64:ngpus=8
#PBS -l walltime=01:30:00
#PBS -N 0162_mstts_v0c_synth_array
#PBS -j oe

# Production array job: 8-GPU multi-process mstts synthesis.
#
# Each array slot processes SLICE_SIZE chunks of the merged
# JMultiWOZ + RPC text_chat dataset using all 8 GPUs of a single H100 node.
# Run as an array job:
#     qsub -J 0-9 -v RTYPE=rt_HF,SLICE_SIZE=5000 pbs/run_mstts_v0c_synth_array.sh
#
# Or as a single test slot (manual START/END override):
#     qsub -v RTYPE=rt_HF,START_IDX=5000,END_IDX=6000 pbs/run_mstts_v0c_synth_array.sh
#
# Assumes raw corpora + text_chat conversion have ALREADY been populated by a
# prior run of pbs/run_mstts_v0c_synth_text_corpora.sh (1758082 did this).
# This script keeps the same idempotent data-prep guards but will normally
# skip them.

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load hpcx/2.20
module load python/3.12/3.12.9

uv sync --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

export NO_TORCH_COMPILE=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
unset NCCL_ASYNC_ERROR_HANDLING

# --- Slice configuration ---------------------------------------------------
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-1000}
if [[ -n "${PBS_ARRAY_INDEX:-}" && -n "${SLICE_SIZE:-}" ]]; then
    # PBS array indices on this site are 1-based; convert to 0-based slice number.
    START_IDX=$(( (PBS_ARRAY_INDEX - 1) * SLICE_SIZE ))
    END_IDX=$((START_IDX + SLICE_SIZE))
fi
NUM_EXAMPLES=$((END_IDX - START_IDX))
echo "SLICE: START=$START_IDX END=$END_IDX (count=$NUM_EXAMPLES)"

# --- Paths -----------------------------------------------------------------
CORPORA_DIR="$PWD/output/text_corpora"
RAW_DIR="$CORPORA_DIR/raw"
TEXT_CHAT_DIR="$CORPORA_DIR/text_chat"
SLICE_DIR="$CORPORA_DIR/slices/${START_IDX}_${END_IDX}/text_chat"
OUT_DIR="$PWD/output/mstts_v0c_synth/${START_IDX}_${END_IDX}"
MODEL_DIR="$PWD/output/mstts_v0c_zoom1_finetune_step1500/step_1500_fp32"

mkdir -p "$RAW_DIR" "$TEXT_CHAT_DIR" "$SLICE_DIR" "$OUT_DIR"

# --- Step 1: Fetch raw corpora (idempotent; skipped on production runs) ----
JMULTIWOZ_ZIP="$RAW_DIR/JMultiWOZ_1.0.zip"
if [[ ! -f "$JMULTIWOZ_ZIP" ]]; then
    curl -sL -o "$JMULTIWOZ_ZIP" \
        "https://raw.githubusercontent.com/nu-dialogue/jmultiwoz/master/dataset/JMultiWOZ_1.0.zip"
fi
RPC_DIR="$RAW_DIR/real-persona-chat"
if [[ ! -d "$RPC_DIR" ]]; then
    git clone --depth 1 https://github.com/nu-dialogue/real-persona-chat.git "$RPC_DIR"
fi

# --- Step 2: Convert to mstts chunks (idempotent) --------------------------
JMULTIWOZ_OUT="$TEXT_CHAT_DIR/jmultiwoz"
RPC_OUT="$TEXT_CHAT_DIR/rpc"
if ! ls "$JMULTIWOZ_OUT"/*.json >/dev/null 2>&1; then
    uv run --no-project --python 3.12 python mstts/data_prep/jmultiwoz_to_mstts.py \
        --zip-path "$JMULTIWOZ_ZIP" \
        --out-dir "$JMULTIWOZ_OUT" \
        --splits train
fi
if ! ls "$RPC_OUT"/*.json >/dev/null 2>&1; then
    uv run --no-project --python 3.12 python mstts/data_prep/real_persona_chat_to_mstts.py \
        --dialogues-dir "$RPC_DIR/real_persona_chat/dialogues" \
        --out-dir "$RPC_OUT"
fi

# --- Step 3: Materialise slice [START_IDX, END_IDX) ------------------------
rm -rf "$SLICE_DIR"
mkdir -p "$SLICE_DIR"
{ ls "$JMULTIWOZ_OUT"/*.json; ls "$RPC_OUT"/*.json; } \
    | sort \
    | awk -v s="$START_IDX" -v e="$END_IDX" 'NR>s && NR<=e' \
    | while read -r f; do ln -s "$f" "$SLICE_DIR/$(basename "$f")"; done
echo "Slice contents: $(ls "$SLICE_DIR" | wc -l) files"

# --- Step 4: Run mstts inference on 8 GPUs ---------------------------------
cd mstts
mpirun \
  -n 8 \
  --map-by ppr:8:node:PE=8 \
  --bind-to none \
  --oversubscribe \
  -x NCCL_NVLS_ENABLE \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING \
  -x NO_TORCH_COMPILE \
  -x LD_LIBRARY_PATH -x PATH \
  uv run python run_ms_tts.py \
      --launcher mpi \
      --output_dir "$OUT_DIR" \
      --text_chat_data_dir "$SLICE_DIR" \
      --model_dir "$MODEL_DIR" \
      --model_dtype bfloat16 \
      --text_tokenizer_repo rinna/japanese-gpt2-medium \
      --text_tokenizer_name spiece.model \
      --prompt_streams_path inference_inputs/prompt_streams/standing_text_padded.npy \
      --per_device_batch_size 4 \
      --max_generation_length 1000 \
      --use_sampling \
      --text_temperature 0.55 \
      --audio_temperature 0.6 \
      --top_k 0 \
      --seed 1 \
  > "$OUT_DIR/infer_$PBS_JOBID.log" 2>&1

cd ..

# --- Step 5: Decode tokens to wavs (1 GPU is enough; decode is I/O light) --
mkdir -p "$OUT_DIR/decoded_audio"
uv run -m tools.decode_tokens \
    --tokens_dir "$OUT_DIR/generated_tokens" \
    --output_dir "$OUT_DIR/decoded_audio" \
    --num_workers 1 \
  > "$OUT_DIR/decode_$PBS_JOBID.log" 2>&1

# --- Step 6: Manifest -------------------------------------------------------
export OUT_DIR SLICE_DIR
uv run --no-project --python 3.12 python - <<'PYEOF'
import json, os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
slice_dir = Path(os.environ["SLICE_DIR"])
manifest = []
for wav in sorted((out_dir / "decoded_audio").glob("*.wav")):
    src_link = slice_dir / f"{wav.stem}.json"
    src_real = src_link.resolve() if src_link.exists() else None
    corpus = "jmultiwoz" if src_real and "jmultiwoz" in str(src_real) else (
        "rpc" if src_real and "rpc" in str(src_real) else "unknown"
    )
    manifest.append({
        "wav": str(wav),
        "source_json": str(src_real) if src_real else None,
        "corpus": corpus,
    })
(out_dir / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Manifest written: {len(manifest)} entries")
PYEOF

echo "DONE. Outputs in $OUT_DIR"
