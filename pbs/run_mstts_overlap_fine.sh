#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=03:00:00
#PBS -N 0162_mstts_overlap_fine
#PBS -j oe
#PBS -o logs/
# 細スイープ: mstts(v0c zoom1) を同一20対話で audio_temp={0.6,0.8,0.9,1.0,1.1,1.2} 生成
#   → デコード → 重なり率 + チャンネル別CER(明瞭度) を測定。
#   狙い: Zoom1並み(~10%)の重なりを CER劣化なしで出す温度を確定。試聴用wavも温度別に残す。
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
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
unset NCCL_ASYNC_ERROR_HANDLING

MODEL=output/mstts_v0c_zoom1_finetune_step1500/step_1500_fp32
BASE=output/mstts_overlap_fine
INPUT=$BASE/input
WX=/groups/gcg51557/experiments/0386_dialogue_model/.venv_whisperx/bin/python
CER=/groups/gcg51557/experiments/0386_dialogue_model/scripts/mstts_overlap_cer.py

# 入力20対話(jmultiwoz 先頭20)を symlink
mkdir -p "$INPUT"
for f in $(ls output/text_corpora/text_chat/jmultiwoz/*.json 2>/dev/null | head -20); do
  ln -sf "$(readlink -f "$f")" "$INPUT/$(basename "$f")"
done
echo "input dialogues: $(ls $INPUT/*.json | wc -l)"

for AT in 0.6 0.8 0.9 1.0 1.1 1.2; do
  OUT="$BASE/at${AT}"; mkdir -p "$OUT"
  echo "=== gen at=$AT ==="
  cd mstts
  mpirun -n 1 --map-by ppr:1:node:PE=1 --bind-to none --oversubscribe \
    -x NCCL_NVLS_ENABLE -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING \
    -x NO_TORCH_COMPILE -x HF_HUB_OFFLINE -x LD_LIBRARY_PATH -x PATH \
    uv run python run_ms_tts.py --launcher mpi \
      --output_dir "../$OUT" --text_chat_data_dir "../$INPUT" \
      --model_dir "../$MODEL" --model_dtype bfloat16 \
      --text_tokenizer_repo rinna/japanese-gpt2-medium --text_tokenizer_name spiece.model \
      --prompt_streams_path inference_inputs/prompt_streams/standing_text_padded.npy \
      --per_device_batch_size 4 --max_generation_length 1000 --use_sampling \
      --text_temperature 0.55 --audio_temperature "$AT" --top_k 0 --seed 1
  cd ..
  uv run -m tools.decode_tokens --tokens_dir "$OUT/generated_tokens" \
      --output_dir "$OUT/decoded_audio" --num_workers 1
  echo "=== overlap+CER at=$AT ==="
  $WX "$CER" --decoded_dir "$OUT/decoded_audio" --text_chat_dir "$INPUT" \
      --out_json "$OUT/overlap_cer.json" --device cuda
done
echo "ALL DONE"
