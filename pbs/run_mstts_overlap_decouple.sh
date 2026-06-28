#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=02:00:00
#PBS -N 0162_mstts_overlap_decouple
#PBS -j oe
#PBS -o logs/
# 重なりと明瞭度のデカップリング: audio_temp(重なり源)を 0.9/1.0 に保ちつつ
#   text_temperature を下げ(台本忠実=明瞭)+ top_p(nucleus)で崩れ裾を切る。
#   同一20対話(前ジョブの input を再利用)で overlap×CER を測り、
#   「0.9並みの重なりを より低CERで」出せる設定を探す。
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
BASE=output/mstts_overlap_decouple
INPUT=output/mstts_overlap_fine/input    # 前ジョブの同一20対話を再利用
WX=/groups/gcg51557/experiments/0386_dialogue_model/.venv_whisperx/bin/python
CER=/groups/gcg51557/experiments/0386_dialogue_model/scripts/mstts_overlap_cer.py

[ -d "$INPUT" ] || { echo "FATAL: missing input $INPUT (run run_mstts_overlap_fine.sh first)"; exit 1; }
echo "input dialogues: $(ls $INPUT/*.json | wc -l)"

# config = "audio_temp text_temp top_p"
CONFIGS=(
  "0.9 0.30 0.0"
  "0.9 0.55 0.9"
  "0.9 0.30 0.9"
  "1.0 0.30 0.0"
  "1.0 0.55 0.9"
  "1.0 0.30 0.9"
)

for cfg in "${CONFIGS[@]}"; do
  read -r AT TT TP <<< "$cfg"
  TAG="at${AT}_tt${TT}_tp${TP}"
  OUT="$BASE/$TAG"; mkdir -p "$OUT"
  echo "=== gen $TAG ==="
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
      --text_temperature "$TT" --audio_temperature "$AT" --top_k 0 --top_p "$TP" --seed 1
  cd ..
  uv run -m tools.decode_tokens --tokens_dir "$OUT/generated_tokens" \
      --output_dir "$OUT/decoded_audio" --num_workers 1
  echo "=== overlap+CER $TAG ==="
  $WX "$CER" --decoded_dir "$OUT/decoded_audio" --text_chat_dir "$INPUT" \
      --out_json "$OUT/overlap_cer.json" --device cuda
done
echo "ALL DONE"
