#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=02:00:00
#PBS -N 0162_mstts_bc_inject
#PBS -j oe
#PBS -o logs/
# L1: 相槌注入×低温。相手ターン途中に相槌を差し込んだ入力(事前生成済
#   output/mstts_backchannel_inject/input_inj{0.6,1.0})を 低温(audio0.6/0.8)で生成。
#   狙い: 低温=明瞭を保ったまま、挿入相槌で重なりを出す。
#   判定: 重なり率が非注入(at0.6=0.2%/at0.8=1.1%)より明確に上がり、CERが低温並みなら成功
#         = モデルが挿入相槌をAに"被せて"鳴らしている。重なりが上がらなければ逐次micro-turn。
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
BASE=output/mstts_backchannel_inject
WX=/groups/gcg51557/experiments/0386_dialogue_model/.venv_whisperx/bin/python
CER=/groups/gcg51557/experiments/0386_dialogue_model/scripts/mstts_overlap_cer.py

# config = "input_subdir audio_temp"
CONFIGS=(
  "input_inj0.6 0.6"
  "input_inj1.0 0.6"
  "input_inj0.6 0.8"
)

for cfg in "${CONFIGS[@]}"; do
  read -r INSUB AT <<< "$cfg"
  INPUT="$BASE/$INSUB"
  [ -d "$INPUT" ] || { echo "FATAL: missing $INPUT"; exit 1; }
  TAG="${INSUB}_at${AT}"
  OUT="$BASE/gen_$TAG"; mkdir -p "$OUT"
  echo "=== gen $TAG (input=$(ls $INPUT/*.json | wc -l) dlg) ==="
  cd mstts
  mpirun -n 1 --map-by ppr:1:node:PE=1 --bind-to none --oversubscribe \
    -x NCCL_NVLS_ENABLE -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING \
    -x NO_TORCH_COMPILE -x HF_HUB_OFFLINE -x LD_LIBRARY_PATH -x PATH \
    uv run python run_ms_tts.py --launcher mpi \
      --output_dir "../$OUT" --text_chat_data_dir "../$INPUT" \
      --model_dir "../$MODEL" --model_dtype bfloat16 \
      --text_tokenizer_repo rinna/japanese-gpt2-medium --text_tokenizer_name spiece.model \
      --prompt_streams_path inference_inputs/prompt_streams/standing_text_padded.npy \
      --per_device_batch_size 4 --max_generation_length 1200 --use_sampling \
      --text_temperature 0.55 --audio_temperature "$AT" --top_k 0 --seed 1
  cd ..
  uv run -m tools.decode_tokens --tokens_dir "$OUT/generated_tokens" \
      --output_dir "$OUT/decoded_audio" --num_workers 1
  echo "=== overlap+CER $TAG ==="
  $WX "$CER" --decoded_dir "$OUT/decoded_audio" --text_chat_dir "$INPUT" \
      --out_json "$OUT/overlap_cer.json" --device cuda
done
echo "ALL DONE"
