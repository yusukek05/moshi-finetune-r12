#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=01:00:00
#PBS -N 0162_mstts_overlap_tempsweep
#PBS -j oe
#PBS -o logs/
# L2確証実験: mstts(v0c zoom1)が audio_temperature を上げると自発相槌/重なりを
#   出すか。同一5対話を audio_temp={0.6(v0c既定),0.9,1.2} で生成→デコード。
#   後で decoded_audio の重なり率を Zoom1(9.8%)/旧v0c(~0%) と同手法で比較する。
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
BASE=output/mstts_overlap_tempsweep

for AT in 0.6 0.9 1.2; do
  OUT="$BASE/at${AT}"
  mkdir -p "$OUT"
  echo "=== audio_temperature=$AT ==="
  cd mstts
  mpirun -n 1 --map-by ppr:1:node:PE=1 --bind-to none --oversubscribe \
    -x NCCL_NVLS_ENABLE -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING \
    -x NO_TORCH_COMPILE -x HF_HUB_OFFLINE -x LD_LIBRARY_PATH -x PATH \
    uv run python run_ms_tts.py \
        --launcher mpi \
        --output_dir "../$OUT" \
        --text_chat_data_dir inference_inputs/text_chat \
        --model_dir "../$MODEL" \
        --model_dtype bfloat16 \
        --text_tokenizer_repo rinna/japanese-gpt2-medium \
        --text_tokenizer_name spiece.model \
        --prompt_streams_path inference_inputs/prompt_streams/standing_text_padded.npy \
        --per_device_batch_size 4 \
        --max_generation_length 1000 \
        --use_sampling \
        --text_temperature 0.55 \
        --audio_temperature "$AT" \
        --top_k 0 \
        --seed 1
  cd ..
  uv run -m tools.decode_tokens \
      --tokens_dir "$OUT/generated_tokens" \
      --output_dir "$OUT/decoded_audio" \
      --num_workers 1
  echo "=== done at=$AT : $(ls $OUT/decoded_audio/*.wav 2>/dev/null | wc -l) wav ==="
done
echo "ALL DONE"
