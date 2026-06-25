#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q R9920261000
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:30:00
#PBS -N 0162_spokenqa_pilot
#PBS -j oe

# Phase 2.2 PILOT (go/no-go): can the Zoom1-trained v1 dialogue model answer a
# spoken question at all? Build a 16-question QA prompt parquet (B=question,
# A=silence) from spoken-magpie-ja, run v1 continuation, decode, ASR channel A
# (the model's answer), and dump a transcript table for eyeballing.
#
# If v1 produces on-topic answers -> scale to 5 models + LLM-judge.
# If it just backchannels/chats -> spoken-QA is the wrong probe for these
# casual-dialogue models; report and rest Part 2 on self-consistency+prosody.

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
export ACCELERATE_DISTRIBUTED_TYPE=gloo
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

QDIR=phase2_spokenqa/pilot_questions
PARQUET=phase2_spokenqa/pilot_prompts.parquet
V1_DIR="output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_7epochs_1node_exp/step_9282_fp32"
OUT=output/spokenqa_pilot/v1
PROMPT_FRAMES=150          # 12 s question window
GEN_FRAMES=250             # 20 s answer

echo "===== [1/4] build QA prompt parquet ====="
uv run python phase2_spokenqa/build_qa_prompt_parquet.py \
    --questions-dir "$QDIR" \
    --output-parquet "$PARQUET" \
    --prompt-frames "$PROMPT_FRAMES"

echo "===== [2/4] v1 continuation (A = answer) ====="
uv run accelerate launch --num_machines 1 --num_processes 1 \
    generate.py \
        --output_dir "$OUT" \
        --model_dir "$V1_DIR" \
        --eval_data_files "$PARQUET" \
        --moshi_speakers A \
        --prompt_length "$PROMPT_FRAMES" \
        --generation_length "$GEN_FRAMES" \
        --example_length $((PROMPT_FRAMES + GEN_FRAMES)) \
        --temperature 0.8 \
        --num_examples 16 \
        --seed 42

echo "===== [3/4] decode tokens -> wav (L=A answer, R=B question) ====="
uv run -m tools.decode_tokens \
    --tokens_dir "$OUT/generated_tokens" \
    --output_dir "$OUT/generated_wavs"

echo "===== [4/4] ASR channel A + build eyeball table ====="
uv run --no-project --python 3.12 \
    --with faster-whisper --with soundfile --with numpy \
    --with transformers --with sentencepiece \
    python phase2_spokenqa/score_pilot.py \
        --questions-dir "$QDIR" \
        --gen-dir "$OUT" \
        --output-json phase2_spokenqa/pilot_v1_answers.json \
        --output-html phase2_spokenqa/pilot_v1_answers.html

echo "DONE: phase2_spokenqa/pilot_v1_answers.{json,html}"
