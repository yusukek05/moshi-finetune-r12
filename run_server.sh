#!/bin/bash
uv run --no-sync -m moshi.server \
    --moshi-weight output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_7epochs_1node_exp_textpad1/step_9282_cleaned/model.safetensors \
    --host 0.0.0.0 \
    --port 8998 \
    --tokenizer data/tokenizer/spiece.model \
    --initial-audio /home/acg17145sv/experiments/0162_dialogue_model/NISQA/data_sample_audio/llmjp-zoom1/000_0671_W02_W28_T11_1704.1_1724.1.wav
    # --initial-audio data/initial_audio/Yuto.wav
