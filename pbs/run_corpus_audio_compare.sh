#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:30:00
#PBS -N 0162_corpus_audio_compare
#PBS -j oe

# Decode sample rows from each Stage-1 mono corpus (LaboroTV / J-CHAT-mono /
# ccaudio) back to wav so the training-data quality can be compared by ear.
# All three use the same Mimi q16 codec -> fair apples-to-apples A/B.

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

OUT=output/corpus_audio_compare
DP=/groups/gcg51557/experiments/0178_dialogue_tts/data/parquet

uv run -m tools.decode_mono_parquet_samples \
    --parquet "$DP/LaboroTVSpeech/rinna_gpt2-kyutai_mimi-q16/shard-00000.parquet" \
    --output_dir "$OUT/laborotv" --prefix laborotv --num_samples 5

uv run -m tools.decode_mono_parquet_samples \
    --parquet "$DP/J-CHAT-mono/rinna_gpt2-kyutai_mimi-q16/shard-00000.parquet" \
    --output_dir "$OUT/jchat" --prefix jchat --num_samples 5

uv run -m tools.decode_mono_parquet_samples \
    --parquet "processed_data/ccaudio/rinna_gpt2-kyutai_mimi-q16/shard-00000.parquet" \
    --output_dir "$OUT/ccaudio" --prefix ccaudio --num_samples 5

echo "DONE: $OUT/{laborotv,jchat,ccaudio}/"
