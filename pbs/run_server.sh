module load python/3.12/3.12.9
module load cuda/12.6/12.6.1

uv sync --python 3.12

uv run -m moshi.server \
    --moshi-weight output/moshiko-finetuned/step_1_cleaned/model.safetensors