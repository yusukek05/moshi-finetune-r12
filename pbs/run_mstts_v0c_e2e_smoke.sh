#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N 0162_mstts_v0c_e2e_smoke
#PBS -j oe

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

# uv is already on PATH for the user; ensure it's available here too.
export PATH="$HOME/.local/bin:$PATH"
export NO_TORCH_COMPILE=1

# Pull HF token from project .env
set -a
source "$PBS_O_WORKDIR/.env"
set +a

# Fresh staging dir to fully simulate a downstream user's experience.
SMOKE_DIR=/tmp/v0c_e2e_smoke_$PBS_JOBID
mkdir -p "$SMOKE_DIR"
cd "$SMOKE_DIR"

# 1. Download the model repository (weights + inference scripts) via uv-managed huggingface-cli.
uvx --from huggingface_hub huggingface-cli download \
    abePclWaseda/llm-jp-moshi-mstts-v0c-zoom1 \
    --local-dir mstts-v0c

cd mstts-v0c

# 2. Resolve and install all declared dependencies.
uv sync --python 3.12

# 3. Run inference on the bundled sample dialogue.
uv run python inference.py \
    --text-chat sample_dialogue.json \
    --output-wav out.wav

# 4. Verify output and report.
ls -la out.wav
uv run python -c "
import soundfile as sf, numpy as np
a, sr = sf.read('out.wav')
print(f'OK: shape={a.shape} sr={sr} dur={a.shape[0]/sr:.2f}s rms={np.sqrt((a**2).mean()):.4f}')
"

# Save the wav back to a discoverable location.
DEST="$PBS_O_WORKDIR/output/v0c_e2e_smoke_$PBS_JOBID"
mkdir -p "$DEST"
cp out.wav "$DEST/out.wav"
echo "wav saved to $DEST/out.wav"
