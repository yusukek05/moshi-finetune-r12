#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N 0162_v1plusv0csynth_utmos
#PBS -j oe

# Phase 2.2 — UTMOS audio quality eval on v1 vs v1+v0csynth Zoom1
# continuation wavs (50 each). Per-channel L/R UTMOS (Moshi stereo: L=A R=B)
# + overall mean, after peak normalization to 0.5 for fair comparison.
#
# Prereq: pbs/run_v1_plus_v0csynth_eval.sh has produced
#         output/v1_plus_v0csynth_eval/{v1,v1_plus_v0csynth}/generated_wavs/*.wav

set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

ICASSP=/home/acg17145sv/projects/icassp-2027-mstts
mkdir -p "$ICASSP/results"

UV_DEPS=(--with "utmosv2 @ git+https://github.com/sarulab-speech/UTMOSv2.git"
         --with soundfile --with numpy --with scipy --with tqdm)

uv run --no-project --python 3.12 "${UV_DEPS[@]}" python - <<'PYEOF'
"""UTMOS on flat-dir wav sets (peak-normalized for fair comparison).
Reads stereo wavs, scores L and R channels separately + reports mean."""
from __future__ import annotations
import json
import shutil
import tempfile
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path("output/v1_plus_v0csynth_eval")
OUT_JSON = Path("/home/acg17145sv/projects/icassp-2027-mstts/results/utmos_v1_plus_v0csynth_eval.json")
TAGS = ["v1", "v1_plus_v0csynth"]
TARGET_PEAK = 0.5  # peak-normalize for fair UTMOS comparison

def to_16k_mono(channel: np.ndarray, sr: int) -> np.ndarray:
    if sr != 16000:
        g = gcd(sr, 16000)
        channel = resample_poly(channel, 16000 // g, sr // g)
    return channel.astype("float32")

import utmosv2
print("loading utmosv2...")
model = utmosv2.create_model(pretrained=True)

results = {}
with tempfile.TemporaryDirectory() as tmp_root:
    tmp_root = Path(tmp_root)
    for tag in TAGS:
        wav_dir = ROOT / tag / "generated_wavs"
        wavs = sorted(wav_dir.glob("*.wav"))
        print(f"\n{tag}: {len(wavs)} wavs")
        if not wavs:
            continue
        scores = {"L": [], "R": [], "all": []}
        for ch_idx, ch_lab in [(0, "L"), (1, "R")]:
            stage = tmp_root / f"{tag}_{ch_lab}"
            stage.mkdir()
            staged = 0
            for w in wavs:
                arr, sr = sf.read(w, always_2d=True)
                if arr.shape[1] <= ch_idx:
                    continue
                ch = arr[:, ch_idx]
                peak = float(np.max(np.abs(ch)))
                if peak < 1e-5:
                    continue
                ch = ch * (TARGET_PEAK / peak)
                ch16k = to_16k_mono(ch, sr)
                sf.write(stage / f"{w.stem}.wav", ch16k, 16000)
                staged += 1
            print(f"  {ch_lab}: staged {staged}")
            if staged == 0:
                continue
            preds = model.predict(input_dir=str(stage))
            for r in preds:
                mos = float(r["predicted_mos"])
                scores[ch_lab].append(mos)
                scores["all"].append(mos)
        results[tag] = {
            "n_L": len(scores["L"]),
            "L_mean": float(np.mean(scores["L"])) if scores["L"] else None,
            "L_median": float(np.median(scores["L"])) if scores["L"] else None,
            "n_R": len(scores["R"]),
            "R_mean": float(np.mean(scores["R"])) if scores["R"] else None,
            "R_median": float(np.median(scores["R"])) if scores["R"] else None,
            "overall_mean": float(np.mean(scores["all"])) if scores["all"] else None,
            "overall_median": float(np.median(scores["all"])) if scores["all"] else None,
        }
        print(f"  -> overall mean MOS: {results[tag]['overall_mean']:.3f}")

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\nwrote: {OUT_JSON}")
print(json.dumps(results, ensure_ascii=False, indent=2))
PYEOF

echo "DONE: $ICASSP/results/utmos_v1_plus_v0csynth_eval.json"
