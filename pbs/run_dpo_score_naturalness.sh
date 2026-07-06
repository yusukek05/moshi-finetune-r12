#!/bin/bash -l
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -q rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N 0162_dpo_score_nat
#PBS -j oe
#PBS -o logs/

# DPO stage 2a: score the acoustic *naturalness* reward for every on-policy
# sample, using the crowdsourcing reward model (librosa + JP-HuBERT L9 + GBDT,
# scikit-learn 1.6.1). We deliberately use only the naturalness head — the
# meaningfulness head over-rates fluent nonsense (docs/2026-07-01 §6).
#
# Writes <gen_root>/naturalness.json = score_audio rows [{path,naturalness,...}],
# consumed by tools/build_dpo_pairs.py --naturalness_json.
set -euxo pipefail
echo "JOB_ID=$PBS_JOBID"
cd "$PBS_O_WORKDIR"

module purge
module load cuda/12.6/12.6.1
module load python/3.12/3.12.9

CW=/groups/gcg51557/experiments/0386_dialogue_model/crowdsourcing
# override at submit: qsub -v RTYPE=rt_HF,GEN_ROOT=/abs/path/output/dpo_scaled
GEN_ROOT="${GEN_ROOT:-$PWD/output/dpo_pilot}"
export HF_HUB_OFFLINE=1

# score_audio.py imports extract_features relatively -> run from the repo root,
# reward_*.joblib load from there too.
cd "$CW"
"$CW/.venv/bin/python" score_audio.py \
    --glob "${GEN_ROOT}/seed*/generated_wavs/*.wav" \
    --json > "${GEN_ROOT}/naturalness_raw.json"

# score_audio prints an "[ssl] loaded ..." banner to stdout before the JSON;
# strip it so the file is valid JSON.
grep -v '^\[ssl\]' "${GEN_ROOT}/naturalness_raw.json" > "${GEN_ROOT}/naturalness.json"
echo "wrote ${GEN_ROOT}/naturalness.json"
"$CW/.venv/bin/python" -c "import json;d=json.load(open('${GEN_ROOT}/naturalness.json'));print('scored',len(d),'wavs')"
