#!/bin/bash -l
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -v RTYPE=rt_HF
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N 0162_v1plussynth_zoom1_post
#PBS -j oe

# zero_to_fp32 for the two Zoom1 re-anchored runs:
#   output/v1_plus_v0csynth_zoom1/step_<final>      -> step_<final>_fp32
#   output/v1_plus_cascadesynth_zoom1/step_<final>  -> step_<final>_fp32
#
# Final step is auto-detected (7 Zoom1 epochs ~= 402, numbering restarts at 0).
# Skips a run if its fp32 dir already exists, so this is safe to re-run.
# clean_moshi (dep_q=8) is NOT done here — eval uses the dep_q=16 fp32 directly.
#
# Prereq: run_train_v1_plus_{v0csynth,cascadesynth}_zoom1.sh completed.
# Next:   qsub -W depend=afterok:<this> pbs/run_v1_plus_v0csynth_eval.sh

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

for spec in \
    "output/v1_plus_v0csynth_zoom1:output/v1_plus_v0csynth/step_2757_fp32/moshi_lm_kwargs.json" \
    "output/v1_plus_cascadesynth_zoom1:output/v1_plus_cascadesynth/step_2714_fp32/moshi_lm_kwargs.json"
do
    RUN_DIR="${spec%%:*}"
    KWARGS="${spec##*:}"

    # latest step_N dir that is a raw ZeRO-3 shard dir (not *_fp32 etc.)
    STEP=$(ls -d "$RUN_DIR"/step_* 2>/dev/null | grep -E 'step_[0-9]+$' | sort -t_ -k2 -n | tail -1 | xargs -r basename)
    if [ -z "$STEP" ]; then
        echo "SKIP $RUN_DIR: no step_* checkpoint found"
        continue
    fi
    if [ -d "$RUN_DIR/${STEP}_fp32" ]; then
        echo "SKIP $RUN_DIR/${STEP}_fp32: already consolidated"
        continue
    fi

    echo "===== zero_to_fp32: $RUN_DIR/$STEP ====="
    uv run -m tools.zero_to_fp32 \
        "$RUN_DIR/$STEP" \
        "$RUN_DIR/${STEP}_fp32" \
        --moshi_lm_kwargs_path "$KWARGS"
    ls -la "$RUN_DIR/${STEP}_fp32/"
done

echo "DONE"
