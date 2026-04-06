#!/bin/bash
# bash scripts/submit_evals.sh
# bash scripts/submit_evals.sh configs/tasks_test.txt configs/models_test.txt

set -euo pipefail

TASKS_FILE="${1:-configs/tasks_test.txt}"
MODELS_FILE="${2:-configs/models_test.txt}"
FILE_PREFIX="${3:-/capstor/store/cscs/swissai/a139/checkpoints/moe_runs}"

while IFS= read -r model; do
    if [[ -n "$FILE_PREFIX" ]]; then
        model="${FILE_PREFIX}/${model}"
    fi
    name="${model##*/}"
    while IFS= read -r task; do
        sbatch --job-name="eval_${name}_${task}" \
               --output="logs/eval_${name}_${task}_%j.out" \
               scripts/evaluate.sbatch "${task}" "${model}"
    done < "$TASKS_FILE"
done < "$MODELS_FILE"
