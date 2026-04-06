#!/bin/bash
# bash scripts/submit_evals.sh
# bash scripts/submit_evals.sh configs/tasks_test.txt configs/models_test.txt

set -euo pipefail

TASKS_FILE="${1:-configs/tasks_test.txt}"
MODELS_FILE="${2:-configs/models_test.txt}"

while IFS= read -r model; do
    name="${model##*/}"
    while IFS= read -r task; do
        sbatch --job-name="eval_${name}_${task}" \
               --output="logs/eval_${name}_${task}_%j.out" \
               scripts/evaluate.sbatch "${task}" "${model}"
    done < "$TASKS_FILE"
done < "$MODELS_FILE"
