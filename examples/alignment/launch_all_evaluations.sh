#!/bin/bash

# launch_all_evaluations.sh - Launch all evaluation scripts
# Usage: bash examples/alignment/launch_all_evaluations.sh [--multilingual]

# Check for multilingual flag
MULTILINGUAL=${MULTILINGUAL:-false}
if [[ "$1" == "--multilingual" ]]; then
    MULTILINGUAL=true
fi

echo "🚀 Launching all evaluation scripts..."
echo "======================================"

# Set default environment variables
export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-sp-tokens-abl-v0.0.1}
export TASKS=${TASKS:-./configs/alignment/tasks_english.txt}
export TABLE_METRICS=${TABLE_METRICS:-./configs/alignment/tasks_english_main_table.txt}
export HF_HOME=$SCRATCH/huggingface
export HF_TOKEN=$(cat $HOME/.hf-token)
export WANDB_API_KEY=$(cat $HOME/.wandb-api-key)

# Check if multilingual flag is set
if [ "$MULTILINGUAL" = "true" ]; then
    echo "🌍 Multilingual mode enabled"
    # Set multilingual-specific configurations
    export TASKS=./configs/alignment/tasks_multilingual_custom.txt
    export TABLE_METRICS=./configs/alignment/tasks_multilingual_main_table_custom.txt
    export WANDB_PROJECT="${WANDB_PROJECT}-multilingual"
fi

# Array of evaluation scripts to run
EVALUATION_SCRIPTS=(
    # "examples/alignment/hf_eval_multiple_apertus_base_models.sh"
    # "examples/alignment/hf_eval_multiple_apertus_models.sh"
    # "examples/alignment/hf_eval_multiple_other_base_models.sh"
    # "examples/alignment/hf_eval_multiple_other_models.sh"
    "examples/alignment/hf_eval_sp_token_apertus_models.sh"
)

echo "📋 Scripts to be launched:"
for script in "${EVALUATION_SCRIPTS[@]}"; do
    echo "  - $script"
done

echo ""
echo "🔧 Environment variables that will be passed:"
echo "  MULTILINGUAL=${MULTILINGUAL}"
echo "  TASKS=${TASKS}"
echo "  TABLE_METRICS=${TABLE_METRICS}"
echo "  WANDB_ENTITY=${WANDB_ENTITY}"
echo "  WANDB_PROJECT=${WANDB_PROJECT}"
echo "  APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-<will use script defaults>}"
echo ""
echo "🚀 Starting launches..."
echo "====================="

# Launch each evaluation script
for script in "${EVALUATION_SCRIPTS[@]}"; do    
    echo ""
    echo "🔄 Launching: $script"
    echo "----------------------------------------"
    
    # Source the script to preserve associative arrays
    bash "$script"
done
