#!/bin/bash

# launch_all_evaluations.sh - Launch all evaluation scripts
# Usage: bash examples/alignment/launch_all_evaluations.sh [english|multilingual|test]

# Set default mode to english
EVAL_MODE=${1:-english}

# Validate mode
VALID_MODES=("english" "multilingual" "test")
if [[ ! " ${VALID_MODES[*]} " =~ " ${EVAL_MODE} " ]]; then
    echo "❌ Error: Invalid mode '$EVAL_MODE'"
    echo "Usage: bash examples/alignment/launch_all_evaluations.sh [english|multilingual|test]"
    echo "  english      - English tasks (default)"
    echo "  multilingual - Multilingual tasks" 
    echo "  test         - Test tasks"
    exit 1
fi

echo "🚀 Launching all evaluation scripts in $EVAL_MODE mode..."
echo "======================================"

# Set default environment variables
export SWISSAI_API_KEY="sk-rc-R-vJqSca2wRZBX5qBAGaqg"
export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-dataset-abl-v0.0.1}
export TASKS=${TASKS:-./configs/alignment/tasks_english_custom.txt}
export TABLE_METRICS=${TABLE_METRICS:-./configs/alignment/tasks_english_main_table_custom.txt}
export HF_HOME=$SCRATCH/huggingface
export HF_TOKEN=$(cat $HOME/.hf-token)
export WANDB_API_KEY=$(cat $HOME/.wandb-api-key)

# Configure based on mode
case "$EVAL_MODE" in
    "english")
        echo "🇬🇧 English mode enabled"
        export TASKS=./configs/alignment/tasks_english.txt
        export TABLE_METRICS=./configs/alignment/tasks_english_main_table.txt
        ;;
    "multilingual")
        echo "🌍 Multilingual mode enabled"
        export TASKS=./configs/alignment/tasks_multilingual.txt
        export TABLE_METRICS=./configs/alignment/tasks_multilingual_main_table.txt
        export WANDB_PROJECT="${WANDB_PROJECT}-multilingual"
        ;;
    "test")
        echo "🧪 Test mode enabled"
        export TASKS=./configs/alignment/tasks_test.txt
        export TABLE_METRICS=./configs/alignment/tasks_test_main_table.txt
        export WANDB_PROJECT="${WANDB_PROJECT}-test"
        ;;
esac

# Array of evaluation scripts to run
EVALUATION_SCRIPTS=(
    # "examples/alignment/hf_eval_multiple_apertus_base_models.sh"
    # "examples/alignment/hf_eval_multiple_apertus_models.sh"
    # "examples/alignment/hf_eval_multiple_other_base_models.sh"
    # "examples/alignment/hf_eval_multiple_other_models.sh"
    # "examples/alignment/hf_eval_sp_token_apertus_models.sh"
    "examples/alignment/hf_eval_dataset_abl_apertus_models.sh"
)

echo "📋 Scripts to be launched:"
for script in "${EVALUATION_SCRIPTS[@]}"; do
    echo "  - $script"
done

echo ""
echo "🔧 Environment variables that will be passed:"
echo "  EVAL_MODE=${EVAL_MODE}"
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
