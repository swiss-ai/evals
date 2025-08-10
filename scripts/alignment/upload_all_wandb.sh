#!/bin/bash

# Script to upload all model evaluation results to W&B
# Uses the same entity and project defaults as evaluate_hf.sbatch

# Set default values (same as evaluate_hf.sbatch)
export WANDB_API_KEY=$(cat $HOME/.wandb-api-key)
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-sp-tokens-abl-v0.0.3-multilingual}
LOGS_ROOT=${LOGS_ROOT:-/capstor/store/cscs/swissai/infra01/eval-logs/apertus/swissai-evals-sp-tokens-abl-v0.0.3-multilingual}

echo "Uploading all model results to W&B..."
echo "Entity: $WANDB_ENTITY"
echo "Project: $WANDB_PROJECT"
echo "Logs root: $LOGS_ROOT"
echo ""

# Run the Python script
python -m scripts.alignment.update_wandb_all_models \
    --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" \
    --logs_root "$LOGS_ROOT"
