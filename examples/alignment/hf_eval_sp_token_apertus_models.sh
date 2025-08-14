#!/bin/bash

# Define MODEL:CKPT_PATH pairs using an associative array
declare -A MODEL_CHECKPOINTS=(
    # ["Apertus-8B-7.2T-tulu3-sft-mix-chatml_no_special_tokens"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-chatml_no_special_tokens/checkpoints/84c2f794bc7c16f2/checkpoint-1461"
    # ["Apertus-8B-7.2T-tulu3-sft-mix-chatml_special_tokens"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-chatml_special_tokens/checkpoints/27de049d7518baf1/checkpoint-1461"
    # ["Apertus-8B-7.2T-tulu3-sft-mix-mistral_no_special_tokens"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-mistral_no_special_tokens/checkpoints/a4aa00d6489e165f/checkpoint-1461"
    # ["Apertus-8B-7.2T-tulu3-sft-mix-mistral_special_tokens"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-mistral_special_tokens/checkpoints/63ea896c516b6ec4/checkpoint-1461"
    # ["Apertus-8B-7.2T-tulu3-sft-mix-xml_no_special_tokens"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-xml_no_special_tokens/checkpoints/1d5c7616c0964c34/checkpoint-1461"
    # ["Apertus-8B-7.2T-tulu3-sft-mix-mistral_special_tokens_eos"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-mistral_special_tokens_eos/checkpoints/2d03375fdc119605/checkpoint-365"
    # ["Apertus-8B-7.2T-tulu3-sft-mix-mistral_asymmetric_assitant"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-mistral_asymmetric_assitant/checkpoints/48ccebcf3fd75f0f/checkpoint-365"
    # ["Apertus-8B-7.2T-tulu3-sft-mix-mistral_beginning_of_message"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-mistral_beginning_of_message/checkpoints/d2791eb90bf59ef0/checkpoint-365"
    # ["Apertus-8B-7.2T-tulu3-sft-mix-mistral_everywhere"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-mistral_everywhere/checkpoints/6adc1ec091e1cce0/checkpoint-365"
    # ["Apertus-8B-7.2T-tulu3-sft-mix-mistral_everywhere_except_eos"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-mistral_everywhere_except_eos/checkpoints/37487cc16bfa93ab/checkpoint-365"
    # ["Apertus-8B-7.2T-tulu3-sft-mix-mistral_only_between_roles"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-mistral_only_between_roles/checkpoints/d0a22e7773b258d9/checkpoint-365"
    # ["Apertus-8B-7.2T-tulu3-sft-mix-mistral_special_tokens_s62_only"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/sp-token-ablation/Apertus8B-tokens7.2T-it1728000-tulu3-sft-mix-subset20-mistral_special_tokens_s62_only/checkpoints/349db5ba991d73f7/checkpoint-365"
)

export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-sp-tokens-abl-v0.0.4}

# SFT model configurations
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}

# Call the common runner script
source examples/alignment/hf_base_runner.sh "Apertus SFT models"
