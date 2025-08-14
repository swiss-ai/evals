#!/bin/bash

# Define MODEL:CKPT_PATH pairs using an associative array
declare -A MODEL_CHECKPOINTS=(
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-tulu3-sft-mixture-licenseFiltered-ln"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-tulu3-sft-mixture-licenseFiltered-ln/checkpoints/a33f200010b779b8/checkpoint-1590"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-tulu3-sft-mixture-ln"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-tulu3-sft-mixture-ln/checkpoints/3d1fadf422dd2b23/checkpoint-1781"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-tulu3-sft-mixture-original-ln"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-tulu3-sft-mixture-original-ln/checkpoints/219fbc9f7c06528a/checkpoint-1826"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-tulu3-sft-olmo-2-mixture-0225-ln"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-tulu3-sft-olmo-2-mixture-0225-ln/checkpoints/bb8cbbf8c687fc5f/checkpoint-1450"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-tulu3-sft-olmo-2-mixture-0225-ln-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-tulu3-sft-olmo-2-mixture-0225-ln-ademamix/checkpoints/0d8c6cb6cb34fcf7/checkpoint-1450"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-5-ln"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-5-ln/checkpoints/697716276b698b1b/checkpoint-3452"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-5-ln-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-5-ln-ademamix/checkpoints/25be975e6fb49231/checkpoint-3452"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-6-ln"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-6-ln/checkpoints/47473bc19f3d81f4/checkpoint-3934"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-6-ln-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-6-ln-ademamix/checkpoints/f3c01f37d68b4655/checkpoint-3934"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-olmo2-with-tools-ln"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-olmo2-with-tools-ln/checkpoints/bbaa40f3db869ff2/checkpoint-1681"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-olmo2-with-tools-ln-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-olmo2-with-tools-ln-ademamix/checkpoints/0476298a15ab946d/checkpoint-1681"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-1"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-1/checkpoints/7f2faa33edb7f13e/checkpoint-1622"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-2"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-2/checkpoints/44532e711f8d5bee/checkpoint-3914"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-3"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-3/checkpoints/7a61ceff935a6765/checkpoint-4577"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-4"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-4/checkpoints/8b8a3a8c41a2697e/checkpoint-4792"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-1-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-1-ademamix/checkpoints/ee969b526b1995f7/checkpoint-1622"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-2-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-2-ademamix/checkpoints/53e2ead7db07314d/checkpoint-3914"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-3-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-3-ademamix/checkpoints/8df0b5c9349b8015/checkpoint-4577"
    # ["Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-4-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-4-ademamix/checkpoints/1ce7109f7dde7ed2/checkpoint-4792"
    ["Apertus70B-tokens15T-longcontext64k-apertus-sft-mixture-5-ln-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus70B-tokens15T-longcontext64k-apertus-sft-mixture-5-ln-ademamix/checkpoints/6772117863c6be50/checkpoint-3452"
    ["Apertus70B-tokens15T-longcontext64k-apertus-sft-mixture-6-ln-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus70B-tokens15T-longcontext64k-apertus-sft-mixture-6-ln-ademamix/checkpoints/221d3f43ba5bd31d/checkpoint-3934"
    ["Apertus70B-tokens15T-longcontext64k-apertus-sft-mixture-7-ln-bs1024-ademamix"]="/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus70B-tokens15T-longcontext64k-apertus-sft-mixture-7-ln-bs1024-ademamix/checkpoints/b9111c4b57952e7b/checkpoint-4462"
)

export WANDB_ENTITY=${WANDB_ENTITY:-apertus}
export WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-dataset-abl-v0.0.1}

# SFT model configurations
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}

# Call the common runner script
source examples/alignment/hf_base_runner.sh "Apertus SFT models"
