
export TOKENIZER="swiss-ai/Apertus-70B-2509"
export BOS=true
export WANDB_PROJECT="data-ablations-evals"
export MEGATRON_BRANCH="81d44dbe25c22190989fa8a12284bdb8d1303b40"
export TRANSFORMERS_BRANCH=v4.50.0+swissai

CKPT_PATH="/iopsstor/scratch/cscs/bmessmer/data_ablations/Megatron-LM/logs/Meg-Runs/data-ablations/apertus3-1b-21-nodes-phase-4-baseline/checkpoints/"

ITS=(0050000)
TOK_PER_IT=$(( 504 * 4096 ))
MODEL=apertus3-1b-21-nodes-phase-4-baseline

for IT in "${ITS[@]}"; do
	sbatch --job-name eval-$MODEL-$IT scripts/evaluate.sbatch $CKPT_PATH $IT $TOK_PER_IT $MODEL
done