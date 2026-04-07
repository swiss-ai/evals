# Evaluate all the checkpoints of the local pretrained models on all the pretraining tasks
# Pretrained models: configs/snr_experiments/models_pretraining_custom.txt
# Checkpoints: each file in the directory
# Pretraining tasks: harness tasks in configs/snr_experiments/tasks_pretraining.txt
# Launch one job per checkpoint-task pair, pass --time=0:30:00
# Wandb config: configs/snr_experiments/config.json

ROOT_DIR=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small

export TOKENIZER=alehc/swissai-tokenizer
export BOS=true
export WANDB_ENTITY=mariagrandury-epflnlp
export WANDB_PROJECT=snr-experiments
export SIZE=1
TOK_PER_IT=$(( 512 * 4096 ))

MODELS_FILE=configs/snr_experiments/models_pretraining_custom.txt
TASKS_FILE=configs/snr_experiments/tasks_pretraining.txt

# Test mode: pass --test to evaluate only the first checkpoint on the first task with limit 2
if [[ "$1" == "--test" ]]; then
    MODEL_NAME=$(head -n1 "$MODELS_FILE")
    CKPT_DIR="${ROOT_DIR}/${MODEL_NAME}/checkpoints"
    FIRST_CKPT=$(ls -d "${CKPT_DIR}"/iter_*/ 2>/dev/null | head -n1)
    IT=$(basename "$FIRST_CKPT" | sed 's/iter_0*//')
    FIRST_TASK=$(head -n1 "$TASKS_FILE")
    export LIMIT=2
    export IT=$IT
    export TOKENS_PER_ITER=$TOK_PER_IT
    export NAME=$MODEL_NAME
    sbatch --time=0:30:00 --job-name "eval-${MODEL_NAME}-test" \
        scripts/evaluate.sbatch "$FIRST_TASK" "$CKPT_DIR"
    exit 0
fi

while IFS= read -r MODEL_NAME || [[ -n "$MODEL_NAME" ]]; do
    [[ -z "$MODEL_NAME" ]] && continue
    CKPT_DIR="${ROOT_DIR}/${MODEL_NAME}/checkpoints"
    for CKPT_PATH in "${CKPT_DIR}"/iter_*/; do
        [[ ! -d "$CKPT_PATH" ]] && continue
        IT=$(basename "$CKPT_PATH" | sed 's/iter_0*//')
        while IFS= read -r TASK || [[ -n "$TASK" ]]; do
            [[ -z "$TASK" ]] && continue
            export IT=$IT
            export TOKENS_PER_ITER=$TOK_PER_IT
            export NAME=$MODEL_NAME
            sbatch --time=0:30:00 --job-name "eval-${MODEL_NAME}-${IT}-${TASK}" \
                scripts/evaluate.sbatch "$TASK" "$CKPT_DIR"
        done < "$TASKS_FILE"
    done
done < "$MODELS_FILE"
