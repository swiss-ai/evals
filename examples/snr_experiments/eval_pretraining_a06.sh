# STATUS: ON HOLD, can't access a06
# Evaluate all the checkpoints of the local pretrained models on all the pretraining tasks
# Pretrained models: configs/snr_experiments/models_pretraining.txt
# Checkpoints: each file in the directory
# Pretraining tasks: harness tasks in configs/snr_experiments/tasks_pretraining.txt
# Launch one job per checkpoint-task pair, pass --time=0:30:00
# Wandb config: configs/snr_experiments/config.json

export TOKENIZER=alehc/swissai-tokenizer
export BOS=true
export WANDB_ENTITY=mariagrandury-epflnlp
export WANDB_PROJECT=snr-experiments
TOK_PER_IT=$((4096 * 2048))

MODELS_FILE=configs/snr_experiments/models_pretraining.txt
TASKS_FILE=configs/snr_experiments/tasks_pretraining.txt

# Test mode: pass --test to evaluate only the first checkpoint on the first task with limit 2
if [[ "$1" == "--test" ]]; then
    CKPT_DIR=$(head -n1 "$MODELS_FILE")
    FIRST_CKPT=$(ls -d "${CKPT_DIR}"iter_*/ 2>/dev/null | head -n1)
    IT=$(basename "$FIRST_CKPT" | sed 's/iter_0*//')
    MODEL_NAME=$(basename "$(dirname "$CKPT_DIR")")
    export TASKS=$(head -n1 "$TASKS_FILE")
    export LIMIT=2
    sbatch --time=0:30:00 --job-name "eval-${MODEL_NAME}-test" \
        scripts/evaluate.sbatch "$CKPT_DIR" "$IT" "$TOK_PER_IT" "$MODEL_NAME"
    exit 0
fi

while IFS= read -r CKPT_DIR || [[ -n "$CKPT_DIR" ]]; do
    [[ -z "$CKPT_DIR" ]] && continue
    MODEL_NAME=$(basename "$(dirname "$CKPT_DIR")")
    for CKPT_PATH in "${CKPT_DIR}"iter_*/; do
        [[ ! -d "$CKPT_PATH" ]] && continue
        IT=$(basename "$CKPT_PATH" | sed 's/iter_0*//')
        while IFS= read -r TASK || [[ -n "$TASK" ]]; do
            [[ -z "$TASK" ]] && continue
            export TASKS=$TASK
            sbatch --time=0:30:00 --job-name "eval-${MODEL_NAME}-${IT}-${TASK}" \
                scripts/evaluate.sbatch "$CKPT_DIR" "$IT" "$TOK_PER_IT" "$MODEL_NAME"
        done < "$TASKS_FILE"
    done
done < "$MODELS_FILE"
