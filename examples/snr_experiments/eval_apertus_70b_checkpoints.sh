# Evaluate all the checkpoints of Apertus 70B on INCLUDE
# Apertus 70B: https://huggingface.co/swiss-ai/Apertus-70B-2509
# Apertus 70B checkpoints: each branch of the HF repo is a checkpoint
# INCLUDE: harness tasks in configs/snr_experiments/tasks_include.txt
# Launch one job per checkpoint-task pair, pass --time=0:30:00
# Wandb config: configs/snr_experiments/config.json

MODEL=swiss-ai/Apertus-70B-2509
NAME_PREFIX=Apertus-70B-2509

export WANDB_ENTITY=mariagrandury-epflnlp
export WANDB_PROJECT=snr-experiments
export SIZE=70
export APPLY_CHAT_TEMPLATE=false

TASKS_FILE=configs/snr_experiments/tasks_include.txt

CHECKPOINTS=(
    main
    step1155828-tokens15T
    longctx-step50
    longctx-step100
    longctx-step150
    longctx-step200
    longctx-step250
    longctx-step300
    longctx-step350
    longctx-step400
    longctx-step450
    longctx-step500
    longctx-step550
    longctx-step600
    longctx-step650
    longctx-step700
    longctx-step750
    longctx-step800
    longctx-step850
    longctx-step900
    longctx-step950
    longctx-step1000
    longctx-step1050
    longctx-step1100
    longctx-step1150
    longctx-step1200
    longctx-step1250
    longctx-step1300
    longctx-step1350
    longctx-step1400
    longctx-step1450
    longctx-step1500
    longctx-step1550
    longctx-step1600
    longctx-step1650
    longctx-step1700
    step25000-tokens210B
    step50000-tokens420B
    step75000-tokens630B
    step100000-tokens840B
    step125000-tokens1050B
    step150000-tokens1260B
    step175000-tokens1470B
    step200000-tokens1680B
    step225000-tokens1890B
    step250000-tokens2100B
    step275000-tokens2310B
    step300000-tokens2520B
    step325000-tokens2730B
    step350000-tokens2940B
    step375000-tokens3150B
    step400000-tokens3360B
    step425000-tokens3570B
    step450000-tokens3780B
    step475000-tokens3990B
    step500000-tokens4200B
    step525000-tokens4420B
    step550000-tokens4840B
    step575000-tokens5260B
    step600000-tokens5680B
    step625000-tokens6100B
    step650000-tokens6520B
    step675000-tokens6940B
    step700000-tokens7360B
    step725000-tokens7780B
    step750000-tokens8200B
    step775000-tokens8620B
    step800000-tokens9040B
    step825000-tokens9460B
    step850000-tokens9880B
    step875000-tokens10300B
    step900000-tokens10720B
    step925000-tokens11140B
    step950000-tokens11560B
    step975000-tokens11980B
    step1000000-tokens12400B
    step1025000-tokens12820B
    step1050000-tokens13240B
    step1075000-tokens13660B
    step1100000-tokens14080B
    step1125000-tokens14500B
    step1150000-tokens14920B
)


# Test mode: pass --test to evaluate only the first checkpoint on the first task with limit 2
if [[ "$1" == "--test" ]]; then
    export REVISION=${CHECKPOINTS[0]}
    export TASKS=$(head -n1 "$TASKS_FILE")
    export LIMIT=2
    NAME=${NAME_PREFIX}-${CHECKPOINTS[0]}-test
    sbatch --time=0:30:00 --job-name "eval-${NAME_PREFIX}-test" \
        scripts/evaluate_hf.sbatch "$MODEL" "$NAME"
    exit 0
fi

for CKPT in "${CHECKPOINTS[@]}"; do
    export REVISION=$CKPT
    NAME=${NAME_PREFIX}-${CKPT}
    while IFS= read -r TASK || [[ -n "$TASK" ]]; do
        [[ -z "$TASK" ]] && continue
        export TASKS=$TASK
        sbatch --time=0:30:00 --job-name "eval-${NAME_PREFIX}-${CKPT}-${TASK}" \
            scripts/evaluate_hf.sbatch "$MODEL" "$NAME"
    done < "$TASKS_FILE"
done
