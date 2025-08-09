#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --job-name=evaluation
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --time=11:59:00

# Aux functions.
usage() {
	echo "Usage: sbatch evaluate_hf.sbatch <model> <name>"
	echo "Runs evaluation of the specified model. The 'model' should a huggingface path/name."
	echo "The 'name' shall be used as an unique identifier."
	echo "In addition of these three necessary positional arguments, you can specify the following bash environment variables. Unless explicitly stated, all of these are optional:"
	echo " TOKENIZER: A huggingface tokenizer path/name. Needed if 'model' is a megatron checkpoint, otherwise the default will be set as the same value as 'model'."
	echo " BS: Batch size."
	echo " REVISION: Only used in huggingface models. If set, this revision of the model will be evaluated."
	echo " SIZE: The (approximate) size of the model in billions of parameters. Used to set model parallelism (needs to be set in larger models)."
	echo " LIMIT: The --limit argument to pass to lm-evaluation-harness. By default, it is unset."
	echo " MAX_LENGTH: The maximum length of the input sequence to the model. By default, it is unset."
	echo " MAX_NEW_TOKENS: The maximum number of new tokens to generate. By default, it is unset."
	echo " BOS: Set this to 'true' if you wish to prepend the BOS token when evaluating models."
	echo " LOGS_ROOT: Where are your evaluation wandb&harness logs going to."
	echo " TASKS: Tasks to run with lm eval harness."
	echo " WANDB_ENTITY: WandB entity name for uploading results (default: apertus)."
	echo " WANDB_PROJECT: WandB project name for uploading results (default: swissai-evals)."
	echo "For more information see the README: https://github.com/swiss-ai/evals?tab=readme-ov-file."
}
die() {
	echo "$*" >& 2
	exit 1
}

# Wakeup logs.
set -e
echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

# Grab variables and arguments.
if (( $# != 2 )); then
	usage
	die "Invalid usage: Invalid argument count"
fi
MODEL=$1
NAME=$2
BS=${BS:-"auto:20"}
SIZE=${SIZE:-1}
BOS=${BOS:-false}
LOGS_ROOT=${LOGS_ROOT:-/capstor/store/cscs/swissai/infra01/eval-logs}
TASKS=${TASKS:-./configs/alignment/tasks_english.txt}
TABLE_METRICS=${TABLE_METRICS:-./configs/alignment/tasks_english_main_table.txt}
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}

# optional eval arguments
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
LIMIT=${LIMIT:-""}

if [ -f "$TASKS" ]; then
    echo "Reading task list from file: $TASKS"
    TASKS=$(paste -sd, "$TASKS")
fi

if [ -f "$TABLE_METRICS" ]; then
	echo "Reading table metrics from file: $TABLE_METRICS"
	TABLE_METRICS=$(paste -sd' ' "$TABLE_METRICS")
fi

export HF_HOME=${HF_HOME:-/capstor/store/cscs/swissai/infra01/hf_home/}
GPUS_PER_NODE=4

# Print configuration.
echo "Configuration set:"
printf "MODEL=$MODEL\nNAME=$NAME\nTOKENIZER=$TOKENIZER\nBS=$BS\nREVISION=$REVISION\nSIZE=$SIZE\nLIMIT=$LIMIT\nBOS=$BOS\nTASKS=$TASKS\nWANDB_ENTITY=$WANDB_ENTITY\nWANDB_PROJECT=$WANDB_PROJECT\nHF_HOME=$HF_HOME\n\n"

# Generic envs.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25678
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WORLD_SIZE=1

# export HF_ALLOW_CODE_EVAL=1

echo "Huggingface checkpoint detected!"
if [ ! -z ${REVISION+x} ]; then
    echo "Using revision=$REVISION"
    MAYBE_REVISION=",revision=$REVISION"
    MAYBE_DOWNLOAD_REVISION="--revision=$REVISION"
fi
if [ -z ${TOKENIZER+x} ]; then
    echo "TOKENIZER not set, using TOKENIZER=$MODEL"
    TOKENIZER=$MODEL
fi


# Use only model parallel because of vLLM
MP=$GPUS_PER_NODE
echo "Using model size ~${SIZE}B, model parallel size of $MP"

# Prepare logs root.
RUN_ROOT=$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/$NAME
HARNESS_DIR=$RUN_ROOT/harness
export WANDB_DIR=$RUN_ROOT
mkdir -p $HARNESS_DIR

# export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
# export PERSPECTIVE_API_KEY="" # Uncomment if you want to use perspective API, needed for realtoxicityprompts task

# Finally, run the actual evaluations.
HF_CHECKPOINT_PATH=$MODEL
DP=$(( GPUS_PER_NODE / MP ))
COMMON_MODEL_ARGS="pretrained=$HF_CHECKPOINT_PATH,tokenizer=$TOKENIZER$MAYBE_REVISION,dtype=bfloat16"
if [[ $BOS = true ]]; then
	echo "Adding BOS token."
	COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,add_bos_token=True"
fi

# Add vLLM specific arguments.
export OMP_NUM_THREADS=$(( 8 * $GPUS_PER_NODE ))
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_COMPILE_CACHE=1
VLLM_MEMORY_FRACTION=0.75
COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,data_parallel_size=$DP,tensor_parallel_size=$MP,gpu_memory_utilization=$VLLM_MEMORY_FRACTION,max_length=$MAX_LENGTH,enable_thinking=False,enforce_eager=True"

HARNESS_EVAL_DIR=$HARNESS_DIR/eval_$(date +%Y%m%d_%H%M%S)_$SLURM_JOBID

COMMON_EVAL_ARGS=(
	--trust_remote_code
	--batch_size $BS
	--tasks $TASKS
	--output $HARNESS_EVAL_DIR
	--max_batch_size 32
	--model vllm
	--log_samples
	--write_out
	--confirm_run_unsafe_code
)

if [[ $APPLY_CHAT_TEMPLATE = true ]]; then
	COMMON_EVAL_ARGS+=(--apply_chat_template)
	COMMON_EVAL_ARGS+=(--fewshot_as_multiturn)
fi

if [[ -n ${LIMIT} ]]; then
	COMMON_EVAL_ARGS+=(--limit $LIMIT)
fi

if [[ -n $MAX_GEN_TOKENS ]]; then
	COMMON_EVAL_ARGS+=(--gen_kwargs max_gen_toks=$MAX_GEN_TOKENS)
fi

INSTALL_CMD="pip install -v --no-cache-dir --no-build-isolation --no-deps --force-reinstall -U \
	datasketch \
	sympy \
	math_verify \
	\"antlr4-python3-runtime==4.11\" \
	latex2sympy2_extended \
    \"lm-eval[vllm] @ git+https://github.com/swiss-ai/lm-evaluation-harness.git@main\""

CMD="lm_eval --model_args=$COMMON_MODEL_ARGS ${COMMON_EVAL_ARGS[@]}"

echo "Installation command: $INSTALL_CMD"
echo "Final command: $CMD"
srun -ul --environment=./containers/env.toml bash -c " \
	$INSTALL_CMD
    $CMD
"

# Goodbye.
echo "Evaluation finished"
echo "Sleeping for 1 minute before uploading results to wandb..."
sleep 60
echo "Uploading results to wandb"
WANDB_CMD="cd $PWD && python -m scripts.alignment.update_wandb_alignment --entity $WANDB_ENTITY --project $WANDB_PROJECT --logs_root $HARNESS_EVAL_DIR --name $NAME --main_metrics $TABLE_METRICS"
echo "Running command to upload results to wandb:"
echo "$WANDB_CMD"
srun -ul --environment=./containers/env.toml bash -c " \
    $WANDB_CMD
"
echo "END TIME: $(date)"
