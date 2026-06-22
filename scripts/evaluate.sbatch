#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --environment=/users/%u/repos/evals/containers/env.toml
#SBATCH --job-name=evaluation
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/data_ablations/%x_%j.out
#SBATCH --error=logs/data_ablations/%x_%j.err
#SBATCH --time=00:59:00
#SBATCH --exclusive

# Aux functions.
usage() {
	echo "Usage: sbatch evaluate.sbatch <model> <iteration> <tokens-per-iter> <name>"
	echo "Runs evaluation of the specified model. The 'model' should either be a megatron checkpoint or a huggingface path/name. The script determines which one is given by looking for 'model/latest_checkpointed_iteration.txt', if it is not found a huggingface model is assumed."
	echo "The 'iteration' and 'tokens-per-iter' is used to calculate the consumed tokens of the model."
	echo "The 'name' shall be used as an unique identifier."
	echo "In addition of these three necessary positional arguments, you can specify the following bash environment variables. Unless explicitly stated, all of these are optional:"
	echo " TOKENIZER: A huggingface tokenizer path/name. Needed if 'model' is a megatron checkpoint, otherwise the default will be set as the same value as 'model'."
	echo " BS: Batch size."
	echo " REVISION: Only used in huggingface models. If set, this revision of the model will be evaluated."
	echo " MEGATRON_BRANCH: Only used in megatron checkpoints. The megatron->huggingface conversion scripts will be looked at this branch."
	echo " HARNESS_BRANCH: lm-evaluation-harness branch to install."
	echo " TRANSFORMERS_BRANCH: transformers branch to install."
	echo " SIZE: The (approximate) size of the model in billions of parameters. Used to set model parallelism (needs to be set in larger models)."
	echo " LIMIT: The --limit argument to pass to lm-evaluation-harness."
	echo " HF_TEMP_DIR: If set the converted megatron checkpoints will be saved here, otherwise they will not be saved anywhere."
	echo " BOS: Set this to 'true' if you wish to prepend the BOS token when evaluating models."
	echo " LOGS_ROOT: Where are your evaluation wandb&harness logs going to."
	echo " TASKS: Tasks to run with lm eval harness."
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
if (( $# != 4 )); then
	usage
	die "Invalid usage: Invalid argument count"
fi
MODEL=$1
IT=$2
TOKENS_PER_ITER=$3
NAME=$4
BS=${BS:-"auto:20"}
MEGATRON_BRANCH=${MEGATRON_BRANCH:-main}
SIZE=${SIZE:-1}
BOS=${BOS:-false}
LOGS_ROOT=${LOGS_ROOT:-$SCRATCH/eval-logs}
TASKS=${TASKS:-swissai_eval}
#TASKS=${TASKS:-winogrande,piqa,openbookqa,commonsense_qa,mmlu_continuation,hellaswag,ai2_arc}
#TASKS=${TASKS:-mmlu_continuation}

TRANSFORMERS_BRANCH=${TRANSFORMERS_BRANCH:-main}
HARNESS_BRANCH=${HARNESS_BRANCH:-granular_eval}
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/$USER/hf_home/}
GPUS_PER_NODE=4

# Print configuration.
echo "Configuration set:"
printf "MODEL=$MODEL\nIT=$IT\nTOKENS_PER_ITER=$TOKENS_PER_ITER\nNAME=$NAME\nTOKENIZER=$TOKENIZER\nBS=$BS\nREVISION=$REVISION\nMEGATRON_BRANCH=$MEGATRON_BRANCH\nSIZE=$SIZE\nLIMIT=$LIMIT\nBOS=$BOS\nLOGS_ROOT=$LOGS_ROOT\nTASKS=$TASKS\n\nTRANSFORMERS_BRANCH=$TRANSFORMERS_BRANCH\nHARNESS_BRANCH=$HARNESS_BRANCH\nHF_HOME=$HF_HOME\n\n"

# Generic envs.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25678
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Determine CONSUMED_TOKENS.
if [ -f $MODEL/latest_checkpointed_iteration.txt ]; then
	echo "Megatron checkpoint detected!"
	IS_MEGATRON=true
	if [[ -z ${TOKENIZER+x} ]]; then
		die "You should set TOKENIZER when using megatron checkpoints."
	fi

	# Determine the CONSUMED_TOKENS based on how TOKENS_PER_ITER is specified.
	if [[ $TOKENS_PER_ITER = *,* ]]; then  # TOKENS_PER_ITER specified in a "tok0:it0,tok1:it1,tok2:" way
		CONSUMED_TOKENS=0
		CONSUMED_ITERS=0
		for SUBSTR in ${TOKENS_PER_ITER//,/ }; do
			if (( CONSUMED_ITERS < IT )); then
				TOK_PER_ITER="$(echo $SUBSTR | cut -d':' -f1)"
				MAX_ITER="$(echo $SUBSTR | cut -d':' -f2)"
				if [[ $MAX_ITER = "" ]]; then
					ITERS_THIS_BLOCK=$(( IT - CONSUMED_ITERS ))
				elif (( IT > MAX_ITER )); then
					ITERS_THIS_BLOCK=$(( MAX_ITER - CONSUMED_ITERS - 1 ))
				else
					ITERS_THIS_BLOCK=$(( IT - CONSUMED_ITERS ))
				fi
				CONSUMED_ITERS=$(( CONSUMED_ITERS + ITERS_THIS_BLOCK ))
				CONSUMED_TOKENS=$(( CONSUMED_TOKENS + TOK_PER_ITER*ITERS_THIS_BLOCK ))
			fi
		done
	else
		CONSUMED_TOKENS=$(( IT*TOKENS_PER_ITER ))
	fi
else
	echo "Huggingface checkpoint detected!"
	IS_MEGATRON=false
	CONSUMED_TOKENS=$(( IT*TOKENS_PER_ITER ))
	if [ ! -z ${REVISION+x} ]; then
		echo "Using revision=$REVISION"
		MAYBE_REVISION=",revision=$REVISION"
		MAYBE_DOWNLOAD_REVISION="--revision=$REVISION"
	fi
	if [ -z ${TOKENIZER+x} ]; then
		echo "TOKENIZER not set, using TOKENIZER=$MODEL"
		TOKENIZER=$MODEL
	fi
fi
echo "Consumed tokens for this iteration: $CONSUMED_TOKENS"

# Determine model parallel based on size.
if (( SIZE <= 35 )); then
	MP=1
	CONVERT_MP=1
elif (( SIZE <= 70 )); then
	MP=2
	CONVERT_MP=4
else
	MP=4
	CONVERT_MP=4
fi
echo "Using model size ~${SIZE}B, model parallel size of $MP, conversion parallel size of $CONVERT_MP"

# Setup temporary paths.
TEMP_PATH_ROOT=$SCRATCH/.tmp
if [ -z ${HF_TEMP_DIR+x} ]; then
	echo "HF_TEMP_DIR not set; HF conversions will not be stored"
	HF_TEMP_PATH=$(mktemp -d -p $TEMP_PATH_ROOT)       # To store hf conversion (if needed).
else
	echo "Using HF_TEMP_DIR=${HF_TEMP_DIR}"
	HF_TEMP_PATH=$HF_TEMP_DIR/$SLURM_JOB_NAME
	mkdir -p $HF_TEMP_PATH
fi
TORCH_NODIST_PATH=$(mktemp -d -p $TEMP_PATH_ROOT)  # To store torch no dist checkpoint converted (if needed).
TEMP_REPOS=$(mktemp -d -p $TEMP_PATH_ROOT)
function cleanup {
	if [ -z ${HF_TEMP_DIR+x} ]; then
		rm -rf $HF_TEMP_PATH
	fi
	rm -rf $TORCH_NODIST_PATH
	rm -rf $TEMP_REPOS
}
trap cleanup EXIT

# Prepare logs root.
RUN_ROOT=$LOGS_ROOT/$NAME/iter_$IT
HARNESS_DIR=$RUN_ROOT/harness
export WANDB_DIR=$RUN_ROOT
mkdir -p $HARNESS_DIR
echo $CONSUMED_TOKENS > $RUN_ROOT/consumed_tokens.txt
echo $SLURM_JOB_ID >> $RUN_ROOT/job_ids.txt

# Install custom transformers and lm-harness.
echo Installing transformers and eval harness.
cd $TEMP_REPOS
git clone https://github.com/swiss-ai/transformers.git
cd transformers
git checkout $TRANSFORMERS_BRANCH
pip install -e .
cd ..

git clone https://github.com/swiss-ai/lm-evaluation-harness.git
#git clone https://github.com/beme248/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout $HARNESS_BRANCH
pip install -e .


# Convert to HF if needed or pre-download the HF model.
if [ $IS_MEGATRON = true ]; then
	# Clone megatron.
	echo Clonning megatron.
	cd $TEMP_REPOS
	git clone https://github.com/swiss-ai/Megatron-LM.git
	cd Megatron-LM
	git checkout $MEGATRON_BRANCH

	echo "Converting megatron checkpoint to huggingface!"
	echo "Running torchdist->torch"
	export PYTHONPATH=$PWD
	torchrun \
		--nproc-per-node $CONVERT_MP \
		scripts/conversion/torchdist_2_torch.py \
		--bf16 --load=$MODEL \
		--ckpt-step=$IT \
		--ckpt-convert-save=$TORCH_NODIST_PATH \
		--pipeline-model-parallel-size $CONVERT_MP
	echo "Running torch->hf"
	python tools/checkpoint/convert.py \
		--model-type=GPT \
		--loader=core \
		--saver=swissai_hf \
		--load-dir=$TORCH_NODIST_PATH/torch \
		--save-dir=$HF_TEMP_PATH \
		--hf-tokenizer=$TOKENIZER
	HF_CHECKPOINT_PATH=$HF_TEMP_PATH
	huggingface-cli download $TOKENIZER
else
	HF_CHECKPOINT_PATH=$MODEL
	if [[ -d $MODEL ]]; then
		echo "$MODEL is a local directory, skipping download."
	else
		huggingface-cli download $MODEL $MAYBE_DOWNLOAD_REVISION
	fi
	if [ $TOKENIZER != $MODEL ]; then
		echo "Why are you using a different tokenizer and model? TOKENIZER=$TOKENIZER, MODEL=$MODEL"
		huggingface-cli download $TOKENIZER
	fi
fi
export HF_HUB_OFFLINE=1

# Finally, run the actual evaluations.
DP=$(( GPUS_PER_NODE / MP ))
COMMON_MODEL_ARGS="pretrained=$HF_CHECKPOINT_PATH,tokenizer=$TOKENIZER,max_length=4096$MAYBE_REVISION,attn_implementation=flash_attention_2,dtype=bfloat16"
if [[ $BOS = true ]]; then
	echo "Adding BOS token."
	COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,add_bos_token=True"
fi
COMMON_EVAL_ARGS=(
	--trust_remote_code
	--batch_size $BS
	--tasks $TASKS
	--output $HARNESS_DIR/eval_$SLURM_JOBID
	--max_batch_size 128
	--cache_requests true
	--model hf
	--log_samples
	--write_out
	$LIMIT_ARGS
)
if [[ $DP -eq 1 ]]; then  # Only use model parallel.
	export WORLD_SIZE=1
	CMD="lm_eval --model_args=$COMMON_MODEL_ARGS,parallelize=True ${COMMON_EVAL_ARGS[@]}"
elif (( MP > 1 )); then  # Use data parallel and model parallel.
	CMD="accelerate launch --multi_gpu --num_processes $DP -m lm_eval --model_args=$COMMON_MODEL_ARGS,parallelize=True ${COMMON_EVAL_ARGS[@]}"
else  # Only use data parallel.
	CMD="accelerate launch -m lm_eval --model_args=$COMMON_MODEL_ARGS ${COMMON_EVAL_ARGS[@]}"
fi
echo "Final command: $CMD"
$CMD

# Goodbye.
echo "Evaluation finished"
echo "Now run 'python scripts/update_wandb.py $LOGS_ROOT --name $NAME --it $IT' to update wandb space."
echo "END TIME: $(date)"
