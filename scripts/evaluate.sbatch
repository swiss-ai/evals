#!/bin/bash
#SBATCH --account=infra01
#SBATCH --gres=gpu:4
#SBATCH --environment=./containers/env.toml

MODEL=${2:-"/capstor/store/cscs/swissai/a139/checkpoints/qwen3_moe_dsa_test_hf"}
NAME="${MODEL##*/}"
TOKENIZER="swiss-ai/Apertus-70B-2509"
TASKS=${1:-"hellaswag,ai2_arc,lambada_openai,winogrande,piqa,openbookqa,commonsense_qa,mmlu,gsm8k,wikitext,lambada,squadv2,include_base_44,global_mmlu,xcopa,xnli,xwinograd,pawsx,m_arc,m_hellaswag"}
WANDB_ENTITY=${WANDB_ENTITY:-"mariagrandury-epflnlp"}
WANDB_PROJECT=${WANDB_PROJECT:-"moe-ablations-evals"}
BOS=true
MEGATRON_BRANCH="81d44dbe25c22190989fa8a12284bdb8d1303b40"
ITS=(0120000)
TOK_PER_IT=$(( 512 * 4096 ))
IT=120000
TOKENS_PER_ITER=$(( 512 * 4096 ))
BS=${BS:-"auto:20"}
MEGATRON_BRANCH=${MEGATRON_BRANCH:-main}
SIZE=${SIZE:-1}
BOS=${BOS:-false}
LOGS_ROOT=${LOGS_ROOT:-$SCRATCH/eval-logs}
TRANSFORMERS_BRANCH=${TRANSFORMERS_BRANCH:-main}
HARNESS_BRANCH=${HARNESS_BRANCH:-main}
export HF_HOME=${HF_HOME:-/capstor/store/cscs/swissai/infra01/hf_home/}
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
echo "Installing standard transformers and custom eval harness..."
cd $TEMP_REPOS

# 1. Install standard transformers with Qwen3 MoE support
pip install "transformers==4.57.5"

# 2. Keep the custom LM-Harness (since you likely still need it for 'swissai_eval' tasks)
echo "Cloning harness FORK"
git clone https://github.com/swiss-ai/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout main # $HARNESS_BRANCH
pip install -e .
cd $SLURM_SUBMIT_DIR

HF_CHECKPOINT_PATH=$MODEL
if [[ -d $MODEL ]]; then
	echo "$MODEL is a local directory, skipping download."
else
	hf download $MODEL $MAYBE_DOWNLOAD_REVISION
fi
if [ $TOKENIZER != $MODEL ]; then
	echo "Why are you using a different tokenizer and model? TOKENIZER=$TOKENIZER, MODEL=$MODEL"
	hf download $TOKENIZER
fi
# export HF_HUB_OFFLINE=0

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
if [ $WANDB_PROJECT != "Null" ]; then
	echo "Now running 'python scripts/update_wandb.py $LOGS_ROOT --name $NAME --it $IT' to update wandb space."
	python scripts/update_wandb.py $LOGS_ROOT --name $NAME --it $IT
	echo "END TIME: $(date)"
else 
	echo "Now run 'python scripts/update_wandb.py $LOGS_ROOT --name $NAME --it $IT' to update wandb space."
	echo "END TIME: $(date)"
fi
