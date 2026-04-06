srun --environment=./containers/env.toml --pty -A a-infra01-1 --gres=gpu:4 -t 02:00:00 bash


for task in hellaswag ai2_arc lambada_openai winogrande piqa openbookqa commonsense_qa mmlu gsm8k wikitext lambada squadv2 include_base_44 global_mmlu xcopa xnli xwinograd pawsx m_arc m_hellaswag; do
    sbatch --job-name="eval_${task}" --account=infra01 --output="logs/eval_${task}_%j.out" scripts/evaluate.sbatch "${task}"
done


sbatch --job-name="eval_test" --account=infra01 --output="logs/eval_test_%j.out" scripts/evaluate.sbatch m_hellaswag




MODEL CHECKPOINTS

/capstor/store/cscs/swissai/a139/checkpoints/moe_runs

TASKS
hellaswag ai2_arc lambada_openai winogrande piqa openbookqa commonsense_qa mmlu gsm8k wikitext lambada squadv2 include_base_44 global_mmlu xcopa xnli xwinograd pawsx m_arc m_hellaswag


hellaswag m_hellaswag
ai2_arc m_arc
winogrande xwinograd
lambada lambada_openai
piqa 
xnli
xcopa
pawsx  
openbookqa
commonsense_qa
mmlu global_mmlu
gsm8k (generate_until)
wikitext
squadv2 (generate_until)
include_base_44


Leandro: commonsense QA, hellaswag, openbook QA, PiQA

FAILED
- include_base_44
- squadv2
- review xcopa





PUSH EVAL RESULTS TO WANDB

export WANDB_ENTITY="mariagrandury-epflnlp"
export WANDB_PROJECT="moe-ablations-evals"

python scripts/update_wandb.py $LOGS_ROOT --name $NAME --it $IT

python scripts/update_wandb.py /iopsstor/scratch/cscs/mariagrandury/eval-logs --name qwen3_moe_dsa_test_hf --it 120000






CONVERT MEGATRON TO HF

Original:

srun --jobid=convert_mla_test_to_hf --nodes=1 --ntasks-per-node=1 --mpi=pmix \
--environment=/capstor/store/cscs/swissai/a139/containers/ngc_25-11-nemo-alps1.toml \
    bash -c "PYTHONPATH=/iopsstor/scratch/cscs/ntazi/projects/Megatron-Bridge/src:\$PYTHONPATH \
    python /iopsstor/scratch/cscs/ntazi/projects/Megatron-Bridge/convert_mla_test_to_hf.py"

Custom:

srun --nodes=1 --ntasks-per-node=1 --mpi=pmix --account=infra01 \
  --environment=/capstor/store/cscs/swissai/a139/containers/nemo-alps2.toml \
  bash -lc "PYTHONPATH=/iopsstor/scratch/cscs/ntazi/projects/Megatron-Bridge/src:$PYTHONPATH \
  python /iopsstor/scratch/cscs/mariagrandury/Megatron-Bridge/convert_mla_test_to_hf.py \
  --checkpoint 256n_gbs4096_muon_localattn"

Current error:

srun: error: Unable to allocate resources: Invalid account or account/partition combination specified
