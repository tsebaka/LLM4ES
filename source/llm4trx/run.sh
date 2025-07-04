eval "$(conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="llm4trx"

exp_name=config_name
log_dir=.../${exp_name}
checkpoint="checkpoint-id"
config_name="${exp_name}.yaml"

export WANDB_DIR=$log_dir

source llmfoundry-venv/bin/activate # для hf-style придётся доставить transformers нужной версии

# llm text augmentations
python -m dataset_preparing \
    --config-dir config \
    --config-name ${config_name} \
    ++exp_name=${exp_name} \
    ++log_dir=${log_dir} \
    ++dataset.presave=false

# ntp train
accelerate launch pretrain.py \
    --config-dir config \
    --config-name ${config_name} \
    ++exp_name=${exp_name} \
    ++log_dir=${log_dir} \
    ++dataset.presave=false

# inference
checkpoint="checkpoint-X"
accelerate launch inference.py \
    --config-dir config \
    --config-name ${config_name} \
    ++exp_name=${exp_name} \
    ++log_dir=${log_dir} \
    ++checkpoint=${checkpoint} \
    ++dataset.presave=false

# downstream validation
conda activate kaggle_kernel
cd .../ptls-experiments/scenario_rosbank
rm -r embeddings_validation.work/

pipenv run python -m embeddings_validation \
    --config-dir conf \
    --config-name embeddings_validation_baselines_unsupervised \
    +workers=10 \
    +total_cpu_count=20 \
    ++report_file=".../checkpoints-logs/${exp_name}/experiment_name.txt"
conda deactivate
