WORK_DIR=$HOME/zoloev-city/gigachat
CONFIG_DIR=$WORK_DIR/source/llm-foundry/scripts/train/yamls/pretrain

source $WORK_DIR/source/llm-foundry/llmfoundry-venv/bin/activate

export WANDB_API_KEY=2736e3a99574e3049342cd33a3154aa307a08aa1
export WANDB_PROJECT="gigachat"
export WANDB_DIR=$WORK_DIR/checkpoints

python $WORK_DIR/source/llm-foundry/scripts/inference/convert_composer_to_hf.py \
    --config-dir $CONFIG_DIR \
    --config-name $CONFIG \
    variables.work_dir=$WORK_DIR
