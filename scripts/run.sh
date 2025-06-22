WORK_DIR=$HOME/zoloev-city/gigachat
CONFIG_DIR=$WORK_DIR/source/llm-foundry/scripts/train/yamls/pretrain

source $WORK_DIR/source/llm-foundry/llmfoundry-venv/bin/activate

export WANDB_API_KEY=2736e3a99574e3049342cd33a3154aa307a08aa1
export WANDB_PROJECT="gigachat"
export WANDB_DIR=$WORK_DIR/checkpoints


CONFIG=age-ntp-llama-3.2-3b-pandas.yaml
echo "========== starting... $CONFIG =========="

echo "========== convert to text... =========="
python $WORK_DIR/source/llm4trx/convert_to_text.py \
    --config-dir $CONFIG_DIR \
    --config-name $CONFIG \
    variables.work_dir=$WORK_DIR

echo "========== convert to streaming... =========="
python $WORK_DIR/source/llm-foundry/scripts/data_prep/convert_dataset_json.py \
    --config-dir $CONFIG_DIR \
    --config-name $CONFIG \
    variables.work_dir=$WORK_DIR

echo "========== training llm foundry... =========="
composer $WORK_DIR/source/llm-foundry/scripts/train/train.py \
    $CONFIG_DIR/$CONFIG \
    variables.work_dir=$WORK_DIR

echo "========== convert model to hf... =========="
python $WORK_DIR/source/llm-foundry/scripts/inference/convert_composer_to_hf.py \
    --config-dir $CONFIG_DIR \
    --config-name $CONFIG \
    variables.work_dir=$WORK_DIR

echo "========== inference... =========="
accelerate launch $WORK_DIR/source/llm4trx/inference.py \
    --config-dir $CONFIG_DIR \
    --config-name $CONFIG \
    variables.work_dir=$WORK_DIR
    
echo "========== completed! =========="
