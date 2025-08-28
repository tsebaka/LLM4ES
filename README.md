# LLM4ES
<p align="center">
  <img src="assets/llm4trx-logo.png" alt="llm4trx" width="1000"/>
</p>

## about
This repository is dedicated to experiments with LLMs on transactional datasets: [Rosbank](https://github.com/pytorch-lifestream/ptls-experiments/tree/main/scenario_rosbank), [Age](https://github.com/pytorch-lifestream/ptls-experiments/tree/main/scenario_age_pred), [Gender](https://github.com/pytorch-lifestream/ptls-experiments/tree/main/scenario_gender).

In this repo:
* `source/` - source code
  * `llm4trx/` - HF-style multi-gpu LLM training
    * `augmentation.py` - code for launching vllm to generate augmentations
    * `pretrain.py` - for next-token-prediction LLM training
    * `inference.py` - multi-gpu LLM inference
    * `convert_to_text.py` - converter to base format, saving as jsonl (for later conversion to streaming)
    * `converters.py` - converters to different formats
      * `src/`
        * `dataset.py` - everything related to dataset processing: text conversion, DataLoader creation, etc.
        * `dataset_hf.py` - dataset version for HF-style training and augmentations
        * `utils.py` - utilities for loading models, embeddings, counting model parameters, etc.
  * `llm-foundry/` - fastest multi-gpu LLM training
  * `ptls-experiments/` - data & downstream embeddings validation
* `scripts/` - scripts for running experiments and augmentations
  * `convert_to_text.sh` - converts arrays of transactions into text format, then into MosaicML Streaming format
  * `train.sh.sh` - multi-gpu LLM training for next-token prediction
  * `model_convertation.sh` - convert model from MosaicML Composer format into HF Transformers format
  * `inference.sh` - multi-gpu inference
  * `run.sh` - full pipeline with given seed
  * `run_multi_seed.sh` - full pipeline with multi-seed
* `source/llm4trx`
  * `run.sh` - run the entire pipeline based on HF Transformers

The three main configs (one per dataset) are located in:  
`source/llm-foundry/scripts/train/yamls/pretrain`

Configs (HF versions, currently used for augmentations) are located in:  
`source/llm4trx/config`

## code
In [llm-foundry](https://github.com/mosaicml/llm-foundry/tree/main) argparse is used by default, which is not very convenient.  
I rewrote part of their code to make it possible to use Hydra and configs more easily.  
I also added to their [ConcatTokensDataset](https://github.com/tsebaka/llm-foundry/blob/c70a4847463da8859d7236874ad6705285460f1a/llmfoundry/data/data.py) the ability to truncate by `max_length` instead of just `concat_tokens` (because the library is used for pretraining LLMs from scratch, where max_length truncation makes sense).  
This is the only difference between the original library and my fork used in this repo.

Some differences between two training variants:
| Parameter  | transformers | llm-foundry |
|-----------|-----------|-----------|
|augmentations|[vllm](https://github.com/vllm-project/vllm)|[vllm](https://github.com/vllm-project/vllm)|
|dataset |.csv converted into HF dataset|.jsonl converted into [Streaming dataset](https://github.com/mosaicml/streaming)|
| FSDP  | - | + |
| inference  | multi gpu [accelerate](https://github.com/huggingface/accelerate)  | multi gpu [accelerate](https://github.com/huggingface/accelerate)  |
| model  | Hugging Face AutoModel  | MosaicML Composer  |
| speed (on Rosbank dataset)  | 2.5h | 1.3h  |
| ease of adding details | highly flexible | hard to add new features without rewriting |
| initial experiments | + | - |

## usage

### с Docker'ом (лучше всего)
image: https://hub.docker.com/orgs/mosaicml/repositories.
<!--pytest.mark.skip-->
```bash
git clone https://github.com/mosaicml/llm-foundry.git
cd llm-foundry
pip install -e ".[gpu]"
```

### настройка окружения (без Docker'а)
```sh
git clone https://github.com/tsebaka/LLM4Trx-research.git
cd LLM4Trx-research
cd source

# prep ptls-experiments
cd ptls-experiments
python3 -m venv ptls-venv
source ptls-venv/bin/activate
pip install pytorch-lifestream
cd ..

# prep llm-foundry
cd llm-foundry
python3 -m venv llmfoundry-venv
source llmfoundry-venv/bin/activate
pip install cmake packaging torch
pip install -e ".[gpu]"
pip install deepspeed=0.15.4
cd ..
```

### llm-foundry training
```sh

WORK_DIR=$HOME/zoloev-city/exp_name
CONFIG_DIR=$WORK_DIR/source/llm-foundry/scripts/train/yamls/pretrain

source $WORK_DIR/source/llm-foundry/llmfoundry-venv/bin/activate

export WANDB_API_KEY=2736e3a99574e3049342cd33a3154aa307a08aa1
export WANDB_PROJECT="llm4trx"
export WANDB_DIR=$WORK_DIR/checkpoints


CONFIG=config_name
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
```

### hf-style training & augmentations
```sh
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
accelerate launch sft_train.py \
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
```

## Hardware
В моём распоряжении было:

- 8x NVIDIA A100 GPUs (80 GB HBM2e per GPU)
- 1 TB of DDR4 RAM

## Results

### Throughput
<p align="center">
  <img src="assets/throughput.png" alt="llm4trx" width="500"/>
</p>

### Loss
<p align="center">
  <img src="assets/loss.png" alt="llm4trx" width="500"/>
</p>

### Metrics
<p align="center">
  <img src="assets/results.png" alt="llm4trx" width="500"/>
</p>
