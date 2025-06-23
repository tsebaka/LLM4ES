# LLM4Trx-research

## about
Этот репозиторий посвящён экспериментам с LLM на транзакционных данных

Включает код для моего [диплома](https://drive.google.com/file/d/1YDm5gYVeSLEMmF_wP3rEfPRPy-1fPvyy/view),
а также для экспериментах на открытых данных 
в ходе моей работы в Sber AI Lab команде [transactional deep learning](https://github.com/pytorch-lifestream)

in this repo:
* `source/` - source code
  * `llm4trx/` - HF-style multi-gpu llm train
    * `augmentation.py` - код для запуска vllm для генерации аугментаций
    * `pretrain.py` - для next-token-prediction LLM training
    * `inference.py` - multi-gpu llm inference
    * `convert_to_text.py` - конвертер в базовый формат с сохранением jsonl файла (для дальнейшей конвертации в стриминг)
    * `converters.py` - конвертеры в разные форматы
      * src/
        * `dataset.py` - всё что связано с обработкой датасета, будь то перевод в текст или создание DataLoader
        * `utils.py` - утилиты для получения моделей, эмбеддинга, подсчёта параметров модели и тд
  * `llm-foundry/` - fastest multi-gpu llm train
  * `ptls-experiments/` - data & downstream embeddings validation
* `scripts/` - scripts for running experiments & augmentations

# usage
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
