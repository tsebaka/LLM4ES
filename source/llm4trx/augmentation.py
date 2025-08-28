import hydra
import numpy as np
import torch
import vllm
import warnings
import pandas as pd


from src.llm4trx.dataset.dataset import get_vllm_dataset
from src.llm4trx.utils.utils import (
    get_vllm_model,
    set_global_seed
)
from vllm import SamplingParams
from transformers import set_seed

warnings.filterwarnings("ignore")


def vllm_inference(
    config
):
    model = get_vllm_model(config)
    tokenizer = model.get_tokenizer()
    dataset = get_vllm_dataset(config, tokenizer)
    
    sampling_params = SamplingParams(
        temperature=config.dataset.sampling.temperature,
        top_p=config.dataset.sampling.top_p, 
        max_tokens=config.dataset.sampling.max_tokens
    )
    outputs = model.generate(dataset, sampling_params)

    dataset = pd.DataFrame({
        "augmentation": [out.outputs[0].text for out in outputs]
    })
    dataset.to_csv(
        "./assets/" + config.dataset.name + f"/{config.exp_name}_augmentation.csv",
        index=False
    )


@hydra.main(version_base=None, config_path="config")
def main(config):
    set_global_seed(config)
    vllm_inference(config)


if __name__ == "__main__":
    main()
