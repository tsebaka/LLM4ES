import hydra
import numpy as np
import torch
import vllm
import warnings
import pandas as pd


from src.llm4trx.dataset.dataset import get_vllm_dataset
from src.llm4trx.utils.utils import get_vllm_model
from vllm import SamplingParams
from transformers import set_seed

warnings.filterwarnings("ignore")


def set_global_seed(config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.seed)
    set_seed(config.seed)


def vllm_inference(
    config,
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