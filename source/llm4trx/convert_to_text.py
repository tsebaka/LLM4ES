import warnings
import hydra
import numpy as np
import omegaconf
import torch

from omegaconf import OmegaConf
from src.dataset.dataset import convertation
from src.utils.utils import (
    get_tokenizer,
)
from transformers import (
    set_seed,
)

warnings.filterwarnings("ignore")


def set_global_seed(config):
    torch.manual_seed(config.variables.global_seed)
    torch.cuda.manual_seed_all(config.variables.global_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.variables.global_seed)
    set_seed(config.variables.global_seed)


def convert_to_text(
    config,
):
    tokenizer = get_tokenizer(config)

    _ = convertation(config, tokenizer)


@hydra.main(version_base=None, config_path="config")
def main(config):
    set_global_seed(config)
    convert_to_text(config)


if __name__ == "__main__":
    main()
