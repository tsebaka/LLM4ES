import warnings
import accelerate
import hydra
import numpy as np
import omegaconf
import torch
import wandb

from omegaconf import OmegaConf
from src.dataset.dataset import get_train_dataset
from src.utils.utils import (
    get_model,
    get_tokenizer,
    get_data_collator
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings("ignore")


def set_global_seed(config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.seed)
    set_seed(config.seed)


def train(
    config,
):
    tokenizer = get_tokenizer(config, train=True)
    model = get_model(config, train=True)
    model.train()

    if config.model.equal_pad_eos_id:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = get_data_collator(config, tokenizer)
    train_dataset = get_train_dataset(config, tokenizer)

    training_args = TrainingArguments(
        seed=config.seed,
        output_dir=config.log_dir,
        logging_dir=config.log_dir,
        report_to=config.args.report_to,
        logging_strategy=config.args.logging_strategy,
        logging_steps=config.args.logging_steps,
        save_strategy=config.args.save_strategy,
        save_steps=config.args.save_steps,
        bf16=config.args.bf16,
        num_train_epochs=config.args.train_epochs,
        per_device_train_batch_size=config.args.train_batch_size,
        warmup_steps=config.args.warmup_steps,
        learning_rate=config.args.learning_rate,
        max_grad_norm=config.args.max_grad_norm,
        lr_scheduler_type=config.args.lr_scheduler_type,
        weight_decay=config.args.weight_decay,
        adam_beta2=config.args.adam_beta2,
        adam_epsilon=config.args.adam_epsilon,
        optim=config.args.optim,
        per_device_eval_batch_size=config.args.eval_batch_size,
        do_eval=config.args.do_eval,
        gradient_accumulation_steps=config.args.gradient_accumulation_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    trainer.train()
    

@hydra.main(version_base=None, config_path="config")
def main(config):
    set_global_seed(config)
    train(config)


if __name__ == "__main__":
    main()