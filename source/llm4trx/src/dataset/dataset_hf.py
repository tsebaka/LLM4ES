import peft
import pandas as pd
import numpy as np
import torch

from src.utils.utils import DataCollatorWithUserIds, get_feature_preprocessor
from tqdm import tqdm
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

tqdm.pandas(leave=True)


def trx_to_text_converter(
    config,
    transaction,
    preprocessor=None,
    tokenizer=None,
    chat=False,
    inference=False,
):
    header = config.dataset.header_separator.join(config.dataset.header_features)

    transactions = [header] + [
        config.dataset.feature_separator.join(
            preprocessor.preprocess(
                config, 
                transaction[feature][timestamp], 
                feature
            ) for feature in list(config.dataset.features)
        ) for timestamp in range(len(transaction[config.dataset.features[0]]))
    ]
    text = config.dataset.trx_separator.join(transactions)


    if chat:
        messages = [
            {"role": "system", "content": config.dataset.chat_messages.system},
            {"role": "user", "content": config.dataset.chat_messages.user + text}
        ]

        text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False
        )
        return text

    if inference:
        return int(transaction[config.dataset.col_id]), text
    return text


def get_vllm_dataset(
    config,
    tokenizer
):  
    transactions = pd.read_parquet(config.dataset.train_path)
    if config.dataset.debug:
        transactions = transactions[0:5]
    preprocessor = get_feature_preprocessor(config)
    
    transactions = transactions.progress_apply(
        lambda row: trx_to_text_converter(config, row, preprocessor=preprocessor, tokenizer=tokenizer, chat=True, inference=False), 
        axis=1
    )
    transactions.to_csv("assets/" + config.dataset.name + "/marking_dataset.csv", index=False)

    return transactions
    

def tokenize_function(
    config, 
    inputs,
    tokenizer
):
    tokens = tokenizer(
        inputs["prompt"],
        padding="max_length", 
        truncation=True, 
        max_length=config.model.max_length
    )
  
    return tokens


def get_train_dataset(
    config, 
    tokenizer
):
    transactions = pd.read_parquet(config.dataset.train_path)

    preprocessor = get_feature_preprocessor(config)
    if config.dataset.marked_dataset:
        print("marked!")
        vllm_text_dataset = pd.read_csv("assets/" + config.dataset.name + "/transactions_text_marked_150.csv").rename({"0": "out"}, axis=1)
        hf_dataset = Dataset.from_pandas(pd.DataFrame({"prompt": vllm_text_dataset["out"]}))
    else:
        transactions = transactions.progress_apply(
            lambda row: trx_to_text_converter(config, row, preprocessor=preprocessor, inference=False), 
            axis=1
        )
        transactions.to_csv("assets/" + config.dataset.name + "/transactions_text.csv", index=False)
        hf_dataset = Dataset.from_pandas(pd.DataFrame({"prompt": transactions}))

    
    dataset = hf_dataset.map(
        lambda inputs: tokenize_function(config, inputs, tokenizer), 
        batched=False, 
        # num_proc=4
    )
    
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return dataset


def get_inference_dataset(
    config, 
    tokenizer
):
    transactions = pd.concat((
        pd.read_parquet(config.dataset.train_path),
        pd.read_parquet(config.dataset.test_path))
    ).reset_index(drop=True)

    preprocessor = get_feature_preprocessor(config)

    if config.dataset.presave:
        transactions_text = pd.read_csv("assets/" + config.dataset.name + "/transactions_text_inference.csv")
    else:
        transactions_text = transactions.progress_apply(
            lambda row: trx_to_text_converter(config, row, preprocessor=preprocessor, inference=True), 
            axis=1
        )
        transactions_text = pd.DataFrame(list(transactions_text), columns=["user_id", "inputs"])
        transactions_text.to_csv("assets/" + config.dataset.name + "/transactions_text_inference.csv", index=False)
    

    tokenized_transactions = [
    {
        "user_id": user_id,
        "inputs": tokenizer(
            prompt,   
            max_length=16384,
            truncation=True,
        )} for user_id, prompt in tqdm(transactions_text.values)
    ]
            
    collator = DataCollatorWithUserIds(tokenizer)

    inference_loader = DataLoader(
        dataset=tokenized_transactions, 
        collate_fn=collator, 
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )
    return inference_loader
