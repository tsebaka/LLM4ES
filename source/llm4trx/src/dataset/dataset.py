import peft
import pandas as pd
import numpy as np
import torch
import yaml
import json
import html

from src.utils.utils import DataCollatorWithUserIds, get_feature_preprocessor
from tqdm import tqdm
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader        
from copy import deepcopy
from html import escape

tqdm.pandas(leave=True)


def trx_to_yaml_converter(
    config,
    transaction: dict,
    preprocessor=None,
    tokenizer=None,
    chat: bool=False,
    yaml_kwargs: dict | None = None,
):
    """
    Перегоняет список фичей по тайм-стемпам в YAML-вид:
    
    transactions:
      - date: 17140
        amount: 4
        category: Banking and Finance Services
        ...
      - ...
    """
    if yaml_kwargs is None:
        yaml_kwargs = dict(
            Dumper   = yaml.SafeDumper,
            allow_unicode = True,
            default_flow_style = False,
            sort_keys = False,
        )

    trx_records = []
    for ts in range(len(transaction[config.variables.dataset.features[0]])):
        item = {}
        for feature in config.variables.dataset.features:
            raw_val = transaction[feature][ts]
            item[feature] = preprocessor.preprocess(
                config,
                raw_val,
                feature
            )
        trx_records.append(item)

    yaml_obj = {"transactions": trx_records}
    text = yaml.dump(yaml_obj, **yaml_kwargs)

    return int(transaction[config.variables.dataset.col_id]), text


def trx_to_json_converter(
    config,
    transaction: dict,
    preprocessor=None,
    tokenizer=None,
    chat: bool = False,
    json_kwargs: dict | None = None,  
):
    """
    Превращает одну «сырую» запись с batched-фичами по таймстемпам
    в строку JSON вида

    {
      "transactions": [
        { "date": 17140, "amount": 4, ... },
        ...
      ]
    }
    """
    if json_kwargs is None:
        json_kwargs = dict(
            ensure_ascii = False,
            indent       = 2,      
        )

    trx_records = []
    for ts in range(len(transaction[config.variables.dataset.features[0]])):
        item = {}
        for feature in config.variables.dataset.features:
            raw_val = transaction[feature][ts]
            item[feature] = preprocessor.preprocess(
                config,
                raw_val,
                feature
            )
        trx_records.append(item)

    json_obj = {"transactions": trx_records}
    text = json.dumps(json_obj, **json_kwargs)

    return int(transaction[config.variables.dataset.col_id]), text


def trx_to_html_converter(
    config,
    transaction: dict,
    preprocessor=None,
    tokenizer=None,
    chat: bool = False,
    table_attrs: str | None = 'border="1" cellspacing="0" cellpadding="4"',
):
    """
    Превращает батч транзакций в человеко-читаемую HTML-таблицу:
    
    <table …>
      <tr>
        <th>…</th>
        …
      </tr>
      <tr>
        <td>…</td>
        …
      </tr>
      …
    </table>
    """

    def indent(line: str, lvl: int) -> str:
        return "  " * lvl + line     

    lines = []
    table_open = f"<table {table_attrs}>" if table_attrs else "<table>"
    lines.append(table_open)

    lines.append(indent("<tr>", 1))
    for col in config.variables.dataset.header_features:
        lines.append(indent(f"<th>{escape(col)}</th>", 2))
    lines.append(indent("</tr>", 1))

    n_rows = len(transaction[config.variables.dataset.features[0]])
    for row_idx in range(n_rows):
        lines.append(indent("<tr>", 1))
        for feature in config.variables.dataset.features:
            raw_val = transaction[feature][row_idx]
            val = preprocessor.preprocess(config, raw_val, feature)
            lines.append(indent(f"<td>{escape(str(val))}</td>", 2))
        lines.append(indent("</tr>", 1))

    lines.append("</table>")
    html_text = "\n".join(lines)

    return int(transaction[config.variables.dataset.col_id]), html_text


def trx_to_text_converter_grouped_by_mcc(
    config,
    transaction,
    preprocessor=None,
    tokenizer=None,
    chat=False,
):
    from collections import defaultdict

    features = [f for f in config.variables.dataset.features if f != "small_group"]
    header = config.variables.dataset.header_separator.join(
        [f for f in config.variables.dataset.header_features if f != "small_group"]
    )

    trx_per_category = defaultdict(list)

    for idx, category in enumerate(transaction["small_group"]):
        trx_per_category[category].append(idx)

    sections = [f"{header}"]

    for cat_code, indices in trx_per_category.items():
        readable_category = preprocessor.preprocess(config, cat_code, "small_group")
        lines = [
            config.variables.dataset.feature_separator.join(
                preprocessor.preprocess(config, transaction[feature][i], feature)
                for feature in features
            ) for i in indices
        ]
        section = f"\n\n### Category: {readable_category}\n" + "\n".join(lines)
        sections.append(section)

    text = "\n".join(sections)

    if chat:
        messages = [
            {"role": "system", "content": config.variables.dataset.chat_messages.system},
            {"role": "user", "content": config.variables.dataset.chat_messages.user + text}
        ]
        text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False
        )

    return int(transaction[config.variables.dataset.col_id]), text


def trx_to_text_converter(
    config,
    transaction,
    preprocessor=None,
    tokenizer=None,
    chat=False,
):

    header = config.variables.dataset.header_separator.join(config.variables.dataset.header_features)

    transactions = [header] + [
        config.variables.dataset.feature_separator.join(
            preprocessor.preprocess(
                config, 
                transaction[feature][timestamp], 
                feature
            ) for feature in list(config.variables.dataset.features)
        ) for timestamp in range(len(transaction[config.variables.dataset.features[0]])) # len(transaction[config.variables.dataset.features[0]])
    ]
    text = config.variables.dataset.trx_separator.join(transactions)

    if chat:
        messages = [
            {"role": "system", "content": config.variables.dataset.chat_messages.system},
            {"role": "user", "content": config.variables.dataset.chat_messages.user + text}
        ]

        text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False
        )

    return int(transaction[config.variables.dataset.col_id]), text


def convertation(
    config,
    tokenizer
):
    if config.variables.text_convertation.use_augmentation:
        return True
    transactions = pd.read_parquet(config.variables.dataset.train_path)
    preprocessor = get_feature_preprocessor(config)
    
    transactions_text = transactions.progress_apply(
        lambda row: trx_to_text_converter_grouped_by_mcc(config, row, preprocessor=preprocessor), 
        axis=1
    )
    transactions_text = pd.DataFrame(list(transactions_text), columns=["user_id", "text"])

    transactions_text.to_json(
        config.variables.text_convertation.out_file,
        orient="records", 
        lines=True
    )
    
    return transactions_text


def get_inference_dataset(
    config, 
    tokenizer
):
    transactions = pd.concat((
        pd.read_parquet(config.variables.dataset.train_path),
        pd.read_parquet(config.variables.dataset.test_path))
    ).reset_index(drop=True)

    preprocessor = get_feature_preprocessor(config)
    
    transactions_text = transactions.progress_apply(
        lambda row: trx_to_text_converter(config, row, preprocessor=preprocessor), 
        axis=1
    )
    transactions_text = pd.DataFrame(list(transactions_text), columns=["user_id", "inputs"])
    
    transactions_text["inputs"] = '''You are an AI model trained to analyze financial transaction sequences.
    Given the following list of user transactions, predict whether the user is likely to stop using the service soon (i.e., churn).''' + transactions_text["inputs"]

    transactions_text.to_csv(
        f'{config.variables.data_local}/transactions_text_inference.csv',
        index=False
    )

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
