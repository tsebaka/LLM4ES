import datasets
import pandas as pd
import tqdm
import numpy as np
import torch

from src.utils.utils import DataCollatorWithUserIds, get_feature_preprocessor
from tqdm import tqdm
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

tqdm.pandas(leave=True)

def get_feature_preprocessor(
    config,
):
    class FeaturePreprocessor:
        def __init__(self, trx_category_path, mcc_path, mapping_path):
            categories = np.load(trx_category_path)
            self.trx_category_dict = {int(key): categories[key].item() for key in categories}
            
            mccs = pd.read_csv(mcc_path)
            self.mcc_dict = dict(zip(mccs["MCC_Code"], mccs["name"]))
    
            mapping = pd.read_csv(mapping_path)
            self.mapping = dict(zip(mapping['mcc'], mapping['_orig_mcc']))
    
        def preprocess(self, config, value, feature):
            if feature == "event_time":
                return str(np.int32(value))
            elif feature == "amount":
                return str(np.int32(value))
            elif feature == "trx_category":
                return self.trx_category_dict[value]
                # return str(value)
            elif feature == "mcc":
                mapped_value = self.mapping.get(value)
                if mapped_value is None:
                    raise ValueError(f"MCC value '{value}' not found in mapping.")
                return self.mcc_dict.get(mapped_value, "Unknown MCC")
                # return str(value)
            else:
                return str(value)
    
    preprocessor = FeaturePreprocessor(
        "/home/jovyan/zoloev-city/gigachat/source/ptls-experiments/scenario_rosbank/data/meta/trx_category_dict.npz",
        "/home/jovyan/zoloev-city/gigachat/source/ptls-experiments/scenario_rosbank/data/meta/MCC_Data_with_Translated_Names.csv",
        "/home/jovyan/zoloev-city/gigachat/source/ptls-experiments/scenario_rosbank/data/meta/part-00000-38e28af2-5979-458d-86e2-363a7cf328be-c000.csv"
    )
    return preprocessor

def trx_to_text_converter(
    config,
    transaction,
    preprocessor=None,
    tokenizer=None,
    chat=False,
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

    # if config.dataset.chat:
    #     messages = [
    #         {"role": "system", "content": config.dataset.chat_messages.system},
    #         {"role": "user", "content": config.dataset.chat_messages.user + text}
    #     ]

    #     text = tokenizer.apply_chat_template(
    #         messages, 
    #         add_generation_prompt=True,
    #         tokenize=False
    #     )

    return int(transaction[config.dataset.col_id]), text

import yaml         
from copy import deepcopy

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
    for ts in range(len(transaction[config.dataset.features[0]])):
        item = {}
        for feature in config.dataset.features:
            raw_val = transaction[feature][ts]
            item[feature] = preprocessor.preprocess(
                config,
                raw_val,
                feature
            )
        trx_records.append(item)

    yaml_obj = {"transactions": trx_records}
    text = yaml.dump(yaml_obj, **yaml_kwargs)

    if chat:
        messages = [
            {"role": "system", "content": config.dataset.chat_messages.system},
            {"role": "user",   "content": config.dataset.chat_messages.user + text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    return int(transaction[config.dataset.col_id]), text

import json
from copy import deepcopy 

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
    for ts in range(len(transaction[config.dataset.features[0]])):
        item = {}
        for feature in config.dataset.features:
            raw_val = transaction[feature][ts]
            item[feature] = preprocessor.preprocess(
                config,
                raw_val,
                feature
            )
        trx_records.append(item)

    json_obj = {"transactions": trx_records}
    text = json.dumps(json_obj, **json_kwargs)

    if chat:
        messages = [
            {"role": "system", "content": config.dataset.chat_messages.system},
            {"role": "user",   "content": config.dataset.chat_messages.user + text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    return int(transaction[config.dataset.col_id]), text

from html import escape

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
    for col in config.dataset.header_features:
        lines.append(indent(f"<th>{escape(col)}</th>", 2))
    lines.append(indent("</tr>", 1))

    n_rows = len(transaction[config.dataset.features[0]])
    for row_idx in range(n_rows):
        lines.append(indent("<tr>", 1))
        for feature in config.dataset.features:
            raw_val = transaction[feature][row_idx]
            val = preprocessor.preprocess(config, raw_val, feature)
            lines.append(indent(f"<td>{escape(str(val))}</td>", 2))
        lines.append(indent("</tr>", 1))

    lines.append("</table>")
    html_text = "\n".join(lines)

    if chat:
        messages = [
            {"role": "system", "content": config.dataset.chat_messages.system},
            {"role": "user",   "content": config.dataset.chat_messages.user + html_text},
        ]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    return int(transaction[config.dataset.col_id]), html_text

def trx_to_sql_converter(
    config,
    transaction: dict,
    preprocessor=None,
    tokenizer=None,
    chat: bool = False,
    table_name: str = "transactions",
    include_ddl: bool = False,
    batch_size: int | None = None,
):
    """
    Делает из одной записи (батч фичей по тайм-стемпам) строку SQL.
    Формат:
        [CREATE TABLE …;]
        INSERT INTO <table_name> VALUES
          (<row1>),
          (<row2>),
          …;
    Возвращает:
        • text  – если chat=False & inference=False
        • (client_id, text) – если inference=True
        • chat-template – если chat=True
    """

    def quote_val(v):
        """Кидаем кавычки только строкам, эскейпим одинарную кавычку."""
        if isinstance(v, (int, float)):
            return str(v)
        return "'" + str(v).replace("'", "''") + "'"

    cols = list(config.dataset.features)                
    col_names_sql = ", ".join(cols)

    n_rows = len(transaction[cols[0]])
    rows_sql = []
    for ts in range(n_rows):
        row_vals = [
            quote_val(
                preprocessor.preprocess(config, transaction[col][ts], col)
            )
            for col in cols
        ]
        rows_sql.append(f"({', '.join(row_vals)})")

    if batch_size is None:
        insert_chunks = [rows_sql]
    else:
        insert_chunks = [
            rows_sql[i : i + batch_size] for i in range(0, n_rows, batch_size)
        ]

    parts = []
    if include_ddl:
        ddl_cols = []
        for col in cols:
            col_type = "TEXT"
            sample = transaction[col][0]
            if isinstance(sample, (int, float)):
                col_type = "INTEGER"
            ddl_cols.append(f"  {col} {col_type}")
        parts.append(
            f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
            + ",\n".join(ddl_cols)
            + "\n);\n"
        )

    for chunk in insert_chunks:
        parts.append(
            f"INSERT INTO {table_name} ({col_names_sql}) VALUES\n  "
            + ",\n  ".join(chunk)
            + ";\n"
        )

    text = "".join(parts)

    if chat:
        messages = [
            {"role": "system", "content": config.dataset.chat_messages.system},
            {"role": "user",   "content": config.dataset.chat_messages.user + text},
        ]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    return int(transaction[config.dataset.col_id]), text


def trx_to_text_converter_grouped_by_mcc(
    config,
    transaction,
    preprocessor=None,
    tokenizer=None,
    chat=False,
):
    # from transformers import AutoTokenizer

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
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


def get_train_dataset(
    config,
):
    transactions = pd.read_parquet("/home/jovyan/zoloev-city/gigachat/source/ptls-experiments/scenario_rosbank/data/train_trx.parquet")
    preprocessor = get_feature_preprocessor(config)

    transactions_text = transactions.progress_apply(
        lambda row: trx_to_text_converter(config, row, preprocessor=preprocessor), 
        axis=1
    )

    transactions_yaml = transactions.progress_apply(
        lambda row: trx_to_yaml_converter(config, row, preprocessor=preprocessor), 
        axis=1
    )

    transactions_json = transactions.progress_apply(
        lambda row: trx_to_json_converter(config, row, preprocessor=preprocessor), 
        axis=1
    )

    transactions_html = transactions.progress_apply(
        lambda row: trx_to_html_converter(config, row, preprocessor=preprocessor), 
        axis=1
    )

    transactions_sql = transactions.progress_apply(
        lambda row: trx_to_sql_converter(config, row, preprocessor=preprocessor), 
        axis=1
    )
    
    return pd.concat((transactions_text, transactions_yaml, transactions_json, transactions_html, transactions_sql)).reset_index(drop=True)