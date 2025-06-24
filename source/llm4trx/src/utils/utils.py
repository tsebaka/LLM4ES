import torch
import transformers
import flash_attn
import numpy as np
import pandas as pd

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding
)
from vllm import LLM


def get_data_collator(
    config,
    tokenizer
):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    return data_collator
    

def get_tokenizer(
    config,
):
    tokenizer = AutoTokenizer.from_pretrained(
        config.variables.tokenizer_name,
    )
    return tokenizer


def get_model(
    config,
    train=True
):
    attn_implementation = "flash_attention_2" if config.model.use_flash_attention_2 else None
    torch_dtype = torch.bfloat16 if config.precision == "amp_bf16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        config.variables.model_convertation.hf_output_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype
    )
    return model



def get_vllm_model(
    config
):
    torch_dtype = torch.bfloat16 if config.model.use_bfloat16 else torch.float16
    
    model = LLM(
        model=config.model.marking_path, 
        dtype=torch_dtype,
        trust_remote_code=True,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.95,
    )
    return model


def get_embedding(
    config,
    batch, 
    model
):
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True).hidden_states
    
    hidden_states = torch.stack(outputs[config.variables.inference.from_layer_slice:]).mean(dim=0)
    
    idx_of_the_last_non_padding_token = batch["attention_mask"].bool().sum(1)
    embeddings = torch.stack([
        embedding[:idx_of_the_last_non_padding_token[pos]].mean(0)
        for pos, embedding in enumerate(hidden_states)
    ])
    torch.cuda.empty_cache()
    return embeddings


class DataCollatorWithUserIds:
    def __init__(
        self,
        tokenizer,
    ):
        self.data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
        )

    def __call__(self, features):
        user_ids = [sample["user_id"] for sample in features]
        features = [sample["inputs"] for sample in features]
        batch = self.data_collator(features)
        batch["user_ids"] = torch.tensor(user_ids)
        return batch


def get_feature_preprocessor(
    config,
):
    if config.variables.dataset_name == "rosbank":
        class FeaturePreprocessor:
            def __init__(self, trx_category_path, mcc_path, mapping_path):
                categories = np.load(trx_category_path)
                self.trx_category_dict = {int(key): categories[key].item() for key in categories}
                
                mccs = pd.read_csv(mcc_path)
                self.mcc_dict = dict(zip(mccs["MCC_Code"], mccs["name"]))
                self.mcc_values = list(self.mcc_dict.values())
                mapping = pd.read_csv(mapping_path)
                self.mapping = dict(zip(mapping['mcc'], mapping['_orig_mcc']))
        
            def preprocess(self, config, value, feature):
                if feature == "event_time":
                    return str(np.int32(value))
                elif feature == "amount":
                    return str(np.int32(value))
                elif feature == "trx_category":
                    return self.trx_category_dict[value]
                elif feature == "mcc":
                    mapped_value = self.mapping.get(value)
                    if mapped_value is None:
                        raise ValueError(f"MCC value '{value}' not found in mapping.")
                    return self.mcc_dict.get(mapped_value, "Unknown MCC")
                else:
                    return str(value)
        
        preprocessor = FeaturePreprocessor(
            f"{config.variables.data_path}/meta/trx_category_dict.npz",
            f"{config.variables.data_path}/meta/MCC_Data_with_Translated_Names.csv",
            f"{config.variables.data_path}/meta/part-00000-38e28af2-5979-458d-86e2-363a7cf328be-c000.csv"
        )
    elif config.variables.dataset_name == "gender":
        class FeaturePreprocessor:
            def __init__(self, mcc_path, tr_type_path):
                self.mcc_dict = np.load(mcc_path, allow_pickle=True).item()

                self.tr_type_dict = np.load(tr_type_path, allow_pickle=True).item()

            def preprocess(self, config, value, feature):
                if feature == "mcc_code":
                    return str(self.mcc_dict[value])
                elif feature == "tr_type":
                    return str(self.tr_type_dict[value])
                elif feature == "event_time":
                    return str(np.int32(value))
                elif feature == "amount":
                    return str(np.int32(value))
                else:
                    return str(value)
                    
        
        preprocessor = FeaturePreprocessor(
            "/home/jovyan/zoloev-city/gigachat/source/script/assets/gender/mcc_dict.npy",
            "/home/jovyan/zoloev-city/gigachat/source/script/assets/gender/tr_type_dict.npy",
        )
    elif config.variables.dataset_name == "age_pred" or config.variables.dataset_name == "mixed":
        class FeaturePreprocessor:
            def __init__(self, mcc_path):
                self.mcc_dict = np.load(mcc_path, allow_pickle=True).item()
        
            def preprocess(self, config, value, feature):
                if feature == "event_time":
                    return str(np.int32(value))
                elif feature == "amount_rur":
                    return str(np.round(value, 2))
                elif feature == "small_group":
                    return str(self.mcc_dict[value])
                else:
                    return str(value)
        
        preprocessor = FeaturePreprocessor(
            f"{config.variables.data_path}/meta/small_group_dict.npy",
        )

    return preprocessor


def coles_concat(
    config,
    llm_embeddings
):
    coles_embeddings = pd.read_pickle(config.dataset.coles_embeddings_path)
    
    llm_embeddings[config.dataset.col_id] = llm_embeddings[config.dataset.col_id].astype(str)
    coles_embeddings[config.dataset.col_id] = coles_embeddings[config.dataset.col_id].astype(str)
    new_column_names = {
        f"emb_{i}": f"emb_{i + 1023}" for i in range(1, len(llm_embeddings.columns))
    }
    llm_embeddings.rename(columns=new_column_names, inplace=True)
    
    coles_llm = coles_embeddings.merge(
        llm_embeddings,
        on=config.dataset.col_id, 
        how="inner"
    )
    coles_llm.to_parquet(f"{config.log_dir}/concat_coles_llm_{config.checkpoint}.parquet")


def count_model_parameters(
    model
):
    total_params = sum(p.numel() for p in model.parameters())
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")


def freeze(
    module
):
    for parameter in module.parameters():
        parameter.requires_grad = False
