from transformers import BertTokenizer, BertForMaskedLM,RobertaForMaskedLM , RobertaTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from pathlib import Path
from datasets import load_dataset,DatasetDict
from src.utils import *
import json



def load_model(model_name,mlm=False,MODEL_DIR=None,num_classes=2):
    if mlm:
        if model_name.startswith("bert"):
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForMaskedLM.from_pretrained(model_name)
        elif model_name.startswith("roberta"):
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaForMaskedLM.from_pretrained(model_name)
    else:
        if MODEL_DIR is None:
            raise ValueError("MODEL_DIR must be specified for loading the classifier model.")
        if model_name.startswith("bert"):
            tokenizer = BertTokenizer.from_pretrained(Path(MODEL_DIR+"_tokenizer"))
            model = BertForSequenceClassification.from_pretrained(Path(MODEL_DIR+"_model"), num_labels=num_classes)
        elif model_name.startswith("roberta"):
            tokenizer = RobertaTokenizer.from_pretrained(Path(MODEL_DIR+"_tokenizer"))
            model = RobertaForSequenceClassification.from_pretrained(Path(MODEL_DIR+"_model"), num_labels=num_classes)
    return model, tokenizer

def load_mlm(tokenizer):
    dataset = load_dataset('AstroAure/dogwhistle_dataset')
    dataset = DatasetDict({
        "train": dataset["train_bhr"],
        "test": dataset["test_bhr"]
    })

    tokenized_datasets = dataset.map(lambda examples: mask_tokens(examples,tokenizer), batched=True)
                                 
    if 'label' in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.remove_columns('label')

    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
    return train_dataset, eval_dataset


def load_classifier(tokenizer,hidden=False):
    # Load dataset
    dataset = load_dataset('AstroAure/dogwhistle_dataset')
    dataset = DatasetDict({
        "train": dataset["train_bhr"],
        "test": dataset["test_bhr"],
        "test_hidden": dataset["test_hidden"]
    })

    # Tokenize the dataset
    tokenized_datasets = dataset.map(lambda example : tokenize_function(example,tokenizer), batched=True)

    # Prepare the data for training
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)  # Example with 1000 samples
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)  # Example with 500 samples
    if hidden:
        hidden_dataset = tokenized_datasets["test_hidden"].shuffle(seed=42)
        return hidden_dataset
    return train_dataset, eval_dataset

def load_classifier_2(tokenizer,hidden=False,dict_ingroup=None):
    # Load dataset
    dataset = load_dataset('AstroAure/dogwhistle_dataset')
    dataset = DatasetDict({
        "train": dataset["train_bhr"],
        "test": dataset["test_bhr"],
        "test_hidden": dataset["test_hidden"]
    })

    # Tokenize the dataset
    tokenized_datasets = dataset.map(lambda example : tokenize_function(example,tokenizer), batched=True)

    filtered_dataset = tokenized_datasets.filter(lambda example: example["label"] == 1)
    if dict_ingroup is None:
        # Create a dictionary to map ingroup labels to integers
        dict_ingroup = {k:i for i,k in enumerate(set(filtered_dataset["train"]["ingroup"]))}

    
    filtered_dataset = filtered_dataset.map(lambda example : update_label(example,dict_ingroup))
    train_dataset = filtered_dataset["train"].shuffle(seed=42)
    eval_dataset = filtered_dataset["test"].shuffle(seed=42)

    if hidden:
        hidden_dataset = filtered_dataset["test_hidden"].shuffle(seed=42)
        return hidden_dataset,dict_ingroup

    return train_dataset, eval_dataset , dict_ingroup