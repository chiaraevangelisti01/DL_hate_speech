from transformers import BertTokenizer, BertForMaskedLM,RobertaForMaskedLM , RobertaTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from pathlib import Path
from datasets import load_dataset, DatasetDict
from src.utils import *
import json

"""
This module provides functions to load models and datasets for training and evaluation.
It supports both masked language modeling (MLM) and sequence classification tasks using BERT and RoBERTa models.
"""

def load_model(model_name,mlm=False,MODEL_DIR=None,num_classes=2):

    """
    Load a pre-trained model and tokenizer based on the specified model name and task type.

    Args:
        model_name (str): The name of the pre-trained model to load (e.g., 'bert-base-uncased').
        mlm (bool): If True, load a model for masked language modeling; otherwise, load a sequence classification model.
        MODEL_DIR (str): Directory where the model and tokenizer are stored for sequence classification tasks.
        num_classes (int): Number of classes for the sequence classification task.
    Returns:
        model: The loaded pre-trained model.
        tokenizer: The tokenizer corresponding to the loaded model.
    """
    if mlm:
        if model_name.startswith("bert"):
            tokenizer = BertTokenizer.from_pretrained(model_name,output_attentions=True)
            model = BertForMaskedLM.from_pretrained(model_name,output_attentions=True)
        elif model_name.startswith("roberta"):
            tokenizer = RobertaTokenizer.from_pretrained(model_name,output_attentions=True)
            model = RobertaForMaskedLM.from_pretrained(model_name,output_attentions=True)
    else:
        if MODEL_DIR is None:
            raise ValueError("MODEL_DIR must be specified for loading the classifier model.")
        if model_name.startswith("bert"):
            tokenizer = BertTokenizer.from_pretrained(Path(MODEL_DIR+"_tokenizer"))
            model = BertForSequenceClassification.from_pretrained(Path(MODEL_DIR+"_model"), num_labels=num_classes,output_attentions=False)
        elif model_name.startswith("roberta"):
            tokenizer = RobertaTokenizer.from_pretrained(Path(MODEL_DIR+"_tokenizer"))
            model = RobertaForSequenceClassification.from_pretrained(Path(MODEL_DIR+"_model"), num_labels=num_classes,output_attentions=False)
    return model, tokenizer

def load_mlm(tokenizer):
    """
    Load the dataset for masked language modeling (MLM) and tokenize it.
    Args:
        tokenizer: The tokenizer to use for tokenizing the dataset.
    Returns:
        train_dataset: The tokenized training dataset.
    """
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
    """
    Load the dataset for sequence classification and tokenize it.
    Args:
        tokenizer: The tokenizer to use for tokenizing the dataset.
        hidden (bool): If True, returns the hidden test dataset (emerging dogwhistles).
    Returns:
        train_dataset: The tokenized training dataset.  
    """
    dataset = load_dataset("AstroAure/dogwhistle_dataset")
    dataset = DatasetDict({
        "train": dataset["train_bhr"],
        "test": dataset["test_bhr"],
        "test_hidden": dataset["test_hidden"]
    })

    tokenized_datasets = dataset.map(lambda example : tokenize_function(example,tokenizer), batched=True)

    train_dataset = tokenized_datasets["train"].shuffle(seed=42)  
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
    if hidden:
        hidden_dataset = tokenized_datasets["test_hidden"].shuffle(seed=42)
        return hidden_dataset
    return train_dataset, eval_dataset

def load_classifier_2(tokenizer,hidden=False,dict_ingroup=None):
    """
    Load the dataset for ingroup classification and tokenize it. If `dict_ingroup` is provided, it will be used to map ingroup labels to integers.
    Args:
        tokenizer: The tokenizer to use for tokenizing the dataset.
        hidden (bool): If True, returns the hidden test dataset (emerging dogwhistles).
        dict_ingroup (dict, optional): A dictionary mapping ingroup labels to integers. If None, it will be created.
    Returns:
        train_dataset: The tokenized training dataset.
    """
    dataset = load_dataset('AstroAure/dogwhistle_dataset')
    dataset = DatasetDict({
        "train": dataset["train_bhr"],
        "test": dataset["test_bhr"],
        "test_hidden": dataset["test_hidden"]
    })

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