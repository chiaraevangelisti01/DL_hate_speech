from datasets import load_dataset,DatasetDict
from src.utils import *
import json


def load_mlm(tokenizer):
    dataset = load_dataset('AstroAure/dogwhistle_dataset')
    dataset = DatasetDict({
        "train": dataset["train_balanced"],
        "test": dataset["test_balanced"]
    })

    tokenized_datasets = dataset.map(lambda examples: mask_tokens(examples,tokenizer), batched=True)
                                 
    if 'label' in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.remove_columns('label')

    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
    return train_dataset, eval_dataset


def load_classifier(tokenizer):
    # Load dataset
    dataset = load_dataset('AstroAure/dogwhistle_dataset')
    dataset = DatasetDict({
        "train": dataset["train_balanced"],
        "test": dataset["test_balanced"]
    })

    # Tokenize the dataset
    tokenized_datasets = dataset.map(lambda example : tokenize_function(example,tokenizer), batched=True)

    # Prepare the data for training
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)  # Example with 1000 samples
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)  # Example with 500 samples

    return train_dataset, eval_dataset

def load_classifier_2(tokenizer):
    # Load dataset
    dataset = load_dataset('AstroAure/dogwhistle_dataset')
    dataset = DatasetDict({
        "train": dataset["train_balanced"],
        "test": dataset["test_balanced"]
    })

    # Tokenize the dataset
    tokenized_datasets = dataset.map(lambda example : tokenize_function(example,tokenizer), batched=True)

    filtered_dataset = tokenized_datasets.filter(lambda example: example["label"] == 1)
    dict_ingroup = {k:i for i,k in enumerate(set(filtered_dataset["train"]["ingroup"]))}
    inv_dict_ingroup ={v:k for (k,v) in dict_ingroup.items()}

    # with open("/home/mhueber/DL_hate_speech/models/ingroup.json", "w") as json_file:
    #     json.dump(inv_dict_ingroup, json_file)

    filtered_dataset = filtered_dataset.map(lambda example : update_label(example,dict_ingroup))
    train_dataset = filtered_dataset["train"].shuffle(seed=42)
    eval_dataset = filtered_dataset["train"].shuffle(seed=42)

    return train_dataset, eval_dataset , dict_ingroup, inv_dict_ingroup