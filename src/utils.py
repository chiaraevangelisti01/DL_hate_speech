import re 
import numpy as np
import random
import torch

"""
This module provides utility functions used across various parts of the project.
"""

def set_seed(seed):
    """
    Set the random seed for reproducibility across different libraries.
    Args:
        seed (int): The seed value to set for random number generation.
    """
    if seed is None:
        seed = 42  # Default seed value if none is provided
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False



def clean_text(text):
    """
    Clean the input text by converting it to lowercase, removing special characters and digits,
    and removing extra spaces.
    Args:
        text (str): The input text to clean.
    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    
def tokenize_function(examples,tokenizer):
    """
    Tokenize the input examples by cleaning the text and applying the tokenizer.
    Args:
        examples (dict): The input examples containing the text to tokenize.
    Returns:
        dict: The tokenized examples with input IDs and attention masks.
    """
    examples['content'] = [clean_text(text) for text in examples['content']]
    return tokenizer(examples['content'], padding="max_length", truncation=True)

def mask_tokens(examples,tokenizer, mlm_probability=0.15):

    """
    Mask tokens in the input examples for masked language modeling. Mask dog whistles and random tokens
    based on the specified probability.
    Args:
        examples (dict): The input examples containing the text to mask.
        mlm_probability (float): The probability of masking a token.
    Returns:
        dict: The tokenized examples with masked input IDs.
    """
    texts = [clean_text(text) for text in examples["content"]]
    dw_texts = examples["dog_whistle"]

    tokenized = tokenizer(texts, padding="max_length", truncation=True, return_special_tokens_mask=True)

    input_ids = np.array(tokenized["input_ids"])
    special_tokens_mask = np.array(tokenized["special_tokens_mask"])

    # Random masking (excluding special tokens)
    mask = (np.random.rand(*input_ids.shape) < mlm_probability) & (special_tokens_mask == 0)

    # Masking dog whistles
    for i, dw in enumerate(dw_texts):
        dw_ids = tokenizer(dw, add_special_tokens=False)["input_ids"]
        for start_idx in range(input_ids.shape[1] - len(dw_ids) + 1):
            if all(input_ids[i, start_idx + j] == dw_ids[j] for j in range(len(dw_ids))):
                mask[i, start_idx:start_idx+len(dw_ids)] = True

    # Apply masking
    input_ids_masked = input_ids.copy()
    input_ids_masked[mask] = tokenizer.mask_token_id

    tokenized["input_ids"] = input_ids_masked.tolist()
    return tokenized

def update_label(example,dict_ingroup):
    """
    Update the label of the example based on the ingroup value.
    Args:
        example (dict): The input example containing the ingroup value.
    Returns:
        dict: The updated example with the new label.
    """
    example["label"] = dict_ingroup[example["ingroup"]]
    return example


