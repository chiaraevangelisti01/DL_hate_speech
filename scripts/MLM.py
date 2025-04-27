from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import RobertaTokenizer, RobertaForMaskedLM
import re
import torch
import random 
import argparse
from pathlib import Path

from src.utils import *
from src.train_test import train_model
from src.load_data import load_mlm

def parse_args():
    #when working with python files from console it's better to specify
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--model_path", required=True, help="Directory to save the model")
    parser.add_argument("--model_name", default = "bert-base-uncased", help="Name of the model to be used")
    args = parser.parse_args()

    return args.model_path,args.model_name

if __name__ == "__main__":

    print("GPU available:", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    MODEL_DIR,model_name = parse_args()
    print("Loading model and tokenizer...")
    # # Load Bert tokenizer and model 
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base")
    # tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    # model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

    print("Loading dataset...")
    train_dataset, eval_dataset = load_mlm(tokenizer)

    print("Training model...")
    model = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="./results",
        mlm=True,
    )
    #Save model 
    print("Saving model...")
    model.save_pretrained(Path(MODEL_DIR) / "fine_tuned_mlm_model_Roberta")
    tokenizer.save_pretrained(Path(MODEL_DIR) / "fine_tuned_mlm_tokenizer_Roberta")
    print("Model saved successfully.")