from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset,DatasetDict
import re 
import argparse
from pathlib import Path
import json


from src.utils import *
from src.load_data import load_classifier, load_classifier_2
from src.train_test import *

def parse_args():
    #when working with python files from console it's better to specify
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--model_path", required=True, help="Directory to save the models")
    args = parser.parse_args()

    return args.model_path


if __name__ == "__main__":

    print("***** Training the first classifier : dog-whistles vs non dog-whistles *****")
    
    MODEL_DIR = parse_args()
    # Load the tokenizer and the pre-trained BERT model
    print("Loading model and tokenizer...")
    # model_classifier = BertForSequenceClassification.from_pretrained(Path(MODEL_DIR) / "fine_tuned_mlm_model", num_labels=2, local_files_only=True)  
    # tokenizer = BertTokenizer.from_pretrained(Path(MODEL_DIR) / "fine_tuned_mlm_tokenizer", local_files_only=True)

    model_classifier = RobertaForSequenceClassification.from_pretrained(Path(MODEL_DIR) / "fine_tuned_mlm_model_Roberta", num_labels=2)
    tokenizer = RobertaTokenizer.from_pretrained(Path(MODEL_DIR) / "fine_tuned_mlm_tokenizer_Roberta")

    # Load dataset
    print("Loading dataset...")
    train_dataset, eval_dataset = load_classifier(tokenizer)

    #Train the model
    print("Training model...")
    model_classifier = train_model(model_classifier, tokenizer, train_dataset, eval_dataset, output_dir="./results", epochs=3)

    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(model_classifier, tokenizer, eval_dataset)

    # Save the model
    print("Saving model...")
    model_classifier.save_pretrained(Path(MODEL_DIR) /"fine_tuned_classifier_1_model_Roberta")
    tokenizer.save_pretrained(Path(MODEL_DIR) /"fine_tuned_classifier_1_tokenizer_Roberta")


    
    
    # Second model : distinguish between dog-whistles
    print("***** Training the second classifier : dog-whistles *****")

    #Load data 
    print("Loading dataset...")
    train_dataset, eval_dataset , dict_ingroup, inv_dict_ingroup = load_classifier_2(tokenizer)
    #Save dict_ingroup and inv_dict_ingroup
    with open(Path(MODEL_DIR) / "ingroup.json", "w") as json_file:
        json.dump(dict_ingroup, json_file)
    with open(Path(MODEL_DIR) / "inv_ingroup.json", "w") as json_file:
        json.dump(inv_dict_ingroup, json_file)
    
    num_classes = len(dict_ingroup)
    #Load model
    print("Loading model and tokenizer...")
    model_classifier_2 = BertForSequenceClassification.from_pretrained(
        Path(MODEL_DIR) / "fine_tuned_mlm_model_Roberta", 
        num_labels=num_classes,
        local_files_only=True  
    )
    #Train model
    print("Training model...")
    model_classifier_2 = train_model(model_classifier_2, tokenizer, train_dataset, eval_dataset, output_dir="./results", epochs=5)

    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(model_classifier_2, tokenizer, eval_dataset)

    #Save model
    print("Saving model...")
    model_classifier_2.save_pretrained(Path(MODEL_DIR) /"fine_tuned_classifier_2_model_Roberta")
    tokenizer.save_pretrained(Path(MODEL_DIR) /"fine_tuned_classifier_2_tokenizer_Roberta")
