from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset,DatasetDict
import re 
import argparse
from pathlib import Path
import json
import torch
import gc

from src.utils import *
from src.load_data import *
from src.train_test import *

def parse_args():
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--model_path", default="/pvc/home/DL_hate_speech/models", help="Directory to save the models")
    parser.add_argument("--model_name", default = "bert-base-uncased", help="Name of the model to be used")
    parser.add_argument("--results_path", default = "/pvc/home/DL_hate_speech/results", help="Path to save the results")
    args = parser.parse_args()

    return args.model_path,args.model_name,args.results_path


if __name__ == "__main__":

    print("***** Training the first classifier : dog-whistles vs non dog-whistles *****")
    set_seed(42)  # For reproducibility
    MODEL_DIR,model_name,results_path = parse_args()

    torch.cuda.empty_cache()
    gc.collect()
    
    print("Loading model and tokenizer...")
    model_classifier, tokenizer = load_model(model_name,mlm=False,MODEL_DIR=MODEL_DIR +"/fine_tuned_mlm_"+model_name,num_classes=2)

    print("Loading dataset...")
    train_dataset, eval_dataset = load_classifier(tokenizer)

    print("Training model...")
    model_classifier = train_model(model_classifier, tokenizer, train_dataset, eval_dataset, output_dir="./results", epochs=3)

    print("Evaluating model...")
    metrics = evaluate_model(model_classifier, tokenizer, eval_dataset,results_path,name=model_name+"_cl1_nonMLM",labels_list=["non dog-whistle"," dog-whistle"])

    print("Saving model...")
    model_classifier.save_pretrained(Path(MODEL_DIR+"/fine_tuned_classifier_1_"+model_name+"_model"))
    tokenizer.save_pretrained(Path(MODEL_DIR+"/fine_tuned_classifier_1_"+model_name+"_tokenizer"))
    

    print("***** Training the second classifier : in-group detection *****")

    # Free memory
    del model_classifier
    torch.cuda.empty_cache()
    gc.collect()

    print("Loading dataset...")
    with open(Path(MODEL_DIR) / "ingroup.json", "r") as json_file:
        dict_ingroup = json.load(json_file)
    train_dataset, eval_dataset , dict_ingroup= load_classifier_2(tokenizer,dict_ingroup=dict_ingroup)
    num_classes = len(dict_ingroup)


    print("Loading model and tokenizer...")
    model_classifier_2, tokenizer = load_model(model_name,mlm=False,MODEL_DIR=MODEL_DIR +"/fine_tuned_mlm_"+model_name,num_classes=num_classes)
    

    print("Training model...")
    model_classifier_2 = train_model(model_classifier_2, tokenizer, train_dataset, eval_dataset, output_dir="./results", epochs=5)

    print("Evaluating model...")
    metrics = evaluate_model(model_classifier_2, tokenizer, eval_dataset,results_path,name=model_name+"_cl2_nonMLM",labels_list=dict_ingroup.keys())

    print("Saving model...")
    model_classifier_2.save_pretrained(Path(MODEL_DIR+"/fine_tuned_classifier_2_"+model_name+"_model"))
    tokenizer.save_pretrained(Path(MODEL_DIR+"/fine_tuned_classifier_2_"+model_name+"_tokenizer"))
    