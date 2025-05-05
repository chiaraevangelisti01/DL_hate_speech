from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset,DatasetDict
import re 
import argparse
from pathlib import Path
import json

from src.utils import *
from src.load_data import *
from src.train_test import *

def parse_args():
    #when working with python files from console it's better to specify
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--model_path", required=True, help="Directory to save the models")
    parser.add_argument("--model_name", default = "bert-base-uncased", help="Name of the model to be used")
    args = parser.parse_args()

    return args.model_path,args.model_name


if __name__ == "__main__":

    print("***** Training the first classifier : dog-whistles vs non dog-whistles *****")
    
    MODEL_DIR,model_name = parse_args()
    
    print("Loading model and tokenizer...")
    model_classifier, tokenizer = load_model(model_name,mlm=False,MODEL_DIR=MODEL_DIR +"/fine_tuned_mlm_"+model_name,num_classes=2)
    # Load dataset
    print("Loading dataset...")
    train_dataset, eval_dataset = load_classifier(tokenizer)

    #Train the model
    print("Training model...")
    model_classifier = train_model(model_classifier, tokenizer, train_dataset, eval_dataset, output_dir="./results", epochs=3)

    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(model_classifier, tokenizer, eval_dataset,name=model_name+"_cl1",labels_list=["non dog-whistle"," dog-whistle"])

    # Save the model
    print("Saving model...")
    model_classifier.save_pretrained(Path(MODEL_DIR+"/fine_tuned_classifier_1_"+model_name+"_model"))
    tokenizer.save_pretrained(Path(MODEL_DIR+"/fine_tuned_classifier_1_"+model_name+"_tokenizer"))
    
    # Second model : distinguish between dog-whistles
    print("***** Training the second classifier : dog-whistles *****")

    #Load data 
    print("Loading dataset...")
    train_dataset, eval_dataset , dict_ingroup= load_classifier_2(tokenizer)
    #Save dict_ingroup and inv_dict_ingroup
    with open(Path(MODEL_DIR) / "ingroup.json", "w") as json_file:
        json.dump(dict_ingroup, json_file)

    num_classes = len(dict_ingroup)
    #Load model
    print("Loading model and tokenizer...")
    model_classifier_2, tokenizer = load_model(model_name,mlm=False,MODEL_DIR=MODEL_DIR +"/fine_tuned_mlm_"+model_name,num_classes=num_classes)

    #Train model
    print("Training model...")
    model_classifier_2 = train_model(model_classifier_2, tokenizer, train_dataset, eval_dataset, output_dir="./results", epochs=5)

    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(model_classifier_2, tokenizer, eval_dataset,name=model_name+"_cl2",labels_list=dict_ingroup.keys())

    #Save model
    print("Saving model...")
    model_classifier_2.save_pretrained(Path(MODEL_DIR+"/fine_tuned_classifier_2_"+model_name+"_model"))
    tokenizer.save_pretrained(Path(MODEL_DIR+"/fine_tuned_classifier_2_"+model_name+"_tokenizer"))
