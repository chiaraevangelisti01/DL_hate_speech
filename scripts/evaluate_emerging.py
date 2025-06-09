import argparse
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from pathlib import Path
import json

from src.utils import *
from src.train_test import *
from src.load_data import *



def parse_args():
    parser = argparse.ArgumentParser(description="Dog-whistle detection script.")
    parser.add_argument("--model_path", default="/pvc/home/DL_hate_speech/models", help="Directory to save the models")
    parser.add_argument("--model_name", default = "bert-base-uncased", help="Name of the model to be used")
    parser.add_argument("--results_path", default ="/pvc/home/DL_hate_speech/results", help="Path to save the results")
    args = parser.parse_args()

    return args.model_path,args.model_name,args.results_path

if __name__ == "__main__":


    MODEL_DIR,model_name,results_path = parse_args()


    print("***** Evaluating first classifier : dog-whistles vs non dog-whistles *****")
    print("Load data and models...")
    classifier_1, tokenizer = load_model(model_name,mlm=False,MODEL_DIR=MODEL_DIR +"/fine_tuned_classifier_1_"+model_name,num_classes=2)
    hidden_dataset = load_classifier(tokenizer,hidden=True)


    print("Evaluating model...")
    metrics = evaluate_model(classifier_1, tokenizer, hidden_dataset,results_path,name=model_name+"_cl1_hidden",labels_list=["non dog-whistle"," dog-whistle"])


    print("***** Evaluating second classifier : in-group detection *****")
    print("Loading model and data...")
    with open(Path(MODEL_DIR) / "ingroup.json", "r") as json_file:
        dict_ingroup = json.load(json_file)
    
    inv_dict_ingroup = {v: k for k, v in dict_ingroup.items()}  
    num_classes = len(dict_ingroup)

    # Load models and tokenizer
    classifier_2, tokenizer = load_model(model_name,mlm=False,MODEL_DIR=MODEL_DIR +"/fine_tuned_classifier_2_"+model_name,num_classes=num_classes)
    # Load dataset
    hidden_dataset,dict_ingroup = load_classifier_2(tokenizer,hidden=True,dict_ingroup=dict_ingroup)
    
    print("Evaluating model...")

    metrics = evaluate_model(classifier_2, tokenizer, hidden_dataset,results_path,name=model_name+"_cl2_hidden",labels_list=list(dict_ingroup.keys()))


