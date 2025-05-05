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
    parser.add_argument("--model_path", required=True, help="Directory to save the models")
    parser.add_argument("--model_name", default = "bert-base-uncased", help="Name of the model to be used")
    args = parser.parse_args()

    return args.model_path,args.model_name

if __name__ == "__main__":


    # Parse arguments
    MODEL_DIR,model_name = parse_args()


    print("***** Evaluating first classifier : dog-whistles vs non dog-whistles *****")
    print("Load data and models...")
    classifier_1, tokenizer = load_model(model_name,mlm=False,MODEL_DIR=MODEL_DIR +"/fine_tuned_classifier_1_"+model_name,num_classes=2)
    #hidden_dataset = load_classifier(tokenizer,hidden=True)


    # Evaluate the model
    print("Evaluating model...")
    #metrics = evaluate_model(classifier_1, tokenizer, hidden_dataset)


    print("***** Evaluating second classifier : dog-whistles *****")
    # Load the tokenizer and the pre-trained BERT model
    print("Loading model and data...")
    with open(Path(MODEL_DIR) / "ingroup.json", "r") as json_file:
        dict_ingroup = json.load(json_file)
    
    inv_dict_ingroup = {v: k for k, v in dict_ingroup.items()}  # Inverse dictionary
    num_classes = len(dict_ingroup)

    # Load models and tokenizer
    classifier_2, tokenizer = load_model(model_name,mlm=False,MODEL_DIR=MODEL_DIR +"/fine_tuned_classifier_2_"+model_name,num_classes=num_classes)
    # Load dataset
    #hidden_dataset,dict_ingroup = load_classifier_2(tokenizer,hidden=True,dict_ingroup=dict_ingroup)
    print("Evaluating model...")
    #metrics = evaluate_model(classifier_2, tokenizer, hidden_dataset)

    
    # Test sentence
    test_sentence = "We need to take our country back from the thugs ruining our cities."
    test_sentence = clean_text(test_sentence)

    # Classify if there is a dog-whistle
    classifier = pipeline('text-classification', model=classifier_1, tokenizer=tokenizer)
    prediction = classifier(test_sentence)

    if prediction[0]["label"] == "LABEL_1":
        print("⚠️ Dogwhistle detected.")

        # If yes, classify which dog-whistle it is
        classifier2 = pipeline('text-classification', model=classifier_2, tokenizer=tokenizer)
        pred2 = classifier2(test_sentence)

        label = int(pred2[0]["label"].replace("LABEL_", ""))  # Assumes labels are like LABEL_0, LABEL_1, etc.
        print("🎯 Target group:", inv_dict_ingroup[label])

    else:
        print("✅ No dogwhistle detected.")

    
