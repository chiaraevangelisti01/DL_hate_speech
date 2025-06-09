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
    parser.add_argument("--test_sentence", default = "We need to take our country back from the thugs ruining our cities.", help="Sentence to test for dog-whistle detection")
    args = parser.parse_args()

    return args.model_path,args.model_name, args.test_sentence

if __name__ == "__main__":


    MODEL_DIR,model_name,test_sentence = parse_args()


    print("Load 1st classifier ...")
    classifier_1, tokenizer = load_model(model_name,mlm=False,MODEL_DIR=MODEL_DIR +"/fine_tuned_classifier_1_"+model_name,num_classes=2)

    with open(Path(MODEL_DIR) / "ingroup.json", "r") as json_file:
        dict_ingroup = json.load(json_file)

    inv_dict_ingroup = {v: k for k, v in dict_ingroup.items()}  
    num_classes = len(dict_ingroup)

    print("Load 2nd classifier ...")
    classifier_2, tokenizer = load_model(model_name,mlm=False,MODEL_DIR=MODEL_DIR +"/fine_tuned_classifier_2_"+model_name,num_classes=num_classes)
    
    print("Sentence to test:", test_sentence)

    test_sentence = clean_text(test_sentence)

    # Classify if there is a dog-whistle
    classifier = pipeline('text-classification', model=classifier_1, tokenizer=tokenizer)
    prediction = classifier(test_sentence)

    if prediction[0]["label"] == "LABEL_1":
        print(" Dogwhistle detected.")

        # If yes, classify which in-group it is targetting
        classifier2 = pipeline('text-classification', model=classifier_2, tokenizer=tokenizer)
        pred2 = classifier2(test_sentence)

        label = int(pred2[0]["label"].replace("LABEL_", ""))  
        print(" Target group:", inv_dict_ingroup[label])

    else:
        print(" No dogwhistle detected.")
