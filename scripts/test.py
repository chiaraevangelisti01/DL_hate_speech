import argparse
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from pathlib import Path

from src.utils import *
from src.train_test import *
from src.load_data import *

def parse_args():
    parser = argparse.ArgumentParser(description="Dog-whistle detection script.")
    parser.add_argument("--model_path", required=True, help="Directory to save the models")

    #parser.add_argument("--inv_dict_path", required=True, help="Path to the inverse dictionary (for dog-whistle types)")
    args = parser.parse_args()
    return args.model_path

if __name__ == "__main__":


    # Parse arguments
    MODEL_DIR = parse_args()


    print("***** Evaluating first classifier : dog-whistles vs non dog-whistles *****")
    print("Load data and models...")
    tokenizer = BertTokenizer.from_pretrained(Path(MODEL_DIR) / "fine_tuned_mlm_tokenizer", local_files_only=True)
    train_dataset, eval_dataset = load_classifier(tokenizer)
    classifier_1 = BertForSequenceClassification.from_pretrained(Path(MODEL_DIR) /"fine_tuned_classifier_1_model",num_labels =2, local_files_only=True)

    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(classifier_1, tokenizer, eval_dataset)


    print("***** Evaluating second classifier : dog-whistles *****")
    # Load the tokenizer and the pre-trained BERT model
    print("Loading model and data...")
    
    train_dataset, eval_dataset , dict_ingroup, inv_dict_ingroup = load_classifier_2(tokenizer)
    num_classes = len(dict_ingroup)

    # Load models and tokenizer
    classifier_2 = BertForSequenceClassification.from_pretrained(Path(MODEL_DIR) /"fine_tuned_classifier_2_model",num_labels =num_classes, local_files_only=True)

    train_dataset, eval_dataset , dict_ingroup, inv_dict_ingroup = load_classifier_2(tokenizer)
    print("Evaluating model...")
    metrics = evaluate_model(classifier_2, tokenizer, eval_dataset)

    # # Load the inverse dictionary for dog-whistle types
    # inv_dict_ingroup = load_inv_dict(inv_dict_path)

    # # Test sentence
    # test_sentence = "Lesbian GC stuff will get wiped as well because queer content gets flagged as NSFW. Idiots. They’d cut off their own nose to spite the face."
    # test_sentence = clean_text(test_sentence)

    # # Classify if there is a dog-whistle
    # classifier = pipeline('text-classification', model=classifier_1, tokenizer=tokenizer)
    # prediction = classifier(test_sentence)

    # if prediction[0]["label"] == "LABEL_1":
    #     print("⚠️ Dogwhistle detected.")

    #     # If yes, classify which dog-whistle it is
    #     classifier2 = pipeline('text-classification', model=classifier_2, tokenizer=tokenizer)
    #     pred2 = classifier2(test_sentence)

    #     label = int(pred2[0]["label"].replace("LABEL_", ""))  # Assumes labels are like LABEL_0, LABEL_1, etc.
    #     print("🎯 Target group:", inv_dict_ingroup[label])

    # else:
    #     print("✅ No dogwhistle detected.")

    
