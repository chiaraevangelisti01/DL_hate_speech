import argparse
from transformers import pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from src.utils import *
from src.train_test import *
from src.load_data import *

def parse_args():
    '''Parse command line arguments for the script.'''
    parser = argparse.ArgumentParser(description="Dog-whistle detection script.")
    parser.add_argument("--model_path", default="/pvc/home/DL_hate_speech/models", help="Directory to save the models")
    parser.add_argument("--model_name", default="roberta-base", help="Name of the model to be used")
    parser.add_argument("--classifier", type=int, choices=[1, 2], default=1, help="Which classifier to evaluate (1 or 2)")
    args = parser.parse_args()
    return args.model_path, args.model_name, args.classifier

def print_misclassified_sentences(model, tokenizer, dataset, label_map=None):
    '''Find and print missclassified sentence in the test dataset'''
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"]) #Specify the format for the dataset


    dataloader = DataLoader(dataset, batch_size=16)

    print(" Checking for misclassified sentences...")
    for batch in tqdm(dataloader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            # Inference
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

        # Reconstruct sentences from input_ids
        decoded_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

        # Check for misclassifications
        for text, pred, true in zip(decoded_texts, preds, labels):
            if pred.item() != true.item():
                pred_label = label_map[pred.item()] if label_map else pred.item()
                true_label = label_map[true.item()] if label_map else true.item()
                print(" Sentence:", text.strip())
                print(f"  Predicted: {pred_label} | True: {true_label}\n")


if __name__ == "__main__":
    MODEL_DIR, model_name, classifier_choice = parse_args()

    if classifier_choice == 1:
        print("***** Evaluating Classifier 1 (dog-whistle vs non-dog-whistle) *****")
        model, tokenizer = load_model(model_name, mlm=False, MODEL_DIR=MODEL_DIR + "/fine_tuned_classifier_1_" + model_name, num_classes=2)
        dataset = load_classifier(tokenizer, hidden=True)
        print_misclassified_sentences(model, tokenizer, dataset)

    else:
        print("***** Evaluating Classifier 2 (which dog-whistle) *****")
        with open(Path(MODEL_DIR) / "ingroup.json", "r") as f:
            dict_ingroup = json.load(f)
        inv_dict_ingroup = {v: k for k, v in dict_ingroup.items()}
        num_classes = len(dict_ingroup)

        model, tokenizer = load_model(model_name, mlm=False, MODEL_DIR=MODEL_DIR + "/fine_tuned_classifier_2_" + model_name, num_classes=num_classes)
        dataset, _ = load_classifier_2(tokenizer, hidden=True, dict_ingroup=dict_ingroup)
        print_misclassified_sentences(model, tokenizer, dataset, label_map=inv_dict_ingroup)
