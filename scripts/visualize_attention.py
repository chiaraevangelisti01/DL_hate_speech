from transformers import (
    BertTokenizer, BertForMaskedLM, BertForSequenceClassification,
    RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification
)
from bertviz import head_view, model_view
from datasets import load_dataset
from pathlib import Path
import torch
import random
from src.load_data import load_model
import argparse


# --- CONFIG ---
MODEL_PATH = "/pvc/home/DL_hate_speech/models"
MODEL_NAME = "roberta-base"
VIS_TYPE = "model"  # choose "head" or "model"
SAVE_DIR = Path("/pvc/home/DL_hate_speech/tmp_eval/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
TASKS = ["mlm", "clf1", "clf2"]

def parse_args():
    '''Parse command line arguments for the visualization script.'''
    parser = argparse.ArgumentParser(description="Attention visualization script for dog-whistle detection.")

    parser.add_argument("--model_path", type=str, default="/pvc/home/DL_hate_speech/models",
                        help="Path to the directory containing the models.")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="Name of the transformer model to use (e.g., roberta-base).")
    parser.add_argument("--vis_type", type=str, choices=["head", "model"], default="model",
                        help="Type of attention visualization (head-level or model-level).")
    parser.add_argument("--save_dir", type=Path, default=Path("/pvc/home/DL_hate_speech/tmp_eval/"),
                        help="Directory where to save the visualizations.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["mlm", "clf1", "clf2"],
                        help="List of tasks to evaluate: mlm, clf1, clf2.")

    args = parser.parse_args()

    # Ensure save_dir exists
    args.save_dir.mkdir(parents=True, exist_ok=True)

    return args

# --- HELPERS ---
def load_sentence():
    dataset = load_dataset('AstroAure/dogwhistle_dataset')
    sample = random.choice(dataset['test_bhr'])
    #return sample.get("text") or sample.get("content")
    return "We need to take our country back from the thugs ruining our cities."

def get_tokenizer():
    if "roberta" in MODEL_NAME:
        return RobertaTokenizer.from_pretrained(f"{MODEL_PATH}/fine_tuned_mlm_{MODEL_NAME}_tokenizer")
    else:
        return BertTokenizer.from_pretrained(f"{MODEL_PATH}/fine_tuned_mlm_{MODEL_NAME}_tokenizer")


def load_model_for_task(task):
    if task == "mlm":
        if "roberta" in MODEL_NAME:
            return RobertaForMaskedLM.from_pretrained(f"{MODEL_PATH}/fine_tuned_mlm_{MODEL_NAME}_model")
        else:
            return BertForMaskedLM.from_pretrained(f"{MODEL_PATH}/fine_tuned_mlm_{MODEL_NAME}_model")
    else:
        num_classes = 2 if task == "clf1" else 17
        model, _ = load_model(
            model_name=MODEL_NAME,
            mlm=False,
            MODEL_DIR=f"{MODEL_PATH}/fine_tuned_classifier_{task[-1]}_{MODEL_NAME}",
            num_classes=num_classes
        )
        return model


def visualize(model, tokenizer, sentence, task, vis_type):
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', return_attention_mask=True)
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attention = outputs.attentions

    if vis_type == "head":
        html = head_view(attention, tokens, html_action='return')
    elif vis_type == "model":
        html = model_view(attention=attention, tokens=tokens, html_action='return')
    else:
        print(f"[!] Unsupported visualization type: {vis_type}. Skipping {task}.")
        return

    file = SAVE_DIR / f"{vis_type}_{task}.html"
    with open(file, "w", encoding="utf-8") as f:
        f.write(html.data)
    print(f"[✓] Saved {vis_type} view for {task} → {file.name}")


# --- MAIN EXECUTION ---
sentence = load_sentence()
tokenizer = get_tokenizer()

for task in TASKS:
    model = load_model_for_task(task)
    visualize(model, tokenizer, sentence, task, VIS_TYPE)

print("\n✅ All visualizations complete.")


from transformers import (
    BertTokenizer, BertForMaskedLM,
    RobertaTokenizer, RobertaForMaskedLM
)
from bertviz import head_view, model_view
from datasets import load_dataset
from pathlib import Path
import torch
import random
import argparse

from src.load_data import load_model


def parse_args():
    '''Parse command line arguments for the visualization script.'''
    parser = argparse.ArgumentParser(description="Attention visualization script for dog-whistle detection.")

    parser.add_argument("--model_path", type=str, default="/pvc/home/DL_hate_speech/models",
                        help="Path to the directory containing the models.")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="Name of the transformer model to use (e.g., roberta-base).")
    parser.add_argument("--vis_type", type=str, choices=["head", "model"], default="model",
                        help="Type of attention visualization (head-level or model-level).")
    parser.add_argument("--save_dir", type=Path, default=Path("/pvc/home/DL_hate_speech/tmp_eval/"),
                        help="Directory where to save the visualizations.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["mlm", "clf1", "clf2"],
                        help="List of tasks to evaluate: mlm, clf1, clf2.")

    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    return args


def load_sentence(input_sentence=None):
    '''Load a sentence from the dog-whistle dataset for visualization'''
    if input_sentence is None:
        # If no input sentence is provided, load a random sample from the dataset
        dataset = load_dataset('AstroAure/dogwhistle_dataset')
        sample = random.choice(dataset['test_bhr'])
    else:
        sample = input_sentence
    return sample


def get_tokenizer(model_name, model_path):
    """Get the appropriate tokenizer based on the model name."""
    if "roberta" in model_name:
        return RobertaTokenizer.from_pretrained(f"{model_path}/fine_tuned_mlm_{model_name}_tokenizer")
    else:
        return BertTokenizer.from_pretrained(f"{model_path}/fine_tuned_mlm_{model_name}_tokenizer")


def load_model_for_task(task, model_name, model_path):
    """Load the appropriate model based on the task."""
    if task == "mlm":
        if "roberta" in model_name:
            return RobertaForMaskedLM.from_pretrained(f"{model_path}/fine_tuned_mlm_{model_name}_model")
        else:
            return BertForMaskedLM.from_pretrained(f"{model_path}/fine_tuned_mlm_{model_name}_model")
    else:
        num_classes = 2 if task == "clf1" else 17
        model, _ = load_model(
            model_name=model_name,
            mlm=False,
            MODEL_DIR=f"{model_path}/fine_tuned_classifier_{task[-1]}_{model_name}",
            num_classes=num_classes
        )
        return model


def visualize(model, tokenizer, sentence, task, vis_type, save_dir):
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', return_attention_mask=True)
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        # Forward pass to get attention weights
        outputs = model(**inputs, output_attentions=True)

    # Extract attention weights
    attention = outputs.attentions

    # Generate HTML visualization based on the type
    if vis_type == "head":
        html = head_view(attention, tokens, html_action='return')
    elif vis_type == "model":
        html = model_view(attention=attention, tokens=tokens, html_action='return')
    else:
        print(f"[!] Unsupported visualization type: {vis_type}. Skipping {task}.")
        return

    file = save_dir / f"{vis_type}_{task}.html"
    with open(file, "w", encoding="utf-8") as f:
        f.write(html.data)
    print(f"[✓] Saved {vis_type} view for {task} → {file.name}")


def main():
    args = parse_args()

    sentence = load_sentence()
    tokenizer = get_tokenizer(args.model_name, args.model_path)

    for task in args.tasks:
        model = load_model_for_task(task, args.model_name, args.model_path)
        visualize(model, tokenizer, sentence, task, args.vis_type, args.save_dir)

    print("\n [✓] All visualizations complete.")


if __name__ == "__main__":
    main()
