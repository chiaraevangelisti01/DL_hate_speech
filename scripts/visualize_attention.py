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


# --- CONFIG ---
MODEL_PATH = "/pvc/home/DL_hate_speech/models"
MODEL_NAME = "bert-base-uncased"  # or "roberta-base"
VIS_TYPE = "head"  # choose "head" or "model"
SAVE_DIR = Path("/pvc/home/DL_hate_speech/tmp_eval/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
TASKS = ["mlm", "clf1", "clf2"]


# --- HELPERS ---
def load_sentence():
    dataset = load_dataset('AstroAure/dogwhistle_dataset')
    sample = random.choice(dataset['test_bhr'])
    return sample.get("text") or sample.get("content")


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
