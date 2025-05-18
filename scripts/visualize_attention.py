

from transformers import (
    BertTokenizer, BertForSequenceClassification, BertForMaskedLM,
    RobertaTokenizer, RobertaForSequenceClassification, RobertaForMaskedLM
)
from bertviz import head_view
from IPython.display import display
import random
import torch
from pathlib import Path
from datasets import load_dataset
from src.load_data import load_model
import os

model_path = "/pvc/home/DL_hate_speech/models"
model_name = "bert-base-uncased"
task = "clf2"
num_samples = 1


def load_test_sentence(tokenizer, task):
    dataset = load_dataset('AstroAure/dogwhistle_dataset')
    split = "test_bhr" 
    sample = random.choice(dataset[split])
    return sample["text"] if "text" in sample else sample["content"]

def visualize_attention(model, tokenizer, text, is_mlm):
    inputs = tokenizer.encode_plus(text, return_tensors='pt', return_attention_mask=True)
    input_ids = inputs['input_ids']
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attention = outputs.attentions

    html = head_view(attention, tokens, html_action='return')
   
    with open("/pvc/home/DL_hate_speech/tmp_eval/attention_viz.html", "w", encoding='utf-8') as f:
        f.write(html.data)
    print("\nSentence:\n", text)


if task == "mlm":
    if "roberta" in model_name:
        model = RobertaForMaskedLM.from_pretrained(Path(model_path) / f"fine_tuned_mlm_{model_name}_model", output_attentions=True)
        tokenizer = RobertaTokenizer.from_pretrained(Path(model_path) / f"fine_tuned_mlm_{model_name}_tokenizer")
    else:
        model = BertForMaskedLM.from_pretrained(Path(model_path) / f"fine_tuned_mlm_{model_name}_model", output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(Path(model_path) / f"fine_tuned_mlm_{model_name}_tokenizer")
    is_mlm = True

elif task == "clf1":
    model, tokenizer = load_model(
        model_name=model_name,
        mlm=False,
        MODEL_DIR=str(Path(model_path) / f"fine_tuned_classifier_1_{model_name}"),
        num_classes=2
    )
    model.eval()
    is_mlm = False

elif task == "clf2":
    model, tokenizer = load_model(
        model_name=model_name,
        mlm=False,
        MODEL_DIR=str(Path(model_path) / f"fine_tuned_classifier_2_{model_name}"),
        num_classes=17  # update if different
    )
    model.eval()
    is_mlm = False


for i in range(num_samples):
    text = load_test_sentence(tokenizer, task)
    visualize_attention(model, tokenizer, text, is_mlm)
