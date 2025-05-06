# # %%
# import argparse
# from transformers import (
#     BertTokenizer, BertForSequenceClassification, BertForMaskedLM,
#     RobertaTokenizer, RobertaForSequenceClassification, RobertaForMaskedLM
# )
# from bertviz import head_view
# from IPython.display import display
# import random
# import torch
# from pathlib import Path
# from datasets import load_dataset
# from src.load_data import load_model

# ''' 
# to run inside the interactive shell
# cd /pvc/home/DL_hate_speech
# PYTHONPATH=. python3 scripts/visualize_attention.py \
#   --model_path /pvc/home/DL_hate_speech/models \
#   --model_name roberta-base \ #change to another model
#   --task clf2  #change to another task
# '''
# # %%
# def parse_args():
#     parser = argparse.ArgumentParser(description="Visualize attention for fine-tuned models.")
#     parser.add_argument("--model_path", required=True, help="Base directory containing the saved model")
#     parser.add_argument("--model_name", default="bert-base-uncased", help="Name of the HuggingFace model")
#     parser.add_argument("--task", choices=["mlm", "clf1", "clf2"], required=True, help="Type of model/task")
#     parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to visualize") 
#     return parser.parse_args()

# # %%
# def load_test_sentence(tokenizer, task):
#     dataset = load_dataset('AstroAure/dogwhistle_dataset')
#     split = "test_bhr" 
#     sample = random.choice(dataset[split])
#     return sample["text"] if "text" in sample else sample["content"]

# def visualize_attention(model, tokenizer, text, is_mlm):
#     inputs = tokenizer.encode_plus(text, return_tensors='pt', return_attention_mask=True)
#     input_ids = inputs['input_ids']
#     with torch.no_grad():
#         outputs = model(**inputs, output_attentions=True)
#     tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
#     attention = outputs.attentions
#     display(head_view(attention, tokens))
#     print("\nSentence:\n", text)

# # %%
# def main():
#     args = parse_args()

#     if args.task == "mlm":
#         if "roberta" in args.model_name:
#             model = RobertaForMaskedLM.from_pretrained(Path(args.model_path) / f"fine_tuned_mlm_{args.model_name}_model", output_attentions=True)
#             tokenizer = RobertaTokenizer.from_pretrained(Path(args.model_path) / f"fine_tuned_mlm_{args.model_name}_tokenizer")
#         else:
#             model = BertForMaskedLM.from_pretrained(Path(args.model_path) / f"fine_tuned_mlm_{args.model_name}_model", output_attentions=True)
#             tokenizer = BertTokenizer.from_pretrained(Path(args.model_path) / f"fine_tuned_mlm_{args.model_name}_tokenizer")
#         is_mlm = True

#     elif args.task == "clf1":
#         model, tokenizer = load_model(
#             model_name=args.model_name,
#             mlm=False,
#             MODEL_DIR=str(Path(args.model_path) / f"fine_tuned_classifier_1_{args.model_name}"),
#             num_classes=2
#         )
#         model.eval()
#         is_mlm = False

#     elif args.task == "clf2":
#         model, tokenizer = load_model(
#             model_name=args.model_name,
#             mlm=False,
#             MODEL_DIR=str(Path(args.model_path) / f"fine_tuned_classifier_2_{args.model_name}"),
#             num_classes=17  # update if different
#         )
#         model.eval()
#         is_mlm = False

#     for i in range(args.num_samples):
#         text = load_test_sentence(tokenizer, args.task)
#         visualize_attention(model, tokenizer, text, is_mlm)
# # %%
# if __name__ == "__main__":
#     main()

# %% [1] Imports
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

# %% [2] Configuration (replace argparse)
model_path = "/pvc/home/DL_hate_speech/models"
model_name = "bert-base-uncased"
task = "clf2"
num_samples = 1

# %% [3] Utility Functions
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
    display(head_view(attention, tokens))
    print("\nSentence:\n", text)

# %% [4] Load model and tokenizer
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

# %% [5] Visualize
for i in range(num_samples):
    text = load_test_sentence(tokenizer, task)
    visualize_attention(model, tokenizer, text, is_mlm)
