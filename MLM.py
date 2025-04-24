from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import re
import torch
import random 



# Tokeniser le dataset
def clean_text(text):
    # Convertir en minuscules
    text = text.lower()
    # Supprimer les caractères spéciaux et les chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text) # ? 
    # Supprimer les espaces superflus
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    
def tokenize_function(examples):
    # Appliquer le nettoyage du texte
    examples['content'] = [clean_text(text) for text in examples['content']]
    # Tokeniser le texte nettoyé
    return tokenizer(examples['content'], padding="max_length", truncation=True)

def transform_label(example):
    example['label'] = 1 if example['label'] == "coded" else 0
    return example


def mask_tokens(examples, mlm_probability=0.15):
    examples['content'] = [clean_text(text) for text in examples['content']]

    tokenized_inputs = tokenizer(examples['content'], padding="max_length", truncation=True, return_tensors="pt")

    for i, tokens in enumerate(tokenized_inputs['input_ids']):
        # Masquer les "dog whistles"
        dw = examples['dog_whistle'][i]
        dw_ids = tokenizer(dw, add_special_tokens=False)['input_ids']
        for start_idx in range(len(tokens) - len(dw_ids) + 1):
            if tokens[start_idx:start_idx+len(dw_ids)].equal(torch.tensor(dw_ids)):
                for idx in range(start_idx, start_idx+len(dw_ids)):
                    tokens[idx] = tokenizer.mask_token_id

        # Masquer des tokens aléatoires
        for idx in range(len(tokens)):
            if random.random() < mlm_probability and tokens[idx] not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                tokens[idx] = tokenizer.mask_token_id

    return tokenized_inputs


dog_whistle_list = list(set(dataset["train"]["dog_whistle"]))


#tokenized_datasets = dataset.map(tokenize_function, batched=True)




if __name__ == "__main__":

    # Charger le tokenizer et le modèle BERT pré-entraîné pour MLM
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    # Charger votre dataset
    # Remplacez par le chargement de votre propre dataset
    dataset = load_dataset('AstroAure/dogwhistle_dataset')

    tokenized_datasets = dataset.map(lambda examples: mask_tokens(examples), batched=True)
                                 
    if 'label' in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.remove_columns('label')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Préparer les données pour l'entraînement
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(50))  # Exemple avec 1000 échantillons
    eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))  # Exemple avec 500 échantillons

    # Configurer les arguments d'entraînement
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    # Initialiser le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator = data_collator,
    )

    # Lancer l'entraînement
    trainer.train()

    model.save_pretrained("./fine_tuned_mlm_model")
    tokenizer.save_pretrained("./fine_tuned_mlm_model")
        
