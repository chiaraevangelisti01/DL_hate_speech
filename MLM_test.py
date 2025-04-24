from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import re 


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # Assurez-vous que num_labels correspond à votre tâche
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Charger votre dataset
# Remplacez par le chargement de votre propre dataset
dataset = load_dataset('SALT-NLP/silent_signals_detection')

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
    examples['example'] = [clean_text(text) for text in examples['example']]
    # Tokeniser le texte nettoyé
    return tokenizer(examples['example'], padding="max_length", truncation=True)

def transform_label(example):
    example['label'] = 1 if example['label'] == "coded" else 0
    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.map(transform_label)



# Préparer les données pour l'entraînement
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(50))  # Exemple avec 1000 échantillons
eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))  # Exemple avec 500 échantillons

# Configurer les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Lancer l'entraînement
trainer.train()
