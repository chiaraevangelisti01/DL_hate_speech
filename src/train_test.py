from transformers import Trainer, TrainingArguments,DataCollatorForLanguageModeling
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import numpy as np

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir="./results", epochs=3,mlm=False):
    """
    Train a model using the provided datasets and configurations.
    Args:
        model: The model to be trained.
        tokenizer: The tokenizer to be used.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        output_dir (str): The directory to save the model and results.
        epochs (int): The number of training epochs.
        mlm (bool): Whether it uses masked language modeling.
    Returns:
        model: The trained model.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    if mlm:
        trainer.data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    trainer.train()
    return model


def compute_metrics(pred):
    """
    Compute evaluation metrics (accuracy, precision, recall, f1) from predictions.
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate_model(model, tokenizer, eval_dataset,name="", batch_size=16,labels_list=None):
    """
    Evaluate a model using the provided evaluation dataset.
    Args:
        model: The trained model to evaluate.
        tokenizer: The tokenizer associated with the model.
        eval_dataset: The evaluation dataset.
        name (str): Name of the model for saving metrics.
        batch_size (int): Batch size for evaluation.
        labels_list (list): List of labels for the confusion matrix.
    Returns:
        metrics (dict): A dictionary of evaluation metrics.
    """

    eval_args = TrainingArguments(
        output_dir="./tmp_eval",    # temporary output
        per_device_eval_batch_size=batch_size,
        do_train=False,
        do_eval=True,
        logging_dir="./tmp_eval_logs",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,  
    )

    metrics = trainer.evaluate()

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save metrics to a JSON file
    metrics_path = f"/pvc/home/DL_hate_speech/results/metrics_{name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nGenerating confusion matrix...")
    preds_output = trainer.predict(eval_dataset)
    labels = preds_output.label_ids
    preds = np.argmax(preds_output.predictions, axis=1)


    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels_list)
    disp.plot(cmap="Blues", ax=ax,values_format='d')
    if labels_list is not None and len(labels_list) > 2:
        plt.xticks(rotation=90)
        
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("/pvc/home/DL_hate_speech/results/confusion_matrix_"+name+".png")

    return metrics
