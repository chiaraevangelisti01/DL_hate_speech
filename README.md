# 🐶 Detecting Emerging Dog Whistles

This repository contains the code and resources for the project **Detecting Emerging Dog Whistles**, developed as part of the course **EE-559: Deep Learning** at EPFL. The project focuses on identifying subtle and coded language—commonly referred to as *dog whistles*—used to covertly convey controversial or extremist ideas in online text.

We build a custom model based on transformer architectures (e.g., BERT/RoBERTa), leveraging Masked Language Modeling (MLM) and supervised classification to train two classifiers:
- **Classifier 1**: Detects whether a dog whistle is present in the text.
- **Classifier 2**: If a dog whistle is detected, identifies the targeted group (e.g., racist, homophobic, etc.).


## Dataset

We use the [Dog Whistle Dataset](https://huggingface.co/datasets/AstroAure/dogwhistle_dataset), which is derived from the [Silent Signals Dataset](https://huggingface.co/datasets/SALT-NLP/silent_signals).

## Repository structure 
The repository is organized as follows:

```plaintext
.
├── models/                 # Folder to save the trained model
├── results/                # Folder containing the results (including metric dictionarries and confusion matrices)
├── scripts/                # Scripts to train / evaluate the models
│   ├── MLM.py                  # Code to fine tune the base model with MLM
│   ├── Classifier.py           # Code to train the two classifiers 
│   ├── evaluate_emerging.py    # Code to evaluate the model on emerging dog whistles
│   ├── test_sentence.py        # Code to test the model on a given sentence/text
│   ├── visualize_attention.py  # Code for attention visualisation
│   ├── find_fail.py  # Code for detecting model failure in test dataset
├── src/                    # Source code for data processing and model training
│   ├── load_data.py            # Code to load the pretrained models and the datasets
│   ├── train_test.py           # Code to train and evaluate our models 
│   ├── utils.py                # Utilities functions
├── tmp_eval/               # Source code for data processing and model training
│   ├── failed_cl1.txt          # Output of find_fail for 1st classifier
│   ├── failed_cl2.txt          # Output of find_fail for 2nd classifier
│   ├── head_clf1.html          # Head view attention visualization 1st classifier
│   ├── model_clf1.html         # Model view attention visualization 1st classifier
│   ├── head_clf2.html          # Head view attention visualization 2st classifier
│   ├── model_clf2.html         # Model view attention visualization 2st classifier
│   ├── head_mlm.html          # Head view attention visualization mlm model
│   ├── model_mlm.html         # Model view attention visualization mlm model
├── requirements.txt      # Dependencies
├── README.md             # This file
└── LICENSE
```

## 🚀 Getting Started

1. Clone the repository :

```bash
git clone https://github.com/chiaraevangelisti01/DL_hate_speech
```

2. Install the dependencies :

```bash
pip install -r requirements.txt
```

3. Train your model :
First run the `MLM.py` script to fine-tune the base model with Masked Language Modeling (MLM). This step allows the model to adapt to the specific language patterns of dog whistles.

Then, run the `Classifier.py` script to train the two classifiers. The first classifier detects if a dog whistle is used in the text, and the second classifier identifies the targeted group if a dog whistle is detected.

```bash
python scripts/MLM.py --model_path <your_model_path>  --model_name roberta-base
python scripts/Classifier.py --model_path <your_model_path> --model_name roberta-base --results_path <your_results_path>    
```
The arguments for these two scripts are :
- `--model_path` : Path to save the model (e.g path to you models/ folder)
- `--model_name` : Name of the base model to use (e.g. roberta-base, bert-base-uncased, etc.). Note that our implementation only accepts BERT or RoBERTa models. Default is `roberta-base`
- `--results_path` : Path to save the results (e.g path to you results/ folder)


The `Classifier.py` script will use the fine-tuned MLM model to train the two classifiers. Make sure to be consistent with the model name and path used in the `MLM.py` script.
The script will also evaluate the model on the test set and save the results in the `results/` folder. This test set contains dog whistles that may have been seen by the model during training, it is not the emerging dog whistles test set. 

4. Evaluate the model on emerging dog whistles :
To evaluate the model on emerging dog whistles, run the `evaluate_emerging.py` script. This script will load the trained model and evaluate it on the emerging dog whistles test set.
```bash
python scripts/evaluate_emerging.py --model_path <your_model_path> --model_name roberta-base --results_path <your_results_path>
```
The arguments for this script are the same as for the `Classifier.py` script. And the results will be saved in the `results/` folder.

5. Test the model on a given sentence :
To test the model on a given sentence or text, run the `test_sentence.py` script. This script will load the trained model and classify the input text.
```bash
python scripts/test_sentence.py --model_path <your_model_path> --model_name roberta-base --sentence "Your sentence here"
```
The script will print the classification results, indicating whether a dog whistle is detected and, if so, the targeted group.
Example : 
```bash
python scripts/test_sentence.py --model_path models/ --model_name roberta-base --sentence "We need to take our country back from the thugs ruining our cities."

Output :

Test sentence: We need to take our country back from the thugs ruining our cities.

Dogwhistle detected.

Target group: Racist 

```


6. Visualize attention :


To visualize attention weights for one or more models on a given sentence, run the `visualize_attention.py` script. The script loads your fine-tuned models and generates attention maps using [BertViz](https://github.com/jessevig/bertviz). The output is saved as HTML files that can be opened in any browser.

Basic usage:

```bash
python scripts/visualize_attention.py --model_path <your_model_path> --model_name roberta-base --vis_type model
```

You can optionally specify a custom sentence through the "--test_sentence" argument, else a random one will be sampled from the test dataset. You can also specify the type of visualization desired (model or head)

```bash
Output :
The script will generate one HTML file per task in the output directory, e.g.:
[✓] Saved head view for mlm → head_mlm.html
[✓] Saved head view for clf1 → head_clf1.html
[✓] Saved head view for clf2 → head_clf2.html

 [✓] All visualizations complete.
```

7. Identify misclassified sentences from the test set:

To identify and print sentences that were misclassified by the classifier, run the `print_misclassified.py` script. This script loads a fine-tuned classification model and compares its predictions to the true labels on the test set.

Basic usage:

```bash
python scripts/print_misclassified.py --model_path <your_model_path> --model_name roberta-base --classifier 1
python scripts/print_misclassified.py --model_path models/ --model_name roberta-base --classifier 2

Output: 
The script will print all misclassified sentences, showing both the predicted and true labels:

 Sentence: why on earth are you supporting this bill when these private property rights organizations are against it in america
 Predicted: 1 | True: 0

 Sentence: it cuts aid to poor children to pay for tax cuts for the rich number of children punished million
 Predicted: anti-liberal | True: racist


