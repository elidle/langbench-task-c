from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
import evaluate
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import os
from tqdm import tqdm
import re
from functools import partial

UPOS_LABELS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ",
    "DET", "INTJ", "NOUN", "NUM", "PART",
    "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
    "VERB", "X", "_"          # "_" = unspecified in UD
]
label2id = {l: i for i, l in enumerate(UPOS_LABELS)}
id2label = {i: l for i, l in enumerate(UPOS_LABELS)}

NUM_SAMPLES = 10000
EPOCHS = 1
LEARNING_RATE = 5e-5
BATCH_SIZE = 16

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

# helper functions to fix UD dataset so tokenizer can work properly
def parse_numpy_str(s):
    matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
    return [a if a else b for a, b in matches]

def fix_columns(example):
    example["tokens"] = parse_numpy_str(example["tokens"])
    example["upos"]   = parse_numpy_str(example["upos"])
    return example

# replicate previous methodology training models with exactly 10k samples
def fix_samples(dataset, num_samples, seed=42):
    size = len(dataset)
    if size >= num_samples:
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    else:
        repeats = num_samples // size
        rem     = num_samples % size

        repeated = [dataset] * repeats
        if rem > 0:
            repeated.append(dataset.shuffle(seed=seed).select(range(rem)))

        dataset = concatenate_datasets(repeated).shuffle(seed=seed)
    
    return dataset

def preprocess(examples, tokenizer):
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, max_length=256, padding="max_length")

    all_labels = []
    for i, upos_seq in enumerate(examples["upos"]):
        word_ids = tokenized.word_ids(batch_index=i)
        labels, prev_word_id = [], None

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                if word_id < len(upos_seq):
                    labels.append(label2id.get(upos_seq[word_id], label2id["X"]))
                else:
                    labels.append(-100)
            else:
                labels.append(-100)
            prev_word_id = word_id
        
        all_labels.append(labels)
    
    tokenized["labels"] = all_labels
    return tokenized
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    predictions = [p for pred_row, label_row in zip(predictions, labels) for p, l in zip(pred_row, label_row) if l != -100]
    labels      = [l for label_row in labels for l in label_row if l != -100]

    acc_score = accuracy.compute(predictions=predictions, references=labels)
    f1_score  = f1.compute(predictions=predictions, references=labels, average='macro')

    return {**acc_score, **f1_score}

def finetune_ud_pos(language: str):
    model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(UPOS_LABELS), id2label=id2label, label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    train_dataset = load_dataset('csv', data_files=f'data/ud-pos/{language}_train.csv')['train'].map(fix_columns)
    eval_dataset  = load_dataset('csv', data_files=f'data/ud-pos/{language}_test.csv')['train'].map(fix_columns)
    train_dataset = fix_samples(train_dataset, NUM_SAMPLES)

    preprocess_fn = partial(preprocess, tokenizer=tokenizer)
    train_dataset = train_dataset.map(preprocess_fn, batched=True, remove_columns=["sent_id", "text", "tokens", "upos"])
    eval_dataset  = eval_dataset.map(preprocess_fn, batched=True, remove_columns=["sent_id", "text", "tokens", "upos"])

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"models/ud-pos/xlm-r/{language}",
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,

        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(f"models/ud-pos/xlm-r/{language}")
    tokenizer.save_pretrained(f"models/ud-pos/xlm-r/{language}")


def evaluate_ud_pos(language: str):
    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    languages = set()

    for filename in os.listdir("data/ud-pos"):
        if filename.endswith("_test.csv"):
            lang = filename.split("_")[0]
            languages.add(lang)

    model = AutoModelForTokenClassification.from_pretrained(f'models/ud-pos/xlm-r/{language}')
    tokenizer = AutoTokenizer.from_pretrained(f'models/ud-pos/xlm-r/{language}')

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    preprocess_fn = partial(preprocess, tokenizer=tokenizer)

    for task_lang in tqdm(languages):
        test_dataset = load_dataset('csv', data_files=f'data/ud-pos/{task_lang}_test.csv')['train'].map(fix_columns)
        test_dataset = test_dataset.map(preprocess_fn, batched=True)

        predictions_output = trainer.predict(test_dataset)

        acc_score = predictions_output.metrics['test_accuracy']
        f1_score  = predictions_output.metrics['test_f1']

        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['accuracy'].append(acc_score)
        results['f1_score'].append(f1_score)

        df_results = pd.DataFrame(results)

        output_dir = f'results/ud-pos/xlm-r'
        os.makedirs(output_dir, exist_ok=True)

        df_results.to_csv(f'{output_dir}/{language}.csv', index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune XLM-R in a speficied language and evaluate cross lingual transfer in the ud-pos dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in ud-pos.')
    # parser.add_argument('--model', type=str, required=True, choices=('xlm-r', 'mt5', 'llama3'),
    #                     help='Model to fine-tune')
    parser.add_argument('--eval_only', action='store_true',
                        help="Skip training and directly load finetuned model for evaluation")
    # parser.add_argument('--resume_step', type=int, default=None,
    #                     help="Resume training from checkpoint step number")
    args = parser.parse_args()

    print(f"Processing language {args.lang} on ud-pos dataset...")

    # Fine-tune model on transfer language
    if not args.eval_only:
        finetune_ud_pos(args.lang)

    # Evaluate fine-tuned model on all other languages in the dataset
    evaluate_ud_pos(args.lang)
