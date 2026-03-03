from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
import evaluate
import tempfile
from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk, DatasetDict
import pandas as pd
import os
from tqdm import tqdm
import torch

evaluate.config.CACHE_DIRECTORY = "/tmp/evaluate_cache"

label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for i, l in enumerate(label_list)}

def calculate_macro_f1(metrics):
    """Calculate macro f1 from seqeval results"""
    type_f1s = [metrics[etype]["f1"] for etype in metrics if isinstance(metrics[etype], dict)]
    macro_f1 = sum(type_f1s) / len(type_f1s)
    return macro_f1

def finetune_wikiann(model_name: str, language: str, resume_step: int = None):
    dataset = load_from_disk(f'data/wikiann/{language}')

    if model_name == "xlm-r":
        model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(label_list), id2label=id_to_label, label2id=label_to_id)
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        all_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_word_id = None

            aligned = []
            for word_id in word_ids:
                if word_id is None: # Special tokens
                    aligned.append(-100)
                elif word_id != prev_word_id: # First sub-token
                    aligned.append(labels[word_id])
                else: # Subsequent sub-token
                    aligned.append(-100)
                prev_word_id = word_id
            
            all_labels.append(aligned)
        tokenized["labels"] = all_labels
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        true_preds = [[id_to_label[p] for p, l in zip(pred_row, label_row) if l != -100]
                    for pred_row, label_row in zip(predictions, labels)]
        true_labels = [[id_to_label[l] for l in label_row if l != -100]
                    for label_row in labels]

        metrics = seqeval.compute(predictions=true_preds, references=true_labels)

        macro_f1 = calculate_macro_f1(metrics)
        return {
            "f1": macro_f1,
            "accuracy": metrics["overall_accuracy"]
        }

    training_args = TrainingArguments(
        output_dir=f"models/wikiann/{model_name}/{language}",
        logging_dir=f"logs/wikiann/{model_name}/{language}",
        eval_strategy="steps",
        save_strategy="steps",
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=5
    )

    if model_name == "xlm-r":
        training_args.num_train_epochs = 5
        training_args.learning_rate = 2e-5
        training_args.per_device_train_batch_size = 16
        training_args.per_device_eval_batch_size  = 16

        training_args.weight_decay = 0.01
        training_args.warmup_ratio = 0.1

        training_args.eval_steps = 300
        training_args.save_steps = 300

        training_args.fp16 = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()

    if model_name == "llama3":
        model.save_pretrained(f"models/wikiann/{model_name}/{language}")
    else:
        trainer.save_model(f"models/wikiann/{model_name}/{language}")

    tokenizer.save_pretrained(f"models/wikiann/{model_name}/{language}")


def evaluate_wikiann(model_name: str, language: str):
    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    languages = set()

    for folder in os.listdir('data/wikiann'):
        languages.add(folder)

    if model_name == "xlm-r":
        model = AutoModelForTokenClassification.from_pretrained(f'models/wikiann/{model_name}/{language}')
        tokenizer = AutoTokenizer.from_pretrained(f'models/wikiann/{model_name}/{language}')

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        all_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_word_id = None

            aligned = []
            for word_id in word_ids:
                if word_id is None: # Special tokens
                    aligned.append(-100)
                elif word_id != prev_word_id: # First sub-token
                    aligned.append(labels[word_id])
                else: # Subsequent sub-token
                    aligned.append(-100)
                prev_word_id = word_id
            
            all_labels.append(aligned)
        tokenized["labels"] = all_labels
        return tokenized
    
    seqeval = evaluate.load("seqeval")

    for task_lang in tqdm(languages):
        dataset = load_from_disk(f'data/wikiann/{task_lang}')

        if model_name == "xlm-r":
            test_ds = dataset["test"].map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=dataset["test"].column_names
            )

            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        trainer = Trainer(model=model, data_collator=data_collator)

        predictions_output = trainer.predict(test_ds)
        predictions = predictions_output.predictions
        labels = predictions_output.label_ids

        if model_name == "xlm-r":
            predictions = predictions.argmax(axis=-1)

        true_preds = [[id_to_label[p] for p, l in zip(pred_row, label_row) if l != -100]
                    for pred_row, label_row in zip(predictions, labels)]
        true_labels = [[id_to_label[l] for l in label_row if l != -100]
                    for label_row in labels]

        metrics = seqeval.compute(predictions=true_preds, references=true_labels)
        macro_f1 = calculate_macro_f1(metrics)

        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['accuracy'].append(metrics['overall_accuracy'])
        results['f1_score'].append(macro_f1)

        df_results = pd.DataFrame(results)

        output_dir = f'results/wikiann/{model_name}'
        os.makedirs(output_dir, exist_ok=True)

        df_results.to_csv(f'{output_dir}/{language}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune a model in a speficied language and evaluate cross lingual transfer in the WikiAnn dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in WikiAnn.')
    parser.add_argument('--model', type=str, required=True, choices=('xlm-r', 'mt5', 'llama3'),
                        help='Model to fine-tune')
    parser.add_argument('--eval_only', action='store_true',
                        help="Skip training and directly load finetuned model for evaluation")
    parser.add_argument('--resume_step', type=int, default=None,
                        help="Resume training from checkpoint step number")
    args = parser.parse_args()

    print(f"Processing model {args.model} and language {args.lang} on WikiAnn dataset...")

    # Fine-tune model on transfer language
    if not args.eval_only:
        finetune_wikiann(args.model, args.lang)

    # Evaluate fine-tuned model on all other languages in the dataset
    evaluate_wikiann(args.model, args.lang)
