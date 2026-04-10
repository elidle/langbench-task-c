from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import evaluate
from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk, DatasetDict
import pandas as pd
import os
from tqdm import tqdm

SIB_LABELS = ['science/technology', 'travel', 'politics', 'sports', 'health', 'entertainment', 'geography']
label2id = {label: idx for idx, label in enumerate(SIB_LABELS)}
id2label = {idx: label for label, idx in label2id.items()}


def finetune_sib200(model_name: str, language: str):
    # dataset = load_dataset('Davlan/sib200', language)
    dataset = load_from_disk(f'data/sib200/{language}')

    if model_name == "xlm-r":
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=len(SIB_LABELS))
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def encode_labels(example):
        example["label"] = label2id[example["category"]]
        return example

    dataset = dataset.map(encode_labels)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        acc = accuracy.compute(predictions=predictions, references=labels)
        f1_score = f1.compute(predictions=predictions, references=labels, average='macro')
        return {**acc, **f1_score}

    training_args = TrainingArguments(
        output_dir=f"models/sib200/{model_name}/{language}",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f"logs/sib200/{model_name}/{language}",
        eval_strategy="steps",
        eval_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
    trainer.save_model(f"models/sib200/{model_name}/{language}")
    tokenizer.save_pretrained(f"models/sib200/{model_name}/{language}")


def evaluate_sib200(model_name: str, language: str):
    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    languages = set()

    for folder in os.listdir('data/sib200'):
        languages.add(folder)

    model = AutoModelForSequenceClassification.from_pretrained(f'models/sib200/{model_name}/{language}', num_labels=len(SIB_LABELS))
    tokenizer = AutoTokenizer.from_pretrained(f'models/sib200/{model_name}/{language}')

    test_datasets = {}
    for task_lang in languages:
        test_datasets[task_lang] = load_from_disk(f'data/sib200/{task_lang}')['test']

        def encode_labels(example):
            example["label"] = label2id[example["category"]]
            return example

        test_datasets[task_lang] = DatasetDict({'test': test_datasets[task_lang].map(encode_labels, num_proc=4)})
    
    for task_lang in tqdm(languages):
        # transfer_results = evaluate_model(
        #     dataset=test_datasets[task_lang],
        #     model=model,
        #     tokenizer=tokenizer,
        #     task_lang=task_lang,
        #     transfer_lang=language
        # )

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

        tokenized_datasets = test_datasets[task_lang]['test'].map(tokenize_function,batched=True)

        trainer = Trainer(model=model)
        
        predictions = trainer.predict(tokenized_datasets)
        preds = predictions.predictions.argmax(axis=-1)
        labels = tokenized_datasets['label']

        accuracy = evaluate.load('accuracy')
        f1 = evaluate.load('f1')

        accuracy = accuracy.compute(predictions=preds, references=labels)
        f1_score = f1.compute(predictions=preds, references=labels, average='macro')

        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['accuracy'].append(accuracy['accuracy'])
        results['f1_score'].append(f1_score['f1'])

        df_results = pd.DataFrame(results)

        output_dir = f'results/sib200/{model_name}'
        os.makedirs(output_dir, exist_ok=True)

        df_results.to_csv(f'{output_dir}/{language}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune a model in a speficied language and evaluate cross lingual transfer in the SIB200 dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in SIB200.')
    parser.add_argument('--model', type=str, required=True, choices=('xlm-r',),
                        help='Model to fine-tune')
    parser.add_argument('--eval_only', action='store_true',
                            help="Skip training and directly load finetuned model for evaluation")
    # parser.add_argument('--resume_step', type=int, default=None,
    #                     help="Resume training from checkpoint step number")
    args = parser.parse_args()

    # Fine-tune model on transfer language
    if not args.eval_only:
        finetune_sib200(args.model, args.lang)

    # Evaluate fine-tuned model on all other languages in the dataset
    evaluate_sib200(args.model, args.lang)