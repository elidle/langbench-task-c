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
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import os
from tqdm import tqdm
from utils import evaluate_model


def preprocess_taxi1500_data(language: str, labels: dict[str, int], splits: set[str]) -> DatasetDict:
    dataset = DatasetDict()
    for split in splits:
        df = pd.read_csv(f"data/taxi1500/{language}_{split}.csv", index_col=0)
        df['label'] = df['classification'].map(labels)
        df = df.drop(columns=['classification'])
        dataset[split] = Dataset.from_pandas(df)

    return dataset


def finetune_taxi1500(model_name: str, language: str):
    labels = {'Recommendation': 0, 'Faith': 1, 'Description': 2, 'Sin': 3, 'Grace': 4, 'Violence': 5}
    dataset = preprocess_taxi1500_data(language, labels, {'train', 'dev'})

    if model_name == "xlm-r":
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=len(labels))
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    elif model_name == "mt5":
        # TODO
        pass
    elif model_name == "llama3":
        # TODO
        pass

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        acc = accuracy.compute(predictions=predictions, references=labels)
        f1_score = f1.compute(predictions=predictions, references=labels, average='weighted')
        return {**acc, **f1_score}

    training_args = TrainingArguments(
        output_dir=f"models/taxi1500/{model_name}/{language}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f"logs/taxi1500/{model_name}/{language}",
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
        eval_dataset=tokenized_datasets['dev'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
    trainer.save_model(f"models/taxi1500/{model_name}/{language}")
    tokenizer.save_pretrained(f"models/taxi1500/{model_name}/{language}")


def evaluate_taxi1500(model_name: str, language: str):
    """
    Evaluate the model on a specific task language.

    Returns:
        dict: Evaluation results including accuracy and F1 score.
    """
    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    languages = set()

    for file in os.listdir('data/taxi1500'):
        languages.add(file.split('_')[0])

    labels = {'Recommendation': 0, 'Faith': 1, 'Description': 2, 'Sin': 3, 'Grace': 4, 'Violence': 5}

    model_path = f"models/taxi1500/{model_name}/{language}"

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(labels))
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test_datasets = {}
    for task_lang in languages:
        test_datasets[task_lang] = preprocess_taxi1500_data(task_lang, labels, {'test'})
    
    for task_lang in tqdm(languages):
        transfer_results = evaluate_model(
            dataset=test_datasets[task_lang],
            model=model,
            tokenizer=tokenizer,
            task_lang=task_lang,
            transfer_lang=language
        )
        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['accuracy'].append(transfer_results['accuracy'])
        results['f1_score'].append(transfer_results['f1_score'])

        df_results = pd.DataFrame(results)

        output_dir = f"results/taxi1500/{model_name}"
        os.makedirs(output_dir, exist_ok=True)

        df_results.to_csv(f'{output_dir}/{language}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune a model in a speficied language and evaluate cross lingual transfer in the Taxi1500 dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in Taxi1500.')
    parser.add_argument('--model', type=str, required=True, choices=('xlm-r', 'mt5', 'llama3'),
                        help='Model to fine-tune')
    args = parser.parse_args()

    # Fine-tune model on transfer language
    finetune_taxi1500(args.model, args.lang)

    # Evaluate fine-tuned model on all other languages in the datase
    evaluate_taxi1500(args.model, args.lang)