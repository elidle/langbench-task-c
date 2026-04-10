from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
import evaluate
from argparse import ArgumentParser
from datasets import load_from_disk
import pandas as pd
import os
from tqdm import tqdm
from functools import partial

evaluate.config.CACHE_DIRECTORY = "/tmp/evaluate_cache"

NER_LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-DATE', 'I-DATE']
label_to_id = {l: i for i, l in enumerate(NER_LABELS)}
id_to_label = {i: l for i, l in enumerate(NER_LABELS)}

seqeval = evaluate.load("seqeval")

EPOCHS = 50
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION = 1
BATCH_SIZE = 32
PATIENCE = 5

def preprocess(examples, tokenizer):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None

        aligned = []
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != prev_word_id:
                aligned.append(labels[word_id])
            else:
                aligned.append(-100)
            prev_word_id = word_id
        
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized

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

def calculate_macro_f1(metrics):
    type_f1s = [metrics[etype]["f1"] for etype in metrics if isinstance(metrics[etype], dict)]
    if len(type_f1s) == 0:
        macro_f1 = 0
    else:
        macro_f1 = sum(type_f1s) / len(type_f1s)
    return macro_f1

def finetune_masakhaner(language: str):
    model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(NER_LABELS), id2label=id_to_label, label2id=label_to_id)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    train_dataset = load_from_disk(f'data/masakhaner/{language}')["train"]
    eval_dataset = load_from_disk(f'data/masakhaner/{language}')["validation"]

    preprocess_fn = partial(preprocess, tokenizer=tokenizer)
    train_dataset = train_dataset.map(preprocess_fn, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset  = eval_dataset.map(preprocess_fn, batched=True, remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"models/masakhaner/xlm-r/{language}",
        
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,

        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        # gradient_accumulation_steps=GRADIENT_ACCUMULATION
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    trainer.train()

    trainer.save_model(f"models/masakhaner/xlm-r/{language}")
    tokenizer.save_pretrained(f"models/masakhaner/xlm-r/{language}")


def evaluate_masakhaner(language: str):
    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    languages = set()

    for folder in os.listdir('data/masakhaner'):
        languages.add(folder)

    model = AutoModelForTokenClassification.from_pretrained(f'models/masakhaner/xlm-r/{language}')
    tokenizer = AutoTokenizer.from_pretrained(f'models/masakhaner/xlm-r/{language}')

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(model=model, data_collator=data_collator, compute_metrics=compute_metrics)      

    preprocess_fn = partial(preprocess, tokenizer=tokenizer)

    for task_lang in tqdm(languages):
        test_dataset = load_from_disk(f'data/masakhaner/{task_lang}')["test"]
        test_dataset = test_dataset.map(preprocess_fn, batched=True, remove_columns=test_dataset.column_names)

        predictions_output = trainer.predict(test_dataset)

        acc_score = predictions_output.metrics['test_accuracy']
        f1_score  = predictions_output.metrics['test_f1']

        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['accuracy'].append(acc_score)
        results['f1_score'].append(f1_score)

        df_results = pd.DataFrame(results)

        output_dir = f'results/masakhaner/xlm-r'
        os.makedirs(output_dir, exist_ok=True)

        df_results.to_csv(f'{output_dir}/{language}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune a model in a speficied language and evaluate cross lingual transfer in the MasakhaNER2.0 dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in MasakhaNER2.0.')
    # parser.add_argument('--model', type=str, required=True, choices=('xlm-r',),
    #                     help='Model to fine-tune')
    parser.add_argument('--eval_only', action='store_true',
                        help="Skip training and directly load finetuned model for evaluation")
    # parser.add_argument('--resume_step', type=int, default=None,
    #                     help="Resume training from checkpoint step number")
    args = parser.parse_args()

    print(f"Processing model {args.model} and language {args.lang} on MasakhaNER2.0 dataset...")

    # Fine-tune model on transfer language
    if not args.eval_only:
        finetune_masakhaner(args.model)

    # Evaluate fine-tuned model on all other languages in the dataset
    evaluate_masakhaner(args.model)
