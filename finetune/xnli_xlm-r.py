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
from datasets import load_from_disk
import pandas as pd
import os
from tqdm import tqdm
from functools import partial

XNLI_LABELS = ["entailment", "neutral", "contradiction"]
label2id = {l: i for i, l in enumerate(XNLI_LABELS)}
id2label = {i: l for i, l in enumerate(XNLI_LABELS)}
XNLI_LANGUAGES = ["en", "fr", "es", "de", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"]

EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
STEPS = 2000
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 15

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def preprocess(examples, tokenizer):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=256)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    acc_score = accuracy.compute(predictions=predictions, references=labels)
    f1_score  = f1.compute(predictions=predictions, references=labels, average='macro')

    return {**acc_score, **f1_score}

def finetune_xnli_xlmr(language: str):
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=len(XNLI_LABELS), id2label=id2label, label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    train_dataset = load_from_disk(f"data/xnli/{language}")["train"]
    eval_dataset  = load_from_disk(f"data/xnli/{language}")["validation"]
    
    preprocess_fn = partial(preprocess, tokenizer=tokenizer)
    train_dataset = train_dataset.map(preprocess_fn, batched=True)
    eval_dataset  = eval_dataset.map(preprocess_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"models/xnli/xlm-r/{language}",

        eval_strategy="steps",
        save_strategy="steps",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,

        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size =BATCH_SIZE,

        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,

        eval_steps=STEPS,
        save_steps=STEPS,
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

    trainer.save_model(f"models/xnli/xlm-r/{language}")
    tokenizer.save_pretrained(f"models/xnli/xlm-r/{language}")


def evaluate_xnli_xlmr(language: str):
    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    model = AutoModelForSequenceClassification.from_pretrained(f"models/xnli/xlm-r/{language}")
    tokenizer = AutoTokenizer.from_pretrained(f"models/xnli/xlm-r/{language}")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )      

    preprocess_fn = partial(preprocess, tokenizer=tokenizer)

    for task_lang in tqdm(XNLI_LANGUAGES):
        test_dataset = load_from_disk(f"data/xnli/{task_lang}")["test"]
        test_dataset = test_dataset.map(preprocess_fn, batched=True)

        predictions_output = trainer.predict(test_dataset)

        acc_score = predictions_output.metrics['test_accuracy']
        f1_score  = predictions_output.metrics['test_f1']

        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['accuracy'].append(acc_score)
        results['f1_score'].append(f1_score)

        df_results = pd.DataFrame(results)

        output_dir = f'results/xnli/xlm-r'
        os.makedirs(output_dir, exist_ok=True)

        df_results.to_csv(f'{output_dir}/{language}.csv', index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune XLM-R in a specified language and evaluate cross lingual transfer in the XNLI dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in XNLI.')
    parser.add_argument('--eval_only', action='store_true',
                        help="Skip training and directly load finetuned model for evaluation")
    # parser.add_argument('--resume_step', type=int, default=None,
    #                     help="Resume training from checkpoint step number")
    args = parser.parse_args()

    print(f"Processing XLM-R and language {args.lang} on XNLI dataset...")

    # Fine-tune model on transfer language
    if not args.eval_only:
        finetune_xnli_xlmr(args.lang)
    
    # Evaluate fine-tuned model on all other languages in the dataset
    evaluate_xnli_xlmr(args.lang)
