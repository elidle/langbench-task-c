from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate
from argparse import ArgumentParser
from datasets import load_from_disk
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from functools import partial
from sklearn.metrics import f1_score

LEARNING_RATE = 1e-4
MAX_SEQ_LEN = 256
MAX_TARGET_LEN = 4
BATCH_SIZE = 16
# GRADIENT_ACCUMULATION = 8
EPOCHS = 30
# MAX_STEPS = 500
SCHEDULER = "linear"
OPTIM = "adamw_torch"
PATIENCE = 10
STEPS = 20

MODEL_SIZE     = "base"
MODEL_NAME     = f"google/mt5-{MODEL_SIZE}"

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

SIB_LABELS = ['science/technology', 'travel', 'politics', 'sports', 'health', 'entertainment', 'geography']
label_to_id = {label: idx for idx, label in enumerate(SIB_LABELS)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

def preprocess(examples, tokenizer):
    # Input:  "sib200: <sentence>"
    # Output: "science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"
    inputs = [f"sib200: {s}" for s in examples["text"]]
    # targets = [id_to_label[id] for id in examples["category"]]
    targets = list(examples["category"])

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SEQ_LEN,
        padding=False,
        truncation=True,
    )
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LEN,
        padding=False,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds, tokenizer):
    predictions, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    acc_score = sum(p == l for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_labels)
    macro_f1  = f1_score(decoded_labels, decoded_preds, average="macro", zero_division=0)

    return {
        "accuracy": acc_score,
        "f1":       macro_f1,
    }


def finetune_sib200_mt5(language: str):
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    train_dataset = load_from_disk(f'data/sib200/{language}')["train"]
    eval_dataset = load_from_disk(f'data/sib200/{language}')["validation"]

    preprocess_fn = partial(preprocess, tokenizer=tokenizer)
    train_dataset = train_dataset.map(preprocess_fn, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset  = eval_dataset.map(preprocess_fn, batched=True, remove_columns=eval_dataset.column_names)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"models/sib200/mt5/{language}",

        eval_strategy="steps",
        eval_steps=STEPS,
        save_strategy="steps",
        save_steps=STEPS,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=1,

        num_train_epochs=EPOCHS,
        # max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        # gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        # warmup_ratio=WARMUP_RATIO,

        learning_rate=LEARNING_RATE,
        lr_scheduler_type=SCHEDULER,
        optim=OPTIM,

        bf16=True,
        bf16_full_eval=True,

        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    trainer.train()

    trainer.save_model(f"models/sib200/mt5/{language}")
    tokenizer.save_pretrained(f"models/sib200/mt5/{language}")


def evaluate_sib200_mt5(language: str):
    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    languages = set()

    for folder in os.listdir('data/sib200'):
        languages.add(folder)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(f"models/sib200/mt5/{language}")
    tokenizer = AutoTokenizer.from_pretrained(f"models/sib200/mt5/{language}")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    args = Seq2SeqTrainingArguments(
        output_dir=f"models/sib200/mt5/{language}",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        bf16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer)
    )

    preprocess_fn = partial(preprocess, tokenizer=tokenizer)

    for task_lang in tqdm(languages):
        test_dataset = load_from_disk(f'data/sib200/{task_lang}')["test"]
        test_dataset = test_dataset.map(preprocess_fn, batched=True, remove_columns=test_dataset.column_names)

        predictions_output = trainer.predict(test_dataset)

        acc_score = predictions_output.metrics['test_accuracy']
        macro_f1  = predictions_output.metrics['test_f1']

        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['accuracy'].append(acc_score)
        results['f1_score'].append(macro_f1)

        df_results = pd.DataFrame(results)

        output_dir = f'results/sib200/mt5'
        os.makedirs(output_dir, exist_ok=True)

        df_results.to_csv(f'{output_dir}/{language}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune mT5 in a specified language and evaluate cross lingual transfer in the SIB200 dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in SIB200.')
    parser.add_argument('--eval_only', action='store_true',
                        help="Skip training and directly load finetuned model for evaluation")
    # parser.add_argument('--resume_step', type=int, default=None,
    #                     help="Resume training from checkpoint step number")
    args = parser.parse_args()

    print(f"Processing mT5 and language {args.lang} on SIB200 dataset...")

    if not args.eval_only:
        finetune_sib200_mt5(args.lang)

    evaluate_sib200_mt5(args.lang)