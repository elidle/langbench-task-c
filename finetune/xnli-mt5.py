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

XNLI_LABELS = ["entailment", "neutral", "contradiction"]
id_to_label = {i: l for i, l in enumerate(XNLI_LABELS)}

XNLI_LANGUAGES = ["en", "fr", "es", "de", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"]

LEARNING_RATE = 3e-5
MAX_SEQ_LEN = 1024
MAX_TARGET_LEN = 8
BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 8
EPOCHS = 5
SCHEDULER = "constant"
OPTIM = "adamw_torch"
PATIENCE = 5
SAVE_STEPS = 200
EVAL_STEPS = 200

MODEL_SIZE     = "base"
MODEL_NAME     = f"google/mt5-{MODEL_SIZE}"

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def preprocess(examples, tokenizer):
    # Input:  "xnli: premise: <premise> hypothesis: <hypothesis>"
    # Output: "entailment" / "neutral" / "contradiction"
    inputs = [
        f"xnli: premise: {p} hypothesis: {h}"
        for p, h in zip(examples["premise"], examples["hypothesis"])
    ]
    targets = [id_to_label[id] for id in examples["label"]]

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

def finetune_xnli_mt5(language: str):
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    train_dataset = load_from_disk(f"data/xnli/{language}")["train"]
    eval_dataset  = load_from_disk(f"data/xnli/{language}")["validation"]
    
    preprocess_fn = partial(preprocess, tokenizer=tokenizer)
    train_dataset = train_dataset.map(preprocess_fn, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset  = eval_dataset.map(preprocess_fn, batched=True, remove_columns=eval_dataset.column_names)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"/models/xnli/mt5/{language}",

        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,

        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

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

    trainer.save_model(f"models/xnli/mt5/{language}")
    tokenizer.save_pretrained(f"models/xnli/mt5/{language}")


def evaluate_xnli_mt5(language: str):
    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }
        
    model = AutoModelForSeq2SeqLM.from_pretrained(f"models/xnli/mt5/{language}")
    tokenizer = AutoTokenizer.from_pretrained(f"models/xnli/mt5/{language}")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    args = Seq2SeqTrainingArguments(
        output_dir=f"models/wikiann/mt5/{language}",
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

    for task_lang in tqdm(XNLI_LANGUAGES):
        test_dataset = load_from_disk(f"data/xnli/{task_lang}")["test"]
        preprocess_fn = partial(preprocess, tokenizer=tokenizer)
        test_dataset = test_dataset.map(preprocess_fn, batched=True)

        predictions_output = trainer.predict(test_dataset)

        acc_score = predictions_output.metrics['test_accuracy']
        macro_f1  = predictions_output.metrics['test_f1']

        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['accuracy'].append(acc_score)
        results['f1_score'].append(macro_f1)

    df_results = pd.DataFrame(results)

    output_dir = f'results/xnli/mt5'
    os.makedirs(output_dir, exist_ok=True)

    df_results.to_csv(f'{output_dir}/{language}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune mT5 in a specified language and evaluate cross lingual transfer in the XNLI dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in XNLI.')
    parser.add_argument('--eval_only', action='store_true',
                        help="Skip training and directly load finetuned model for evaluation")
    # parser.add_argument('--resume_step', type=int, default=None,
    #                     help="Resume training from checkpoint step number")
    args = parser.parse_args()

    print(f"Processing mT5 and language {args.lang} on XNLI dataset...")

    if not args.eval_only:
        finetune_xnli_mt5(args.lang)

    evaluate_xnli_mt5(args.lang)