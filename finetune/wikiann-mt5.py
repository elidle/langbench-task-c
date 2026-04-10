from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
# import evaluate
from argparse import ArgumentParser
from datasets import load_from_disk
import pandas as pd
import os
from tqdm import tqdm
from functools import partial
from collections import Counter
import numpy as np

NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label_to_id = {l: i for i, l in enumerate(NER_LABELS)}
id_to_label = {i: l for i, l in enumerate(NER_LABELS)}

LEARNING_RATE = 3e-5
MAX_SEQ_LEN = 1024
MAX_TARGET_LEN = 32
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2
# EPOCHS = 75
MAX_STEPS = 5000
SCHEDULER = "constant"
OPTIM = "adamw_torch"
PATIENCE = 5
STEPS = 200

def preprocess(examples, tokenizer):
    inputs  = []
    targets = []

    for tokens, ner_tags in zip(examples["tokens"], examples["ner_tags"]):
        sentence = " ".join(tokens)
        inputs.append(f"ner: {sentence}")
        
        entity_tags = [id_to_label[tag] for tag in ner_tags if id_to_label[tag] != "O"]
        targets.append(" ".join(entity_tags) if entity_tags else "None")

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

def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred

    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
    preds = np.where(preds < tokenizer.vocab_size, preds, tokenizer.pad_token_id)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    exact_matches = []
    for pred, label in zip(decoded_preds, decoded_labels):
        exact_matches.append(int(pred.strip() == label.strip()))

    # Token-level F1 over the entity tag sequences
    all_pred_tags  = [p.strip().split() for p in decoded_preds]
    all_true_tags  = [l.strip().split() for l in decoded_labels]

    f1_scores = []
    for pred_tags, true_tags in zip(all_pred_tags, all_true_tags):
        pred_set = Counter(pred_tags)
        true_set = Counter(true_tags)
        common = pred_set & true_set
        tp = sum(common.values())
        fp = sum((pred_set - true_set).values())
        fn = sum((true_set - pred_set).values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)

    return {
        "f1": macro_f1,
        "accuracy": np.mean(exact_matches),
    }

def finetune_wikiann_mt5(language: str):
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    train_dataset = load_from_disk(f'data/wikiann/{language}')["train"]
    eval_dataset = load_from_disk(f'data/wikiann/{language}')["validation"]
    eval_dataset = eval_dataset.select(range(min(1000, len(eval_dataset))))

    preprocess_fn = partial(preprocess, tokenizer=tokenizer)
    train_dataset = train_dataset.map(preprocess_fn, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset  = eval_dataset.map(preprocess_fn, batched=True, remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"models/wikiann/mt5/{language}",

        eval_strategy="steps",
        eval_steps=STEPS,
        save_strategy="steps",
        save_steps=STEPS,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,

        # num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
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
        generation_num_beams=1
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    trainer.train()
    
    trainer.save_model(f"models/wikiann/mt5/{language}")

    tokenizer.save_pretrained(f"models/wikiann/mt5/{language}")


def evaluate_wikiann_mt5(language: str):
    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    languages = set()

    for folder in os.listdir('data/wikiann'):
        languages.add(folder)

    model = AutoModelForSeq2SeqLM.from_pretrained(f'models/wikiann/mt5/{language}')
    tokenizer = AutoTokenizer.from_pretrained(f'models/wikiann/mt5/{language}')

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    args = Seq2SeqTrainingArguments(
        output_dir=f"models/wikiann/mt5/{language}",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        generation_num_beams=1,
        bf16=True,
        per_device_eval_batch_size=BATCH_SIZE * 2,
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
        test_dataset = load_from_disk(f'data/wikiann/{task_lang}')["test"]
        test_dataset = test_dataset.map(preprocess_fn, batched=True, remove_columns=test_dataset.column_names)

        predictions_output = trainer.predict(test_dataset)

        acc_score = predictions_output.metrics['test_accuracy']
        f1        = predictions_output.metrics['test_f1']

        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['accuracy'].append(acc_score)
        results['f1_score'].append(f1)

        df_results = pd.DataFrame(results)

        output_dir = f'results/wikiann/mt5'
        os.makedirs(output_dir, exist_ok=True)

        df_results.to_csv(f'{output_dir}/{language}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune a model in a speficied language and evaluate cross lingual transfer in the WikiAnn dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in WikiAnn.')
    # parser.add_argument('--model', type=str, required=True, choices=('xlm-r', 'mt5', 'llama3'),
    #                     help='Model to fine-tune')
    parser.add_argument('--eval_only', action='store_true',
                        help="Skip training and directly load finetuned model for evaluation")
    # parser.add_argument('--resume_step', type=int, default=None,
    #                     help="Resume training from checkpoint step number")
    args = parser.parse_args()

    print(f"Processing mt5 and language {args.lang} on WikiAnn dataset...")

    # Fine-tune model on transfer language
    if not args.eval_only:
        finetune_wikiann_mt5(args.lang)

    # Evaluate fine-tuned model on all other languages in the dataset
    evaluate_wikiann_mt5(args.lang)
