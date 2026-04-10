from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from argparse import ArgumentParser
from datasets import load_from_disk
import pandas as pd
import os
from tqdm import tqdm
from functools import partial
import numpy as np
import evaluate
import collections
import string
import re

LEARNING_RATE = 3e-5
MAX_SEQ_LEN = 1024
MAX_TARGET_LEN = 128
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2
MAX_STEPS = 5000
SCHEDULER = "constant"
OPTIM = "adamw_torch"
PATIENCE = 5
STEPS = 200

def _filter_by_language(dataset, language: str):
    return dataset.filter(lambda x: x["id"].startswith(language))


def _extract_languages(dataset):
    languages = set()
    for example in dataset:
        if "id" in example and isinstance(example["id"], str):
            lang = example["id"].split("-", 1)[0]
            languages.add(lang)
    return languages


def _normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation/articles/whitespace (mirrors SQuAD eval)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))

def preprocess_train(examples, tokenizer):
    """
    Format: "question: <Q>  context: <C>"  ->  "<answer text>"
    For unanswerable questions we target the empty string.
    """
    inputs  = []
    targets = []

    for question, context, answers in zip(
        examples["question"], examples["context"], examples["answers"]
    ):
        inputs.append(f"question: {question.strip()}  context: {context}")

        if len(answers["text"]) == 0:
            targets.append("")
        else:
            targets.append(answers["text"][0])

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


def preprocess_eval(examples, tokenizer):
    """Same as training but we also keep the example id for answer extraction."""
    model_inputs = preprocess_train(examples, tokenizer)
    model_inputs["example_id"] = examples["id"]
    return model_inputs

def compute_metrics(eval_pred, tokenizer):
    """
    Token-level F1 / exact-match over decoded strings.
    Used during training for model selection.
    """
    preds, labels = eval_pred

    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
    preds = np.where(preds < tokenizer.vocab_size, preds, tokenizer.pad_token_id)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    exact_matches = []
    f1_scores     = []

    for pred, label in zip(decoded_preds, decoded_labels):
        pred_norm  = _normalize_answer(pred)
        label_norm = _normalize_answer(label)

        exact_matches.append(int(pred_norm == label_norm))

        pred_tokens  = pred_norm.split()
        label_tokens = label_norm.split()

        common = collections.Counter(pred_tokens) & collections.Counter(label_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            f1_scores.append(0.0)
        else:
            precision = num_common / len(pred_tokens)
            recall    = num_common / len(label_tokens)
            f1_scores.append(2 * precision * recall / (precision + recall))

    return {
        "exact_match": np.mean(exact_matches),
        "f1": np.mean(f1_scores),
    }


def finetune_tydiqa_mt5(language: str):
    model     = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    dataset = load_from_disk("data/tydiqa")

    train_dataset = _filter_by_language(dataset["train"],      language)
    eval_dataset  = _filter_by_language(dataset["validation"], language)

    preprocess_train_fn = partial(preprocess_train, tokenizer=tokenizer)
    preprocess_eval_fn  = partial(preprocess_eval,  tokenizer=tokenizer)

    train_dataset = train_dataset.map(
        preprocess_train_fn, batched=True, remove_columns=train_dataset.column_names
    )
    eval_dataset  = eval_dataset.map(
        preprocess_eval_fn, batched=True, remove_columns=eval_dataset.column_names
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"models/tydiqa/mt5/{language}",

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
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()

    trainer.save_model(f"models/tydiqa/mt5/{language}")
    tokenizer.save_pretrained(f"models/tydiqa/mt5/{language}")

def evaluate_tydiqa_mt5(language: str):
    results = {
        "task_lang":   [],
        "transfer_lang": [],
        "exact_match": [],
        "f1_score":    [],
    }

    dataset   = load_from_disk("data/tydiqa")
    languages = _extract_languages(dataset["validation"])

    model     = AutoModelForSeq2SeqLM.from_pretrained(f"models/tydiqa/mt5/{language}")
    tokenizer = AutoTokenizer.from_pretrained(f"models/tydiqa/mt5/{language}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, pad_to_multiple_of=8
    )

    eval_args = Seq2SeqTrainingArguments(
        output_dir=f"models/tydiqa/mt5/{language}",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        generation_num_beams=1,
        bf16=True,
        per_device_eval_batch_size=BATCH_SIZE * 2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
    )

    preprocess_eval_fn = partial(preprocess_eval, tokenizer=tokenizer)
    metric = evaluate.load("squad")

    for task_lang in tqdm(languages):
        eval_dataset = _filter_by_language(dataset["validation"], task_lang)

        # Keep original examples for SQuAD-style references
        original_examples = eval_dataset

        tokenized = eval_dataset.map(
            preprocess_eval_fn,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

        # Retain example_id through prediction
        example_ids = tokenized["example_id"]
        tokenized_no_id = tokenized.remove_columns(["example_id"])

        predictions_output = trainer.predict(tokenized_no_id)
        pred_ids = predictions_output.predictions
        pred_ids = np.where(pred_ids >= 0, pred_ids, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # Build id -> prediction map (last prediction wins for multi-chunk, though
        # mT5 sees the full context so there's only one chunk per example here)
        pred_map = {}
        for eid, pred_text in zip(example_ids, decoded_preds):
            pred_map[eid] = pred_text.strip()

        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in pred_map.items()
        ]

        references = [
            {"id": ex["id"], "answers": ex["answers"]}
            for ex in original_examples
        ]

        metric_scores = metric.compute(
            predictions=formatted_predictions,
            references=references,
        )

        results["task_lang"].append(task_lang)
        results["transfer_lang"].append(language)
        results["exact_match"].append(metric_scores["exact_match"])
        results["f1_score"].append(metric_scores["f1"])

        output_dir = "results/tydiqa/mt5"
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(f"{output_dir}/{language}.csv", index=False)

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Fine-tune mT5 on a given language and evaluate cross-lingual "
                    "transfer on the TyDiQA dataset."
    )
    parser.add_argument(
        "--lang", type=str, required=True,
        help="Language to fine-tune on. Must follow the TyDiQA id convention "
             "(e.g. 'english', 'arabic').",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training and load the already fine-tuned model for evaluation.",
    )
    args = parser.parse_args()

    print(f"Processing mt5 and language '{args.lang}' on TyDiQA dataset...")

    if not args.eval_only:
        finetune_tydiqa_mt5(args.lang)

    evaluate_tydiqa_mt5(args.lang)