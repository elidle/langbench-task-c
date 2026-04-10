from transformers import (
    AutoModelForQuestionAnswering,
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
import collections
import numpy as np


def _filter_by_language(dataset, language: str):
    return dataset.filter(lambda x: x["id"].startswith(language))


def _extract_languages(dataset):
    # LANGUAGES_LIST = ['arabic', 'bengali', 'english', 'finnish', 'indonesian', 'korean', 'russian', 'swahili', 'telugu']
    languages = set()
    for example in dataset:
        if "id" in example and isinstance(example["id"], str):
            lang = example["id"].split("-", 1)[0]
            languages.add(lang)
    return languages


def finetune_tydiqa(model_name: str, language: str, resume_step: int = None):
    dataset = load_from_disk('data/tydiqa')

    if model_name == "xlm-r":
        model = AutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_dataset = _filter_by_language(dataset["train"], language)

    def preprocess(examples):
        questions = [q.strip() for q in examples["question"]]

        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answers = examples["answers"][sample_idx]

            if len(answers["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                sequence_ids = inputs.sequence_ids(i)

                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx

                while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                if not (offsets[context_start][0] <= start_char and
                        offsets[context_end][1] >= end_char):
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    idx = context_start
                    while idx <= context_end and offsets[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offsets[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

        return inputs
        
    def preprocess_validation(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs["offset_mapping"]
        example_ids = []
        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_mapping[i]
            example_ids.append(examples["id"][sample_idx])
            sequence_ids = inputs.sequence_ids(i)
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(offset_mapping[i])
            ]
        inputs["example_id"] = example_ids
        return inputs

    eval_dataset_raw = _filter_by_language(dataset["validation"], language)
    eval_features = eval_dataset_raw.map(
        preprocess_validation,
        batched=True,
        remove_columns=eval_dataset_raw.column_names
    )
    eval_features_no_id = eval_features.remove_columns(["example_id", "offset_mapping"])

    def compute_metrics(pred):
        try:
            start_logits, end_logits = pred.predictions

            example_to_features = collections.defaultdict(list)
            for idx, feature in enumerate(eval_features):
                example_to_features[feature["example_id"]].append(idx)

            final_predictions = {}
            for example in eval_dataset_raw:
                example_id = example["id"]
                best_score = -1e9
                best_answer = ""
                for idx in example_to_features[example_id]:
                    offsets = eval_features[idx]["offset_mapping"]
                    start_indexes = np.argsort(start_logits[idx])[-20:]
                    end_indexes = np.argsort(end_logits[idx])[-20:]
                    for start in start_indexes:
                        for end in end_indexes:
                            if offsets[start] is None or offsets[end] is None:
                                continue
                            if end < start or end - start > 30:
                                continue
                            score = start_logits[idx][start] + end_logits[idx][end]
                            if score > best_score:
                                best_answer = example["context"][offsets[start][0]:offsets[end][1]]
                                best_score = score
                final_predictions[example_id] = best_answer

            metric = evaluate.load("squad")
            metric_scores = metric.compute(
                predictions=[{"id": k, "prediction_text": v} for k, v in final_predictions.items()],
                references=[{"id": ex["id"], "answers": ex["answers"]} for ex in eval_dataset_raw],
            )
            return {"f1": metric_scores["f1"], "exact_match": metric_scores["exact_match"]}
        except Exception as e:
            print(f"compute_metrics failed: {e}")
            raise  # re-raise so training fails loudly instead of silently

    tokenized_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir=f"models/tydiqa/{model_name}/{language}",
        logging_dir=f"logs/tydiqa/{model_name}/{language}",

        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,
    )

    if model_name == "xlm-r":
        training_args.num_train_epochs = 10
        training_args.learning_rate = 3e-5
        training_args.per_device_train_batch_size = 16
        training_args.per_device_eval_batch_size = 16
        training_args.weight_decay = 0.01
        training_args.warmup_ratio = 0.1
        training_args.fp16 = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_features_no_id,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()

    trainer.save_model(f"models/tydiqa/{model_name}/{language}")
    tokenizer.save_pretrained(f"models/tydiqa/{model_name}/{language}")


def evaluate_tydiqa(model_name: str, language: str):
    results = {
        'task_lang': [],
        'transfer_lang': [],
        'exact_match': [],
        'f1_score': []
    }

    dataset = load_from_disk('data/tydiqa')
    languages = _extract_languages(dataset["validation"])

    if model_name == "xlm-r":
        model = AutoModelForQuestionAnswering.from_pretrained(f'models/tydiqa/{model_name}/{language}')
        tokenizer = AutoTokenizer.from_pretrained(f'models/tydiqa/{model_name}/{language}')

    for task_lang in tqdm(languages):
        eval_dataset = _filter_by_language(dataset["validation"], task_lang)

        def preprocess_validation(examples):
            questions = [q.strip() for q in examples["question"]]

            inputs = tokenizer(
                questions,
                examples["context"],
                max_length=384,
                truncation="only_second",
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            sample_mapping = inputs.pop("overflow_to_sample_mapping")
            offset_mapping = inputs["offset_mapping"]

            example_ids = []

            for i in range(len(inputs["input_ids"])):
                sample_idx = sample_mapping[i]
                example_ids.append(examples["id"][sample_idx])

                sequence_ids = inputs.sequence_ids(i)
                offsets = offset_mapping[i]

                inputs["offset_mapping"][i] = [
                    o if sequence_ids[k] == 1 else None
                    for k, o in enumerate(offsets)
                ]

            inputs["example_id"] = example_ids
            return inputs

        features = eval_dataset.map(
            preprocess_validation,
            batched=True,
            remove_columns=eval_dataset.column_names
        )

        trainer = Trainer(model=model)

        predictions = trainer.predict(features)
        start_logits, end_logits = predictions.predictions

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        final_predictions = {}

        for example in eval_dataset:
            example_id = example["id"]
            feature_indices = example_to_features[example_id]

            best_score = -1e9
            best_answer = ""

            for idx in feature_indices:
                start_logit = start_logits[idx]
                end_logit = end_logits[idx]
                offsets = features[idx]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-20:]
                end_indexes = np.argsort(end_logit)[-20:]

                for start in start_indexes:
                    for end in end_indexes:
                        if offsets[start] is None or offsets[end] is None:
                            continue
                        if end < start or end - start > 30:
                            continue

                        score = start_logit[start] + end_logit[end]

                        if score > best_score:
                            start_char = offsets[start][0]
                            end_char = offsets[end][1]
                            best_answer = example["context"][start_char:end_char]
                            best_score = score

            final_predictions[example_id] = best_answer

        formatted_predictions = [
            {"id": k, "prediction_text": v}
            for k, v in final_predictions.items()
        ]

        references = [
            {"id": ex["id"], "answers": ex["answers"]}
            for ex in eval_dataset
        ]

        metric = evaluate.load("squad")
        metric_scores = metric.compute(
            predictions=formatted_predictions,
            references=references,
        )

        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['exact_match'].append(metric_scores['exact_match'])
        results['f1_score'].append(metric_scores['f1'])

        df_results = pd.DataFrame(results)

        output_dir = f'results/tydiqa/{model_name}'
        os.makedirs(output_dir, exist_ok=True)

        df_results.to_csv(f'{output_dir}/{language}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune a model in a specified language and evaluate cross lingual transfer in the TyDiQA dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in TyDiQA')
    parser.add_argument('--model', type=str, required=True, choices=('xlm-r', 'mt5', 'llama3'),
                        help='Model to fine-tune')
    parser.add_argument('--eval_only', action='store_true',
                        help="Skip training and directly load finetuned model for evaluation")
    parser.add_argument('--resume_step', type=int, default=None,
                        help="Resume training from checkpoint step number")
    args = parser.parse_args()

    print(f"Processing model {args.model} and language {args.lang} on TyDiQA dataset...")

    # Fine-tune model on transfer language
    if not args.eval_only:
        finetune_tydiqa(args.model, args.lang)

    # Evaluate fine-tuned model on all other languages in the dataset
    evaluate_tydiqa(args.model, args.lang)
