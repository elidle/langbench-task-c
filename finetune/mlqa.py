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


def finetune_mlqa(model_name: str, language: str, resume_step: int = None):
    dataset = load_from_disk(f'data/mlqa/{language}')

    if model_name == "xlm-r":
        model = AutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

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

                # Find context start and end
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx

                while sequence_ids[idx] == 1:
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
    
    tokenized_dataset = dataset["test"].map(
        preprocess,
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    training_args = TrainingArguments(
        output_dir=f"models/mlqa/{model_name}/{language}",
        logging_dir=f"logs/mlqa/{model_name}/{language}",
        save_strategy="epoch",
        save_total_limit=5
    )

    if model_name == "xlm-r":
        training_args.num_train_epochs = 2
        training_args.learning_rate = 3e-5
        training_args.per_device_train_batch_size = 16
        training_args.per_device_eval_batch_size  = 16

        training_args.weight_decay = 0.01
        training_args.warmup_ratio = 0.1

        # training_args.eval_steps = 40
        # training_args.save_steps = 40

        training_args.fp16 = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()

    tokenizer.save_pretrained(f"models/mlqa/{model_name}/{language}")


def evaluate_mlqa(model_name: str, language: str):
    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    languages = set()

    for folder in os.listdir('data/mlqa'):
        languages.add(folder)

    if model_name == "xlm-r":
        model = AutoModelForQuestionAnswering.from_pretrained(f'models/mlqa/{model_name}/{language}')
        tokenizer = AutoTokenizer.from_pretrained(f'models/mlqa/{model_name}/{language}')

    for task_lang in tqdm(languages):
        dataset = load_from_disk(f'data/mlqa/{task_lang}')

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

                # Set non-context offsets to None
                inputs["offset_mapping"][i] = [
                    o if sequence_ids[k] == 1 else None
                    for k, o in enumerate(offsets)
                ]

            inputs["example_id"] = example_ids
            return inputs
        
        features = dataset["validation"].map(
            preprocess_validation,
            batched=True,
            remove_columns=dataset["validation"].column_names
        )

        trainer = Trainer(model=model)

        predictions = trainer.predict(features)
        start_logits, end_logits = predictions.predictions

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        final_predictions = {}

        for example in dataset:
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
            for ex in raw_dataset
        ]

        metric = evaluate.load("squad")
        results = metric.compute(
            predictions=formatted_predictions,
            references=references,
        )

        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['accuracy'].append(metrics['overall_accuracy'])
        results['f1_score'].append(macro_f1)

        df_results = pd.DataFrame(results)

        output_dir = f'results/mlqa/{model_name}'
        os.makedirs(output_dir, exist_ok=True)

        df_results.to_csv(f'{output_dir}/{language}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune a model in a speficied language and evaluate cross lingual transfer in the MLQA dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in MLQA.')
    parser.add_argument('--model', type=str, required=True, choices=('xlm-r', 'mt5', 'llama3'),
                        help='Model to fine-tune')
    parser.add_argument('--eval_only', action='store_true',
                        help="Skip training and directly load finetuned model for evaluation")
    parser.add_argument('--resume_step', type=int, default=None,
                        help="Resume training from checkpoint step number")
    args = parser.parse_args()

    print(f"Processing model {args.model} and language {args.lang} on MLQA dataset...")

    # Fine-tune model on transfer language
    if not args.eval_only:
        finetune_mlqa(args.model, args.lang)

    # Evaluate fine-tuned model on all other languages in the dataset
    evaluate_mlqa(args.model, args.lang)
