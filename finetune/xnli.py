from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate
from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk, DatasetDict
import pandas as pd
import os
from tqdm import tqdm
import torch

LLAMA3_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

LLAMA3_LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

def flatten_to_lang(batch, lang, model):
    if model == "xlm-r":
        premises = [p[lang] for p in batch["premise"]]
        hypotheses = []
        for h in batch["hypothesis"]:
            idx = h["language"].index(lang)
            hypotheses.append(h["translation"][idx])

        return {
            "premise": premises,
            "hypothesis": hypotheses,
            "label": batch["label"]
        }

    elif model == "mt5":
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

        inputs = [
            f"premise: {p[lang]}\n"
            f"hypothesis: {h['translation'][h['language'].index(lang)]}\n"
            f"label: entailment, neutral, or contradiction?"
            for p, h in zip(batch["premise"], batch["hypothesis"])
        ]
        targets = [label_map[label] for label in batch["label"]]

        return {
            "input_text": inputs, 
            "target_text": targets
        }

    elif model == "llama3":
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

        premises = [p[lang] for p in batch["premise"]]
        hypotheses = [
            h["translation"][h["language"].index(lang)]
            for h in batch["hypothesis"]
        ]

        prompts = [
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Premise: {p}\n"
            f"Hypothesis: {h}\n"
            f"Does the premise entail, contradict, or is it neutral toward the hypothesis? "
            f"Answer with one word: entailment, neutral, or contradiction."
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            for p, h in zip(premises, hypotheses)
        ]
        targets = [label_map[label] for label in batch["label"]]

        return {
            "prompt": prompts,
            "target_text": targets,
            "label": batch["label"],
        }

def finetune_xnli(model_name: str, language: str, resume_step: int = None):
    label_list = ['entailment', 'neutral', 'contradiction']
    label_to_id = {l: i for i, l in enumerate(label_list)}

    dataset = load_dataset("xnli", "all_languages")

    train_ds = dataset["train"].map(
        lambda x: flatten_to_lang(x, language, model_name),
        batched=True,
        remove_columns=["premise", "hypothesis"]
    )

    val_ds = dataset["validation"].map(
        lambda x: flatten_to_lang(x, language, model_name),
        batched=True,
        remove_columns=["premise", "hypothesis"]
    )

    del dataset

    if model_name == "xlm-r":
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=len(label_list))
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        def tokenize_fn(examples):
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=128)
        
        tokenized_train = train_ds.map(tokenize_fn, batched=True)
        tokenized_val = val_ds.map(tokenize_fn, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    elif model_name == "mt5":
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

        def tokenize_fn(examples):
            model_inputs = tokenizer(
                examples["input_text"],
                text_target=examples["target_text"],
                truncation=True,
                max_length=128,
                max_target_length=10
            )

            return model_inputs
        
        tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["label"])
        tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=["label"])

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    elif model_name == "llama3":
        tokenizer = AutoTokenizer.from_pretrained(LLAMA3_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            LLAMA3_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        base_model.config.use_cache = False
        model = get_peft_model(base_model, LLAMA3_LORA_CONFIG)
        model.print_trainable_parameters()

        def tokenize_fn(examples):
            full_texts = [p + t for p, t in zip(examples["prompt"], examples["target_text"])]
            tokenized = tokenizer(full_texts, truncation=True, max_length=256, padding=False)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_train = train_ds.map(tokenize_fn, batched=True)
        tokenized_val = val_ds.map(tokenize_fn, batched=True)

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if model_name == "xlm-r":
            predictions = predictions.argmax(axis=-1)
        
        elif model_name == "mt5":
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

            labels = [[l for l in label if l != -100] for label in labels]
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions = [label_to_id.get(p.strip(), -1) for p in decoded_preds]
            labels = [label_to_id.get(l.strip(), -1) for l in decoded_labels]

        elif model_name == "llama3":
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

            predictions = [label_to_id.get(p.strip().split()[-1].lower(), -1) for p in decoded_preds]
            labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)

        acc = accuracy.compute(predictions=predictions, references=labels)
        f1_score = f1.compute(predictions=predictions, references=labels, average='macro')
        return {**acc, **f1_score}

    training_args = TrainingArguments(
        output_dir=f"models/xnli/{model_name}/{language}",
        logging_dir=f"logs/xnli/{model_name}/{language}",
        eval_strategy="steps",
        save_strategy="steps",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=5
    )

    if model_name == "xlm-r":
        training_args.num_train_epochs = 5
        training_args.learning_rate = 1e-5
        training_args.per_device_train_batch_size = 16
        training_args.per_device_eval_batch_size  = 16

        training_args.weight_decay = 0.01
        training_args.warmup_ratio = 0.1

        training_args.eval_steps = 5000
        training_args.save_steps = 5000

        training_args.fp16 = True

    elif model_name == "mt5":
        training_args.num_train_epochs = 3
        training_args.learning_rate = 1e-5
        training_args.per_device_train_batch_size = 8
        training_args.per_device_eval_batch_size  = 8
        training_args.gradient_accumulation_steps = 4
        training_args.weight_decay = 0.01
        training_args.warmup_ratio = 0.1
        training_args.predict_with_generate = True

        training_args.eval_steps = 5000
        training_args.save_steps = 5000

        training_args.bf16 = True

    elif model_name == "llama3":
        training_args.num_train_epochs = 3
        training_args.learning_rate = 2e-4
        training_args.per_device_train_batch_size = 8
        training_args.per_device_eval_batch_size = 8
        training_args.gradient_accumulation_steps = 4
        training_args.weight_decay = 0.01
        training_args.warmup_ratio = 0.1
        training_args.predict_with_generate = True
        
        training_args.eval_steps = 5000
        training_args.save_steps = 5000

        training_args.bf16 = True
        training_args.gradient_checkpointing = True
        training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    resume_checkpoint = None

    if resume_step is not None:
        resume_checkpoint = f"models/xnli/{model_name}/{language}/checkpoint-{resume_step}"

        if not os.path.exists(resume_checkpoint):
            raise ValueError(f"Checkpoint path does not exist: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if model_name == "llama3":
        model.save_pretrained(f"models/xnli/{model_name}/{language}")
    else:
        trainer.save_model(f"models/xnli/{model_name}/{language}")

    tokenizer.save_pretrained(f"models/xnli/{model_name}/{language}")


def evaluate_xnli(model_name: str, language: str):
    label_list = ['entailment', 'neutral', 'contradiction']
    label_to_id = {l: i for i, l in enumerate(label_list)}

    results = {
        'task_lang': [],
        'transfer_lang': [],
        'accuracy': [],
        'f1_score': []
    }

    languages = ["en", "fr", "es", "de", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"]

    if model_name == "xlm-r":
        model = AutoModelForSequenceClassification.from_pretrained(f'models/xnli/{model_name}/{language}')
        tokenizer = AutoTokenizer.from_pretrained(f'models/xnli/{model_name}/{language}')

        def tokenize_fn(examples):
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=128)
        
    elif model_name == "mt5":
        model = AutoModelForSeq2SeqLM.from_pretrained(f'models/xnli/{model_name}/{language}')
        tokenizer = AutoTokenizer.from_pretrained(f'models/xnli/{model_name}/{language}')

        def tokenize_fn(examples):
            model_inputs = tokenizer(
                examples["input_text"],
                text_target=examples["target_text"],
                truncation=True,
                max_length=128,
                max_target_length=10
            )
            return model_inputs
    
    elif model_name == "llama3":
        tokenizer = AutoTokenizer.from_pretrained(f'models/xnli/{model_name}/{language}')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            LLAMA3_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, f'models/xnli/{model_name}/{language}')
        model.eval()

        def tokenize_fn(examples):
            full_texts = [p + t for p, t in zip(examples["prompt"], examples["target_text"])]
            tokenized = tokenizer(full_texts, truncation=True, max_length=256, padding=False)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

    dataset = load_dataset("xnli", "all_languages")

    for task_lang in tqdm(languages):
        if model_name == "xlm-r":
            test_ds = dataset["test"].map(
                lambda x: flatten_to_lang(x, task_lang, model_name),
                batched=True,
                remove_columns=["premise", "hypothesis"]
            ).map(tokenize_fn, batched=True)

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        else:
            test_ds = dataset["test"].map(
                lambda x: flatten_to_lang(x, task_lang, model_name),
                batched=True,
                remove_columns=["premise", "hypothesis", "label"]
            ).map(tokenize_fn, batched=True)

            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")

        trainer = Trainer(model=model, data_collator=data_collator)

        predictions_output = trainer.predict(test_ds)
        predictions = predictions_output.predictions
        labels = predictions_output.label_ids

        if model_name == "xlm-r":
            predictions = predictions.argmax(axis=-1)

        elif model_name == "mt5":
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

            labels = [[l for l in label if l != -100] for label in labels]
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions = [label_to_id.get(p.strip(), -1) for p in decoded_preds]
            labels = [label_to_id.get(l.strip(), -1) for l in decoded_labels]

        elif model_name == "llama3":
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            predictions = [label_to_id.get(p.strip().split()[-1].lower(), -1) for p in decoded_preds]
            labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)

        accuracy = accuracy.compute(predictions=predictions, references=labels)
        f1_score = f1.compute(predictions=predictions, references=labels, average='macro')

        results['task_lang'].append(task_lang)
        results['transfer_lang'].append(language)
        results['accuracy'].append(accuracy['accuracy'])
        results['f1_score'].append(f1_score['f1'])

        df_results = pd.DataFrame(results)

        output_dir = f'results/xnli/{model_name}'
        os.makedirs(output_dir, exist_ok=True)

        df_results.to_csv(f'{output_dir}/{language}.csv', index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune a model in a speficied language and evaluate cross lingual transfer in the XNLI dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in XNLI.')
    parser.add_argument('--model', type=str, required=True, choices=('xlm-r', 'mt5', 'llama3'),
                        help='Model to fine-tune')
    parser.add_argument('--eval_only', action='store_true',
                        help="Skip training and directly load finetuned model for evaluation")
    parser.add_argument('--resume_step', type=int, default=None,
                        help="Resume training from checkpoint step number")
    args = parser.parse_args()

    print(f"Processing model {args.model} and language {args.lang} on XNLI dataset...")

    # Fine-tune model on transfer language
    if not args.eval_only:
        finetune_xnli(args.model, args.lang)

    # Evaluate fine-tuned model on all other languages in the dataset
    evaluate_xnli(args.model, args.lang)
