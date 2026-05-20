import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

from argparse import ArgumentParser

import evaluate

from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

ALL_LANGS = [
    'af','am','an','ar','as','az','be','bg','bn','br','bs','ca','cs','cy','da','de',
    'dz','el','eo','es','et','eu','fa','fi','fr','fy','ga','gd','gl','gu','ha','he',
    'hi','hr','hu','hy','id','ig','is','it','ja','ka','kk','km','kn','ko','ku','ky',
    'li','lt','lv','mg','mk','ml','mn','mr','ms','mt','my','nb','ne','nl','nn','no',
    'oc','or','pa','pl','ps','pt','ro','ru','rw','se','sh','si','sk','sl','sq','sr',
    'sv','ta','te','tg','th','tk','tr','tt','ug','uk','ur','uz','vi','wa','xh','yi',
    'yo','zh','zu'
]

AVAILABLE_PAIRS = set([
    'af-en','am-en','an-en','ar-de','ar-en',
    'as-en','az-en','be-en','bg-en','bn-en','br-en','bs-en','ca-en','cs-en',
    'cy-en','da-en','de-en','dz-en','el-en',
    'en-eo','en-es','en-et','en-eu','en-fa','en-fi','en-fr','en-fy','en-ga',
    'en-gd','en-gl','en-gu','en-ha','en-he','en-hi','en-hr','en-hu','en-hy',
    'en-id','en-ig','en-is','en-it','en-ja','en-ka','en-kk','en-km','en-kn',
    'en-ko','en-ku','en-ky','en-li','en-lt','en-lv','en-mg','en-mk','en-ml',
    'en-mn','en-mr','en-ms','en-mt','en-my','en-nb','en-ne','en-nl','en-nn',
    'en-no','en-oc','en-or','en-pa','en-pl','en-ps','en-pt','en-ro','en-ru',
    'en-rw','en-se','en-sh','en-si','en-sk','en-sl','en-sq','en-sr','en-sv',
    'en-ta','en-te','en-tg','en-th','en-tk','en-tr','en-tt','en-ug','en-uk',
    'en-ur','en-uz','en-vi','en-wa','en-xh','en-yi','en-yo','en-zh','en-zu',
])

LANG_NAMES = {
    'af':'Afrikaans','am':'Amharic','an':'Aragonese','ar':'Arabic','as':'Assamese',
    'az':'Azerbaijani','be':'Belarusian','bg':'Bulgarian','bn':'Bengali','br':'Breton',
    'bs':'Bosnian','ca':'Catalan','cs':'Czech','cy':'Welsh','da':'Danish','de':'German',
    'dz':'Dzongkha','el':'Greek','eo':'Esperanto','es':'Spanish','et':'Estonian',
    'eu':'Basque','fa':'Persian','fi':'Finnish','fr':'French','fy':'Frisian',
    'ga':'Irish','gd':'Scottish Gaelic','gl':'Galician','gu':'Gujarati','ha':'Hausa',
    'he':'Hebrew','hi':'Hindi','hr':'Croatian','hu':'Hungarian','hy':'Armenian',
    'id':'Indonesian','ig':'Igbo','is':'Icelandic','it':'Italian','ja':'Japanese',
    'ka':'Georgian','kk':'Kazakh','km':'Khmer','kn':'Kannada','ko':'Korean',
    'ku':'Kurdish','ky':'Kyrgyz','li':'Limburgish','lt':'Lithuanian','lv':'Latvian',
    'mg':'Malagasy','mk':'Macedonian','ml':'Malayalam','mn':'Mongolian','mr':'Marathi',
    'ms':'Malay','mt':'Maltese','my':'Burmese','nb':'Norwegian Bokmål','ne':'Nepali',
    'nl':'Dutch','nn':'Norwegian Nynorsk','no':'Norwegian','oc':'Occitan','or':'Odia',
    'pa':'Punjabi','pl':'Polish','ps':'Pashto','pt':'Portuguese','ro':'Romanian',
    'ru':'Russian','rw':'Kinyarwanda','se':'Northern Sami','sh':'Serbo-Croatian',
    'si':'Sinhala','sk':'Slovak','sl':'Slovenian','sq':'Albanian','sr':'Serbian',
    'sv':'Swedish','ta':'Tamil','te':'Telugu','tg':'Tajik','th':'Thai','tk':'Turkmen',
    'tr':'Turkish','tt':'Tatar','ug':'Uyghur','uk':'Ukrainian','ur':'Urdu',
    'uz':'Uzbek','vi':'Vietnamese','wa':'Walloon','xh':'Xhosa','yi':'Yiddish',
    'yo':'Yoruba','zh':'Chinese','zu':'Zulu','en':'English',
}

TO_TRAIN = {
    'ar', 'de', 'fr', 'ja', 'es', 'zh', 'bg', 'bn', 'bs', 'ca', 'cs', 'da', 'el',
    'et', 'eu', 'fa', 'fi', 'he', 'hr', 'hu', 'id', 'it', 'ko', 'lt', 'lv', 'ms',
    'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sv', 'th', 'tr', 'uk', 'vi'
}

LANG2PAIR = {
    'af':'af-en','am':'am-en','an':'an-en','ar':'ar-en','as':'as-en',
    'az':'az-en','be':'be-en','bg':'bg-en','bn':'bn-en','br':'br-en',
    'bs':'bs-en','ca':'ca-en','cs':'cs-en','cy':'cy-en','da':'da-en',
    'de':'de-en','dz':'dz-en','el':'el-en','eo':'en-eo','es':'en-es',
    'et':'en-et','eu':'en-eu','fa':'en-fa','fi':'en-fi','fr':'en-fr',
    'fy':'en-fy','ga':'en-ga','gd':'en-gd','gl':'en-gl','gu':'en-gu',
    'ha':'en-ha','he':'en-he','hi':'en-hi','hr':'en-hr','hu':'en-hu',
    'hy':'en-hy','id':'en-id','ig':'en-ig','is':'en-is','it':'en-it',
    'ja':'en-ja','ka':'en-ka','kk':'en-kk','km':'en-km','kn':'en-kn',
    'ko':'en-ko','ku':'en-ku','ky':'en-ky','li':'en-li','lt':'en-lt',
    'lv':'en-lv','mg':'en-mg','mk':'en-mk','ml':'en-ml','mn':'en-mn',
    'mr':'en-mr','ms':'en-ms','mt':'en-mt','my':'en-my','nb':'en-nb',
    'ne':'en-ne','nl':'en-nl','nn':'en-nn','no':'en-no','oc':'en-oc',
    'or':'en-or','pa':'en-pa','pl':'en-pl','ps':'en-ps','pt':'en-pt',
    'ro':'en-ro','ru':'en-ru','rw':'en-rw','se':'en-se','sh':'en-sh',
    'si':'en-si','sk':'en-sk','sl':'en-sl','sq':'en-sq','sr':'en-sr',
    'sv':'en-sv','ta':'en-ta','te':'en-te','tg':'en-tg','th':'en-th',
    'tk':'en-tk','tr':'en-tr','tt':'en-tt','ug':'en-ug','uk':'en-uk',
    'ur':'en-ur','uz':'en-uz','vi':'en-vi','wa':'en-wa','xh':'en-xh',
    'yi':'en-yi','yo':'en-yo','zh':'en-zh','zu':'en-zu'
}

MODEL_NAME       = "google/mt5-base"

MAX_INPUT_LEN    = 128
MAX_TARGET_LEN   = 128
BATCH_SIZE       = 64
LR               = 5e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

def finetune_opus(language: str):
    source_ds = load_from_disk(f"data/opus-100/{LANG2PAIR[language]}")

    train_ds = source_ds["train"]
    val_ds   = source_ds["validation"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def make_preprocess(src_lang):
        prefix = f"translate from {LANG_NAMES[src_lang]} to English: "

        def preprocess(examples):
            inputs  = [prefix + ex[src_lang] for ex in examples["translation"]]
            targets = [ex["en"]          for ex in examples["translation"]]

            model_inputs = tokenizer(
                inputs,
                max_length=MAX_INPUT_LEN,
                truncation=True,
                padding=False,
            )
            labels = tokenizer(
                text_target=targets,
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding=False,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return preprocess

    tok_train = train_ds.map(make_preprocess(language), batched=True,
                            remove_columns=train_ds.column_names)
    tok_val   = val_ds.select(range(min(2000, len(val_ds)))).map(make_preprocess(language),   batched=True,
                        remove_columns=val_ds.column_names)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = model.to(device)

    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        int32_max = np.iinfo(np.int32).max
        preds = np.where(
            (preds >= 0) & (preds <= int32_max),
            preds,
            tokenizer.pad_token_id
        ).astype(np.int32)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id).astype(np.int32)

        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": round(result["score"], 4)}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"models/opus-100/{language}",

        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        metric_for_best_model="bleu",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,

        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,

        bf16=True,
        bf16_full_eval=True,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    def make_prefix(src_lang):
        src_name = LANG_NAMES.get(src_lang, src_lang)
        return f"translate from {src_name} to English: "
    
    model = trainer.model

    results = []

    TRAIN_ONLY_LANGS = {'an', 'dz', 'hy', 'mn', 'yo'}

    for lang in tqdm(ALL_LANGS, desc="Languages"):
        split = "train" if lang in TRAIN_ONLY_LANGS else "test"
        ds = load_from_disk(f"data/opus-100/{LANG2PAIR[lang]}")[split]

        raw_refs = [ex["translation"]["en"] for ex in ds]
        prefix   = make_prefix(lang)

        def preprocess(examples):
            inputs = [prefix + ex[lang] for ex in examples["translation"]]
            return tokenizer(inputs, max_length=MAX_INPUT_LEN, truncation=True, padding=False)

        tok = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
        tok.set_format("torch")

        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        loader   = DataLoader(tok, batch_size=BATCH_SIZE, collate_fn=collator)

        all_preds = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                generated = model.generate(
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    max_new_tokens = MAX_TARGET_LEN,
                    num_beams      = 1,
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_preds.extend([p.strip() for p in decoded])

        bleu = sacrebleu.compute(
            predictions = all_preds,
            references  = [[r] for r in raw_refs],
        )["score"]

        results.append({
            "task_lang" : lang,
            "transfer_lang": language,
            "bleu": bleu,
        })

        df = pd.DataFrame(results).sort_values("bleu", ascending=False)
        df.to_csv("results/opus-100", index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune a model in a speficied language and evaluate cross lingual transfer in the Masakha NER 2.0 dataset.")

    parser.add_argument('--lang', type=str, required=True,
                        help='Language to fine-tune the model on. Must follow the convention in Masakha NER 2.0.')
    args = parser.parse_args()

    print(f"Processing mt5 and language {args.lang} on opus dataset...")

    finetune_opus(args.lang)
