from datasets import load_dataset

languages = ["en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

for lang in languages:
    dataset = load_dataset("xnli", lang)
    dataset.save_to_disk(f"data/xnli/{lang}")