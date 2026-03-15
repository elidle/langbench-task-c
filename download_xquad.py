from datasets import load_dataset

languages = [
    "ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi" ,"zh"
]

for lang in languages:
    dataset = load_dataset("xquad", f"xquad.{lang}", split="validation")
    dataset.save_to_disk(f"data/xquad/{lang}")