from datasets import load_dataset

languages = ["en", "de", "es", "ar", "hi", "vi", "zh"]

for lang in languages:
    dataset = load_dataset("facebook/mlqa", f"mlqa.{lang}.{lang}")
    dataset.save_to_disk(f"data/mlqa/{lang}")