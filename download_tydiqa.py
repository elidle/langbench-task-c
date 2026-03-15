from datasets import load_dataset

dataset = load_dataset("tydiqa", "secondary_task")
dataset.save_to_disk(f"data/tydiqa")