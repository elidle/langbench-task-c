from datasets import load_dataset

languages = [
    "bam", "bbj", "ewe", "fon", "hau", "ibo", "kin", "lug", "luo", "pcm",
    "mos", "nya", "sna", "swa", "tsn", "twi", "wol", "xho", "yor", "zul",
]

for lang in languages:
    dataset = load_dataset("masakhane/masakhaner2", lang)
    dataset.save_to_disk(f"data/masakhaner/{lang}")