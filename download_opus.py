from datasets import load_dataset

AVAILABLE_PAIRS = set([
    'af-en','am-en','an-en','ar-en',
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

for pair in AVAILABLE_PAIRS:
    dataset = load_dataset("Helsinki-NLP/opus-100", pair)
    dataset.save_to_disk(f"data/opus-100/{pair}")