## Hyperparameters

### TAXI1500

| Hyperparameter | XLM-R |
|---|---|
| Learning Rate | 2e-5 |
| Train Batch Size (per device) | 16 |
| Eval Batch Size (per device) | 16 |
| Epochs | 20 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Eval Steps | 20 |
| Best Model Metric | eval_f1 |
| Early Stopping Patience | 5 |

---

### SIB200

| Hyperparameter | XLM-R |
|---|---|
| Learning Rate | 1e-5 |
| Train Batch Size (per device) | 16 |
| Eval Batch Size (per device) | 16 |
| Epochs | 10 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Eval Steps | 20 |
| Best Model Metric | eval_f1 |
| Early Stopping Patience | 5 |

---

### XNLI

| Hyperparameter | XLM-R | mT5 | LLaMA-3 |
|---|---|---|---|
| Learning Rate | 1e-5 | 1e-5 | 2e-4 |
| Train Batch Size (per device) | 16 | 8 | 8 |
| Eval Batch Size (per device) | 16 | 8 | 8 |
| Epochs | 5 | 3 | 3 |
| Weight Decay | 0.01 | 0.01 | 0.01 |
| Warmup Ratio | 0.1 | 0.1 | 0.1 |
| Gradient Accumulation Steps | — | 4 | 4 |
| Eval / Save Steps | 5000 | 5000 | 5000 |
| Precision | FP16 | BF16 | BF16 |
| Gradient Checkpointing | — | — | ✓ |
| Predict with Generate | — | ✓ | ✓ |
| Early Stopping Patience | 5 | 5 | 5 |

---

### WikiANN

| Hyperparameter | XLM-R |
|---|---|
| Learning Rate | 2e-5 |
| Train Batch Size (per device) | 16 |
| Eval Batch Size (per device) | 16 |
| Epochs | 5 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Eval / Save Steps | 300 |
| Precision | FP16 |
| Early Stopping Patience | 5 |

---

### MLQA (XLM-R)

| Hyperparameter | Value |
|---|---|
| Learning Rate | 3e-5 |
| Train Batch Size (per device) | 16 |
| Eval Batch Size (per device) | 16 |
| Epochs | 2 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Precision | FP16 |