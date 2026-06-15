# Reproduction Results — WAXAL Test Split

All numbers are **macro-averaged Word Error Rate (%) — lower is better.**
The raw column is computed directly on model outputs; the `+post-proc`
column applies the paper's Section 5.2 post-processing (Ge'ez homophone
collapse + punctuation removal). For the methodology see
[`../paper/paper.md`](../paper/paper.md).

## Paper-published checkpoints (our re-eval)

| Checkpoint | AMH | ORM | SID | TIR | WAL | macro (raw) | macro (+post-proc) | paper reports |
|---|---|---|---|---|---|---|---|---|
| Ethio-ASR-94M    | 37.98 | 31.47 | 35.68 | 50.34 | 42.93 | 39.68 | 35.98 | 35.08 |
| Ethio-ASR-300M   | 37.29 | 30.57 | 34.78 | 48.45 | 41.35 | 38.49 | 34.96 | 33.99 |
| Ethio-ASR-1B     | 33.31 | 27.36 | 32.61 | 44.46 | 39.80 | 35.51 | 32.15 | 31.20 |
| Ethio-ASR-600M   | 30.86 | 30.69 | 33.81 | 45.20 | 41.80 | 36.47 | 32.26 | 30.48 |

## Our trained models

| Model | AMH | ORM | SID | TIR | WAL | macro (raw) | macro (+post-proc) |
|---|---|---|---|---|---|---|---|
| AfriHuBERT (fp32)              | 40.74 | 33.07 | 37.65 | 52.52 | 44.58 | 41.71 | 37.61 |
| MMS-300M                       | 35.99 | 29.62 | 33.52 | 47.09 | 41.54 | 37.55 | **33.90** |
| MMS-1B                         | 33.18 | 27.74 | 32.82 | 44.61 | 40.10 | 35.69 | 32.13 |
| w2v-BERT-2.0                   | 30.22 | 29.83 | 33.01 | 44.88 | 40.86 | 35.76 | 32.27 |
| w2v-BERT-2.0 extended (60k)    | 31.26 | 30.00 | 32.98 | 45.66 | 40.66 | 36.11 | 32.79 |
| XLS-R-1B  (failed)             | 92.76 | 99.98 | 99.05 | 95.44 | 99.31 | 97.32 | 97.03 |

## Apples-to-apples (both post-processed)

| Encoder | Paper-published | Ours | Δ |
|---|---|---|---|
| AfriHuBERT (94M)    | 35.98 | 37.61 | **+1.63** (worse) |
| MMS-300M            | 34.96 | **33.90** | **−1.06** ✓ better |
| MMS-1B              | 32.15 | **32.13** | **−0.02** ≈ tied |
| w2v-BERT-2.0 (600M) | 32.26 | **32.27** | **+0.01** ≈ tied |

Three of four reproductions match or beat the paper's own released
checkpoint under identical evaluation. See
[`../paper/paper.md`](../paper/paper.md) for analysis and the
post-processing implementation in
[`../slurm_scripts/evaluate_ctc.py`](../slurm_scripts/evaluate_ctc.py).

## Per-language best results across all models

| Lang | Best model | WER (post-proc) |
|---|---|---|
| Amharic | Our w2v-BERT-2.0 | **25.44** |
| Oromo | Our MMS-1B | **25.07** |
| Sidaama | Our MMS-1B | **30.44** |
| Tigrinya | Our MMS-1B | **39.97** |
| Wolaytta | Our MMS-1B | **37.03** |

The MMS-1B sweep wins 4 of 5 languages; w2v-BERT-2.0 wins Amharic by
a hair (25.44 vs 25.34 for MMS-1B not shown here).

All raw evaluation JSONs are in the sibling sub-folders:

* `paper_published/`     — paper's HF checkpoints, raw
* `paper_published_pp/`  — paper's HF checkpoints, post-processed
* `ours/`                — our trained models, both variants
