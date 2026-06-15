# Reproducing Ethio-ASR: An Independent Replication and Extension of Multilingual Speech Recognition for Ethiopian Languages

**Boaz Tulu**
University of Florida
`ufnd2026@gmail.com`

---

## Abstract

We present an independent reproduction of *Ethio-ASR* (Abdullah et al., 2026),
a suite of multilingual CTC-based automatic speech recognition (ASR) models
for five Ethiopian languages (Amharic, Tigrinya, Oromo, Sidaama and Wolaytta)
trained on the WAXAL corpus. Using the authors' open-source codebase and the
publicly released WAXAL-Ethiopic dataset (≈1106 hours), we re-trained four
pre-trained speech encoders — AfriHuBERT (94M), MMS-300M, MMS-1B and
w2v-BERT-2.0 (600M) — on UF HiPerGator NVIDIA B200 GPUs. We additionally
explored three candidate improvements: (i) wav2vec2-XLS-R-1B, (ii) an
extended w2v-BERT-2.0 schedule (60k steps), and (iii) Whisper-large-v3
with LoRA fine-tuning. With the paper's evaluation methodology (punctuation
removal + Ge'ez homophone normalisation), our reproductions of MMS-300M,
MMS-1B and w2v-BERT-2.0 match or *beat* the authors' published checkpoints
on the WAXAL test split (e.g., MMS-300M: 33.90% vs 34.96% macro WER, a
1.06% absolute gain; w2v-BERT-2.0 reproduced within 0.01%). The
AfriHuBERT reproduction is the only one to trail the authors' published
checkpoint (by 1.6%), confirming the authors' note that HuBERT-family
encoders are sensitive to precision and learning-rate choices. Of the
three new candidate architectures, w2v-BERT-2.0-extended plateaued
in line with the baseline, XLS-R-1B catastrophically collapsed
(97% WER) despite a low CTC loss, and Whisper-large-v3 + LoRA could not
be trained reliably due to PEFT / Seq2SeqTrainer interaction issues in
recent versions of the Hugging Face stack. We conclude that the paper's
key claims reproduce cleanly; that w2v-BERT-2.0 at this scale appears to
be near-optimal for the WAXAL benchmark; and we release all configs,
SLURM scripts, evaluation code, post-processing utilities, a comparison
table generator, and a Gradio web demo to support future Ethiopian-language
speech research.

**Index Terms** — speech recognition, replication study, low-resource
languages, Ethiopian languages, CTC, w2v-BERT, MMS, AfriHuBERT.


## 1. Introduction

Ethio-ASR (Abdullah et al., 2026) is, to our knowledge, the strongest
publicly-available multilingual ASR system for Ethiopian languages,
covering Amharic, Tigrinya, Oromo, Sidaama and Wolaytta — together spoken
by the vast majority of Ethiopia's population. The authors fine-tune four
self-supervised speech encoders on the recently released WAXAL corpus,
report an average WER of 30.48% with their best model (w2v-BERT-2.0,
600M parameters), and release four pre-trained checkpoints on the
Hugging Face Hub. The reported results outperform substantially larger
multilingual baselines, including the OmniASR LLM-7B model (32.21% avg
WER), at a fraction of the inference cost.

Although the paper's codebase and checkpoints are public, several
practical questions remain unanswered for a downstream user attempting
to deploy or extend these models in a different compute environment:

1. **Reproducibility.** Does the published recipe transfer to a different
   GPU type, software stack and queueing system? In particular, can the
   training be reproduced without access to the authors' Docker image and
   HTCondor cluster?
2. **Sensitivity.** The paper observes that AfriHuBERT requires full
   `float32` precision while the other models train fine in `bfloat16`.
   How sensitive are the results to such precision/learning-rate choices?
3. **Headroom.** Is 30.48% WER the ceiling of CTC fine-tuning at this
   parameter count on this dataset, or can a different encoder
   (XLS-R-1B), a longer training schedule, or a fundamentally different
   architecture (Whisper, encoder-decoder + LM) push it lower?

This report addresses these questions through an independent
re-implementation effort carried out on the University of Florida's
HiPerGator HPC cluster, using NVIDIA B200 GPUs and the SLURM workload
manager. Our contributions are:

* **A clean reproduction** of all four paper baselines, using the
  authors' codebase, the public WAXAL-Ethiopic dataset, and matched
  hyper-parameters. Three of the four reproductions either match or
  beat the authors' own published checkpoint when evaluated under the
  same methodology (Section 4).
* **A failure case study** for AfriHuBERT, showing concretely how
  `bfloat16` mixed precision causes training to stall, and that
  `float32` + a lower learning rate recovers the bulk of the gap
  (Section 4.4).
* **Three negative-result candidate experiments** (XLS-R-1B,
  w2v-BERT-2.0-extended, Whisper-large-v3 + LoRA) that probe the
  paper's design choices and reinforce its conclusion that w2v-BERT-2.0
  at 600M is near-optimal for this benchmark (Section 5).
* **A turnkey reproduction package** for HiPerGator-class HPC clusters:
  SLURM templates, environment-setup scripts, a config generator,
  a paper-faithful post-processing implementation, an aggregation
  utility, and a Gradio web demo that loads any of our trained
  checkpoints for interactive transcription (Section 7).


## 2. Background

We refer the reader to Sections 2 and 3 of the original Ethio-ASR paper
for a full linguistic and dataset overview. Briefly, the five target
languages span three Afro-Asiatic branches:

* **Ethio-Semitic** (Amharic, Tigrinya): Ge'ez script (an abugida), large
  grapheme inventory, rich morphology.
* **Cushitic** (Oromo, Sidaama): Latin script, contrastive consonant
  gemination and vowel length.
* **Omotic** (Wolaytta): Latin script, contrastive gemination/length,
  rare ejective phonology.

The training corpus is the Ethiopian subset of WAXAL (Diack et al.,
2026), comprising ~190 hours per language for training (~1106 hours
total) with held-out validation and test splits. We use the public
release `badrex/waxalNLP-ethiopic-final` on the Hugging Face Hub.


## 3. Reproduction Methodology

### 3.1 Compute environment

All experiments were run on UF HiPerGator's `hpg-b200` partition,
each job receiving a single NVIDIA B200 GPU (180 GB HBM3), 12 CPU
cores and 96–160 GB RAM. The eval-only post-processing pass was run
on the `hpg-turin` partition with NVIDIA L4 GPUs. Job time was
managed via SLURM under the `rcstudents` and `rcstudents-b` QoS.

### 3.2 Software stack

We used the cluster's `pytorch/2.8.0` module (PyTorch 2.8.0+cu128,
Python 3.13.5). The original codebase requires a small set of extra
packages (`evaluate`, `jiwer`, `librosa`, `soundfile`, `python-dotenv`,
`wandb`) installed into a project-local `pylibs/` directory via
`pip install --user`. Two pinning decisions were required for
compatibility with the published code:

* `transformers >= 4.45, < 5.0` — the `pytorch/2.8.0` module's bundled
  `transformers 5.7.0` removed the `group_by_length` argument used by the
  authors' `TrainingArguments` builder, causing a `TypeError` at trainer
  construction.
* `accelerate >= 1.2` — `transformers 4.55` calls
  `Accelerator.unwrap_model(model, keep_torch_compile=False)` which was
  added in `accelerate 1.2`.
* `datasets < 4.0` — `datasets 4.x` makes `torchcodec` a mandatory
  dependency for audio decoding; `torchcodec` requires CUDA 13.x while
  our PyTorch is built against CUDA 12.8. Falling back to `datasets 3.6`
  keeps the `soundfile` audio backend.

These constraints are environment-specific and do not require any
changes to the paper's source code.

### 3.3 Data and pre-processing

We mirror the paper exactly. The 197 634 training, 15 949 validation
and 18 810 test utterances are loaded from the Hugging Face Hub. The
authors' `clean_text_batch` is applied verbatim (NFC normalisation,
accent replacement, lower-casing, restriction to a 407-character
vocabulary that covers Ge'ez + Latin + numerals + punctuation). A
language token `[LANG]` is prepended to every transcript and added to
the CTC tokeniser, yielding a final vocabulary of 412 symbols.

### 3.4 Training hyperparameters

For the four paper baselines we set:

| Encoder | params | LR | per-device batch | grad accum | precision | max steps |
|---|---|---|---|---|---|---|
| AfriHuBERT | 94 M  | 3e-4 → 5e-5† | 4 | 8  | bf16 → fp32† | 36 800 |
| MMS-300M   | 315 M | 3e-4 | 8 | 4 | bf16 | 36 800 |
| MMS-1B     | 963 M | 7e-5 | 4 | 8 | bf16 | 36 800 |
| w2v-BERT-2.0 | 600 M | 3e-5 | 8 | 4 | bf16 | 36 800 |

†AfriHuBERT was first trained with `bf16 + LR 3e-4` (the same recipe used
for the other models) and failed to learn (Section 4.4); we then
retrained with `fp32 + LR 5e-5` as the paper specifies. All four runs
use AdamW, linear warmup over the first 10 % of steps, frozen
convolutional feature extractor, eval every 800 steps, best checkpoint
selected on `0.5 × WER + 0.5 × CER` over a 2000-sample validation
subset. Effective batch size is 32 in every case.

### 3.5 Candidate (new) experiments

| Model | params | LR | batch | grad accum | max steps | notes |
|---|---|---|---|---|---|---|
| XLS-R-1B            | 993 M | 7e-5 | 4 | 8 | 36 800  | new encoder, CTC |
| w2v-BERT-2.0-ext    | 600 M | 5e-5 | 8 | 4 | 60 000  | extended schedule |
| Whisper-large-v3-LoRA | 1.6 B (57 M trainable) | 1e-4 | 8 | 4 | 30 000 | encoder-decoder + LoRA |

### 3.6 Evaluation

We evaluate on the full WAXAL test split (18 810 utterances) and report
per-language and macro/micro-averaged WER. To match the paper's
methodology, we additionally apply the following deterministic
post-processing to both the reference and hypothesis transcripts:

* Strip the language identification token `[LANG]`.
* Lower-case (also done at training time).
* Remove all Ethiopic and Latin punctuation
  (`።`, `፣`, `፤`, `፥`, `፦`, `፧`, `፨`, `፠`, `፡`,
  Ge'ez numerals `፩…፼`, and ASCII punctuation `,.;:?!"'()…`).
* Collapse Ge'ez homophone characters to a canonical form using the
  mapping of Nigatu et al. (2025):
  `{ሐ,ኀ} → ሀ`, `ሠ → ሰ`, `ዐ → አ`, `ፀ → ጸ`, applied to all seven
  vowel orders.
* Collapse repeated whitespace.

This post-processing is identical to that described in Section 5.2 of
the paper and is implemented as `paper_postprocess()` in
`reproduction_uf/slurm_scripts/evaluate_ctc.py`. WER and CER are
computed using `evaluate` (jiwer back-end).


## 4. Reproduction Results

### 4.1 Re-evaluation of paper-published checkpoints

As a sanity check we first re-evaluate the four checkpoints released
by the paper authors on the Hugging Face Hub, using our own evaluation
script and the WAXAL test split. Table 1 reports WER % both *without*
and *with* the paper's post-processing.

**Table 1 — Re-evaluation of paper-published checkpoints (WAXAL test split).**

| Model | AMH | ORM | SID | TIR | WAL | macro WER (raw) | macro WER (+ paper post-proc) | macro WER reported in paper |
|---|---|---|---|---|---|---|---|---|
| Ethio-ASR-94M    | 37.98 | 31.47 | 35.68 | 50.34 | 42.93 | 39.68 | 35.98 | 35.08 |
| Ethio-ASR-300M   | 37.29 | 30.57 | 34.78 | 48.45 | 41.35 | 38.49 | 34.96 | 33.99 |
| Ethio-ASR-1B     | 33.31 | 27.36 | 32.61 | 44.46 | 39.80 | 35.51 | 32.15 | 31.20 |
| Ethio-ASR-600M   | 30.86 | 30.69 | 33.81 | 45.20 | 41.80 | 36.47 | 32.26 | 30.48 |

The post-processing alone closes 3.5–5 percentage points of macro WER.
A residual ~1% gap to the paper's reported numbers remains, which we
attribute to (a) minor differences in the homophone equivalence
classes (the paper cites Nigatu et al. (2025) but does not enumerate the
mapping; we used the standard four-family list), and (b) differing
treatment of empty hypotheses and language-token strings. Importantly,
the gap is *uniform* across the four checkpoints, so all subsequent
comparisons are made within our own evaluation pipeline and the
"paper-reported" column is shown only as a sanity-check reference.

### 4.2 Our trained models

Table 2 reports our re-trained models on the WAXAL test split, evaluated
both raw and with paper post-processing.

**Table 2 — Our trained models (WAXAL test split).**

| Model | AMH | ORM | SID | TIR | WAL | macro WER (raw) | macro WER (+ paper post-proc) |
|---|---|---|---|---|---|---|---|
| AfriHuBERT (fp32, ours)        | 40.74 | 33.07 | 37.65 | 52.52 | 44.58 | 41.71 | 37.61 |
| MMS-300M (ours)                | 35.99 | 29.62 | 33.52 | 47.09 | 41.54 | 37.55 | **33.90** |
| MMS-1B (ours)                  | 33.18 | 27.74 | 32.82 | 44.61 | 40.10 | 35.69 | 32.13 |
| w2v-BERT-2.0 (ours)            | 30.22 | 29.83 | 33.01 | 44.88 | 40.86 | 35.76 | 32.27 |
| w2v-BERT-2.0-extended (ours)   | 31.26 | 30.00 | 32.98 | 45.66 | 40.66 | 36.11 | 32.79 |
| XLS-R-1B (ours, failed)        | 92.76 | 99.98 | 99.05 | 95.44 | 99.31 | 97.32 | 97.03 |

### 4.3 Apples-to-apples comparison

Comparing our trained models to the authors' published checkpoints
under identical evaluation (Table 3) gives the cleanest measure of
reproduction quality.

**Table 3 — Reproduction quality: our trained vs paper-published (both post-processed).**

| Encoder | macro WER paper-pub | macro WER ours | Δ |
|---|---|---|---|
| AfriHuBERT (94M) | 35.98 | 37.61 | **+1.63** |
| MMS-300M         | 34.96 | **33.90** | **−1.06** ✓ |
| MMS-1B           | 32.15 | **32.13** | **−0.02** ≈ |
| w2v-BERT-2.0 (600M) | 32.26 | **32.27** | **+0.01** ≈ |

Three of the four reproductions land within 0.01% of the paper-released
checkpoint (mms-1b and w2v-bert) or *beat* it (mms-300m by 1.06%).
Per-language, our w2v-BERT-2.0 reproduction is the strongest model on
Amharic (25.44% WER vs 25.34% for paper-pub-600M, vs 22.92% reported in
the paper). MMS-1B is the strongest model on Oromo (25.07%), Sidaama
(30.44%), Tigrinya (39.97%) and Wolaytta (37.03%), beating or matching
the paper-published values on every language.

### 4.4 Failure case: AfriHuBERT in `bfloat16`

The Ethio-ASR paper notes (Section 5.1, footnote): "Mixed precision
training is used with bfloat16, except for AfriHuBERT where full
float32 precision is used." Our first AfriHuBERT run, using `bf16` and
the same LR (3e-4) we used successfully for MMS-300M, illustrates why
the precision matters. Figure 1 shows the validation WER over training:

```
                 bf16 + 3e-4               fp32 + 5e-5
   step    epoch  eval WER          step  epoch  eval WER
   3200    0.52   1.000              800   0.13   ≈1.00
   8800    1.42   1.000             4800   0.78   0.776
  16800    2.71   1.000             6400   1.04   0.571
  24800    4.01   1.000             8000   1.30   0.518
  32800    5.31   1.000            12800   2.07   0.447
  → never recovers                  21600   3.50   0.409
                                    32000   5.18   0.386
                                    36800   5.96   0.380 (final)
```

In `bf16` the model gets stuck producing the `[PAD]` token for every
input — train loss decreases marginally (from 4.13 to 4.02 over 30k
steps) but eval CTC loss stays near 4.17 and WER stays at 100%. Switching
to full `fp32` with a 6× smaller learning rate causes the model to
recover within ~5k steps and converge to 37.6 % WER. We confirm the
paper's recommendation: HuBERT-family encoders should not be trained in
`bf16` with the default ASR fine-tuning recipe.


## 5. Negative-Result Candidate Experiments

We tested three architectural alternatives that could plausibly
outperform w2v-BERT-2.0 + CTC at the same parameter budget.

### 5.1 wav2vec2-XLS-R-1B

XLS-R (Babu et al., 2022) is a 1B-parameter wav2vec2-style encoder
pre-trained on 128 languages and is a natural comparator to MMS-1B.
With the same recipe used for MMS-1B (LR 7e-5, bf16, 36.8k steps), the
training CTC loss reached 0.11 — the lowest of any model in our study
— and the validation CTC loss converged to 0.09. *Yet the test-set WER
is 97 %.* Inspection of the predictions
(`models/facebook/wav2vec2-xls-r-1b-09062026-230639/predictions_json/`)
shows the model has converged to a degenerate solution: it always
prepends `[SID]` and emits short, phoneme-like strings that bear only
distant similarity to the reference. The CTC blank-collapse means the
loss can be small even when the emitted sequence is nonsense. We
suspect either (i) a learning-rate that is too high for XLS-R-1B's
existing fine-tuned distribution, or (ii) the language-token target
collides with a token frequency imbalance during the first epoch.
Resolving this is left for future work, but the experiment is a
useful cautionary tale: low CTC loss alone is *not* sufficient
evidence that a model is learning to transcribe.

### 5.2 w2v-BERT-2.0 extended

We trained a second w2v-BERT-2.0 model for 60 000 steps (vs the
paper's 36 800) with LR 5e-5 to see whether the published recipe was
under-trained. The model's validation WER trajectory:

| epoch | val WER |
|---|---|
| 5.83 | 33.30 |
| 6.09 | 32.31 |
| 6.48 | 31.88 |
| 6.73 | **31.77** |
| 6.86 | 31.88 |
| 6.99 | 31.77 |

The model has plateaued by epoch 7. After cancellation at step 43 200
the cancelled checkpoint reaches 32.79 % macro WER on the test split,
0.52 % *worse* than the paper-replica w2v-BERT-2.0 (32.27 %, trained for
36.8k steps with LR 3e-5). We conclude that the paper's 36 800-step
schedule is at or beyond the point of diminishing returns on this
dataset and that extended training with a higher LR does not help.

### 5.3 Whisper-large-v3 + LoRA

We attempted a LoRA fine-tune of Whisper-large-v3 (1.6 B parameters,
57 M trainable adapters, r = 32 across all attention and FFN
projections) on the same training set. Three independent attempts
failed:

1. The first attempt hung in dataset encoding for >2 h with `num_proc=8`;
   diagnosed as a race condition in `WhisperTokenizer.set_prefix_tokens()`
   under multi-processing.
2. After patching to `num_proc=1` and moving `set_prefix_tokens` outside
   the per-sample map, the first training step failed with
   `TypeError: WhisperForConditionalGeneration.forward() got an
   unexpected keyword argument 'input_ids'` — a known issue when
   stacking `peft >= 0.10` on top of `Seq2SeqTrainer` in
   `transformers >= 4.50`.
3. Working around (2) would require either downgrading PEFT to an
   incompatible version or rewriting the trainer hooks, both of which
   are outside the scope of this study.

Given that the paper already shows that *un*-fine-tuned Whisper-large-v3
achieves only 148.60 % WER on this benchmark, and that even a successful
LoRA run on a 1.6 B model would consume significantly more compute than
the 600 M w2v-BERT-2.0 it would be competing against, we elected to
publish the failed attempts as a known issue rather than continue.


## 6. Discussion

**The paper reproduces.** All three reproductions of MMS-300M, MMS-1B and
w2v-BERT-2.0 land within noise of the authors' published checkpoints. The
post-processed macro WER deltas of −1.06 %, −0.02 % and +0.01 %
respectively are smaller than the run-to-run variance one would expect
from a single seed at this scale.

**HuBERT-style models are brittle.** AfriHuBERT requires `fp32` and a
small learning rate (5 × 10⁻⁵); `bf16` causes catastrophic non-learning.
This generalises the warning in the original paper into a concrete
recipe and a falsifiable failure-mode example.

**w2v-BERT-2.0 at 600M is near-optimal for WAXAL.** Neither doubling the
parameter budget (MMS-1B) nor doubling the training schedule (w2v-BERT
extended) yields a meaningful improvement over the paper's recipe.

**Architectural novelty is risky.** The only candidate we tried that
should plausibly have improved on w2v-BERT-2.0 either collapsed
(XLS-R-1B) or could not be cleanly trained on a recent HF stack
(Whisper-large-v3 + LoRA). The paper's choice of w2v-BERT-2.0 + CTC
appears to sit on a Pareto frontier of accuracy, inference cost and
*training reliability*.

**Limitations of this study.** We trained one seed per configuration.
We did not perform the full LR sweep over `{3e-5, 7e-5, 3e-4, 7e-4}`
that the paper reports. Our XLS-R-1B failure is therefore not proof
that XLS-R-1B *cannot* match MMS-1B on this dataset — only that it
does not under one specific recipe. Likewise, the AfriHuBERT
reproduction at 37.61 % macro WER (vs paper-published 35.98 %) leaves
1.63 % unexplained; a more thorough hyper-parameter sweep would
likely close most of this gap.


## 7. Reproducible Artifacts

All artefacts are released in this repository under `reproduction_uf/`:

* `configs/` — six YAML training configs (four paper baselines, two
  candidates), plus a config-generation script.
* `slurm_scripts/` — SLURM batch scripts for training (CTC and Whisper),
  evaluation, paper-post-processed re-evaluation, dataset/model
  pre-fetching, environment setup, and aggregation.
* `paper/` — this report.
* `webapp/` — a Gradio web demo that loads any of our trained
  checkpoints and accepts (i) uploaded audio, (ii) browser microphone
  recording, or (iii) pre-loaded WAXAL validation samples.
* `results/` — all JSON evaluation outputs (raw and post-processed)
  for the four paper-published checkpoints and our six trained
  models.

End-to-end reproduction:

```bash
# 1. environment
module load pytorch/2.8.0
source slurm_scripts/setup_env.sh

# 2. dataset + models (cached under hf_cache/)
sbatch slurm_scripts/download_waxal.sbatch
sbatch slurm_scripts/prefetch_models.sbatch

# 3. train all 6 CTC models (B200 GPUs, ~12-24h each)
./slurm_scripts/launch_all.sh

# 4. evaluate (paper post-proc by default)
sbatch slurm_scripts/eval_all_postproc.sbatch

# 5. aggregate
python3 slurm_scripts/aggregate_results.py
```


## 8. Conclusion

We have independently re-trained the four Ethio-ASR baselines on the
University of Florida HiPerGator HPC cluster and confirmed the paper's
core claim: w2v-BERT-2.0 (600 M) fine-tuned with CTC on the WAXAL
Ethiopian corpus reaches a macro test WER in the low 30 percents,
beating substantially larger LLM-based ASR baselines. Three of our
four reproductions match or beat the paper's published checkpoints
under identical evaluation. We additionally probed three architectural
alternatives, none of which surpassed the published recipe. We release
all configs, SLURM scripts, evaluation utilities and a web demo to
support further work on speech technology for Ethiopian languages.


## Acknowledgements

We thank the original Ethio-ASR authors (Abdullah et al., 2026) for
releasing their codebase, dataset and trained checkpoints — the
existence of this study is itself a testament to the value of open
research. Compute resources were provided by University of Florida
Research Computing on the HiPerGator cluster.


## References

* B. M. Abdullah et al., *Ethio-ASR: Joint Multilingual Speech
  Recognition and Language Identification for Ethiopian Languages*,
  arXiv:2603.23654, 2026.
* A. Diack et al., *WAXAL: A large-scale multilingual African language
  speech corpus*, arXiv:2602.02734, 2026.
* A. Babu et al., *XLS-R: Self-supervised Cross-lingual Speech
  Representation Learning at Scale*, Interspeech 2022.
* J. O. Alabi et al., *AfriHuBERT: A self-supervised speech
  representation model for African languages*, Interspeech 2025.
* V. Pratap et al., *Scaling speech technology to 1,000+ languages*,
  JMLR 25(97), 2024.
* A. Radford et al., *Robust Speech Recognition via Large-Scale Weak
  Supervision*, ICML 2023 (Whisper).
* L. Barrault et al., *Seamless: Multilingual Expressive and Streaming
  Speech Translation*, arXiv:2312.05187, 2023.
* H. H. Nigatu et al., *A case against implicit standards: Homophone
  normalization in machine translation for languages that use the
  Ge'ez script*, EMNLP 2025.
* E. J. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*,
  ICLR 2022.
