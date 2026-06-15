# Ethio-TTS — Amharic TTS efficiency study on WAXAL

How efficient is the WAXAL Ethiopian speech corpus for training a
*text-to-speech* model? This repository answers that question with a
round-trip ASR evaluation, using our re-trained
[Ethio-ASR](https://github.com/boaztulu/Ethio-ASR) as the judge.

See **[REPORT.md](REPORT.md)** for the full write-up. TL;DR:

| Audio | ASR WER |
|---|---|
| Real WAXAL Amharic recordings | 20.92% |
| Zero-shot `facebook/mms-tts-amh` synth (same text) | 21.60% |
| Gap | **+0.68%** |

Zero-shot Amharic TTS is essentially at parity with real recordings on
intelligibility. WAXAL's value for TTS is in speaker diversity (452 voices),
prosody and style — not in raw intelligibility.

## Layout

```
Ethio-TTS/                 # this git repo
├── REPORT.md              # full findings report
├── slurm_scripts/         # data filtering, zero-shot eval, fair-comparison eval
│   ├── filter_waxal_amh.py
│   ├── filter_waxal_amh.sbatch
│   ├── smoke_test_tts.py
│   ├── eval_tts_wer.py
│   ├── eval_tts_wer.sbatch
│   ├── eval_tts_fair.py
│   └── setup_env.sh
├── webapp/                # Gradio web demo
│   ├── app.py
│   ├── serve_webapp.sbatch
│   └── README.md
└── results/               # eval JSON outputs + speaker stats
```

Data, model weights and caches live *outside* this git repo at
`/blue/rcstudents/btulu/Projects/Ethio-TTS/{filtered_audio,hf_cache,...}/`
to keep the repo small.

## Quick start

```bash
module load pytorch/2.8.0
source slurm_scripts/setup_env.sh

# 1. (One-time, ~22 min) Filter WAXAL Amharic by duration + speaker
sbatch slurm_scripts/filter_waxal_amh.sbatch

# 2. Zero-shot synthesis smoke test (5 sentences, runs anywhere)
python3 slurm_scripts/smoke_test_tts.py

# 3. Paired real-vs-synth eval (N=300)
sbatch slurm_scripts/eval_tts_wer.sbatch    # uses --wrap submission

# 4. Live Gradio demo (B200 GPU, gradio.live share URL)
sbatch webapp/serve_webapp.sbatch
```

## Models used

* **TTS:** [`facebook/mms-tts-amh`](https://huggingface.co/facebook/mms-tts-amh)
  — 36 M params, 16 kHz, romanised input via [`uroman`](https://github.com/isi-nlp/uroman).
* **ASR (judge):** [`boazsew/Ethio-ASR-w2v-bert-2.0-uf`](https://huggingface.co/boazsew/Ethio-ASR-w2v-bert-2.0-uf)
  — our reproduction of the Ethio-ASR paper's best model (600 M).

## Dataset

`badrex/waxalNLP-ethiopic-final` (Diack et al., 2026) — Amharic split
filtered to 1–25 s clips and speakers with ≥3 min, leaving:

* **195.8 hours**
* **452 speakers** (top speaker 111 min; top-15 each ≥91 min)
* 36 769 train + ~2 900 validation clips

See `results/speaker_stats.json` for the full distribution.
