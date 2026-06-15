# How efficient is WAXAL for Amharic text-to-speech?

**Boaz Tulu — University of Florida**
*June 2026*

## Phase 1 follow-up: 4-voice multilingual TTS

The original question was *intelligibility-efficiency*. A natural follow-up
is **voice variety** — can we get more than one voice from WAXAL? Yes:
we built a pipeline that takes `text → MMS-TTS-{lang} → OpenVoice v2 tone
color converter (target voice)`. Target voices were auto-selected from
WAXAL speakers using an F0-based age proxy, producing 4 profiles per
language (young M/F, old M/F) across **Amharic, Tigrinya, Oromo, Sidaama**.
A live Gradio demo with a voice radio button is in
`tts_extension/webapp/app.py`; SLURM launcher in
`tts_extension/webapp/serve_webapp.sbatch`.


## TL;DR

WAXAL Amharic contains **195.8 hours of audio across 452 speakers**, with rich
prosodic and speaker diversity. We tested the data-efficiency question by
**round-tripping**: synthesize Amharic from text with a pretrained TTS model,
then transcribe the synthesis with our own WAXAL-trained ASR model and compute
WER against the original text. On a held-out evaluation set:

| Source of audio | ASR WER | ASR CER |
|---|---|---|
| Real WAXAL Amharic recordings | 20.92% | 5.16% |
| Zero-shot `facebook/mms-tts-amh` synthesis of the **same sentences** | 21.60% | 5.84% |
| **Intelligibility gap** | **+0.68%** | **+0.68%** |

*(N=300 sentences, 2–25 s, identical text for both rows)*

**Zero-shot Amharic TTS is already at near-parity with real recordings on the
intelligibility metric our ASR cares about.** WAXAL's value for TTS therefore
is *not* in pushing intelligibility lower; it is in giving you **speaker
diversity, expressive style and prosody variation** that the pretrained model
(trained on Bible recordings) lacks.

## What WAXAL gives you for Amharic TTS

| Aspect | WAXAL Amharic | Comment |
|---|---|---|
| Total hours (after 1–25 s clip filter, ≥3 min per speaker) | **195.8 h** | More than LJSpeech (24 h), AISHELL-3 (85 h); close to LibriTTS-clean-100 |
| Distinct speakers (after filter) | **452** | Excellent for multi-speaker / voice-cloning TTS |
| Top-15 speakers, hours each | 1.52–1.85 h | Borderline for single-speaker studio TTS (5–10 h preferred) |
| Sample rate | 16 kHz mono | Caps perceived "sparkle"; modern TTS prefers 22–48 kHz |
| Recording | Crowdsourced + validated, not studio | Variable background noise |
| Average Amharic clip length | ~17 s | Outside the comfort zone of standard VITS/Tacotron alignment (≤12 s preferred) |
| Speech styles | scripted + spontaneous + expressive + interrogative + emphatic | Unusually rich for any African-language corpus |
| Licence | CC-BY-SA-4.0 | Open, with attribution |

## Round-trip evaluation details

We use `boazsew/Ethio-ASR-w2v-bert-2.0-uf` (our re-trained 600 M w2v-BERT-2.0
ASR from the Ethio-ASR reproduction) to grade synthesis quality.

* **Input text** comes from the WAXAL **validation** split (so it was unseen
  by both the TTS and the ASR).
* For each sentence we (a) run the real audio through the ASR to get a
  WER lower bound, and (b) synthesise the text and run that through the same
  ASR.
* Both transcripts go through the paper's eval-time post-processing
  (Ge'ez homophone normalisation + punctuation removal) before WER is
  computed.

This procedure was chosen because subjective MOS evaluation is expensive
and ASR-based intelligibility metrics correlate well with human judgement
for diagnostic comparisons.

## Why the gap is so small

The MMS-TTS-Amharic model (Pratap et al., MMS, JMLR 2024) is a 36 M-parameter
single-speaker VITS trained on read Amharic Bible recordings, romanised with
uroman. It produces clean, narrow-band (16 kHz) speech with consistent prosody
and very predictable timing. Our ASR was trained on noisy crowd-sourced
WAXAL Amharic; the synthesised speech is *easier* to transcribe than the
training distribution because it has no background noise, no laughter, no
disfluencies and only one accent.

In other words, **on intelligibility, zero-shot is already at the ceiling
that an ASR trained on noisy field data can measure.** A WAXAL-fine-tuned
TTS would have to match the *distribution* of WAXAL audio (more variation,
field-recording acoustics) to score *better* — and that's not necessarily
what you want from a TTS.

## What WAXAL fine-tuning *would* improve

A TTS fine-tuned on WAXAL would deliver gains that the round-trip WER
metric cannot see:

1. **Voice variety.** 452 different voices versus the single Bible reader
   in MMS-TTS-Amh. Useful for any product where one voice gets boring,
   for audiobook narration, or for inclusive voice assistants.
2. **Speaking style.** WAXAL has expressive, emphatic and interrogative
   prosody. MMS-TTS-Amh has flat declarative prosody only.
3. **Code-switched and conversational Amharic.** WAXAL includes
   spontaneous-description utterances that mirror real Amharic
   conversational patterns.
4. **Field-realistic audio.** If you want TTS that *sounds like* a real
   crowd-recorded voice (e.g., for data-augmentation use), the WAXAL
   acoustic profile is what you want.

## Practical recommendations

* If the goal is **"Amharic TTS that just works"** — use
  `facebook/mms-tts-amh` straight off the shelf with `uroman`. No
  training needed. Our Ethio-TTS demo runs it on a HiPerGator B200 GPU
  through Gradio and shows live round-trip ASR scores.
* If the goal is **multi-speaker Amharic TTS** — WAXAL is an unusually
  good fit. Use `ylacombe/finetune-hf-vits` or a from-scratch VITS with
  a speaker embedding head. The 452-speaker × ~12-min-per-speaker
  budget puts you comfortably in the AISHELL-3 / VCTK regime.
* If the goal is **studio-quality single-voice TTS** — neither WAXAL nor
  MMS-TTS-Amh will get you there alone. Pair WAXAL data of the top 1–3
  speakers with audio super-resolution (NU-Wave 2 or AudioSR) to
  upsample 16→24 kHz, then fine-tune a phoneme-input VITS or
  StyleTTS-2. Expect to top out at "good consumer-grade", not
  ElevenLabs-grade, without studio re-recordings.

## Repro artefacts

* `slurm_scripts/filter_waxal_amh.py` — produces `filtered_audio/` (22 GB)
  + `metadata_{train,val}.csv` + `speaker_stats.json` (the numbers in
  Table 1 of this report).
* `slurm_scripts/smoke_test_tts.py` — zero-shot synthesis demo.
* `slurm_scripts/eval_tts_fair.py` — paired real-vs-synth ASR-based
  evaluation script.
* `webapp/app.py` — Gradio demo that synthesises any Amharic input and
  runs the round-trip ASR live.
* `webapp/serve_webapp.sbatch` — SLURM script that launches the demo on
  hpg-b200 and exposes a `gradio.live` public URL.

All numbers in this report are deterministic with seed 42.
