#!/usr/bin/env python3
"""Gradio TTS demo for Ethio-TTS (Amharic).

Inputs:
  - Amharic text (Ge'ez script) → auto-romanized with uroman
  - OR pre-loaded example sentences

Output:
  - Synthesized audio (16 kHz mono, from MMS-TTS-Amh)
  - Round-trip ASR transcription (using our Ethio-ASR model) — shows
    whether the TTS output is intelligible to a model trained on
    real Amharic speech.

Usage:
  python app.py [--tts_model facebook/mms-tts-amh]
                [--asr_model boazsew/Ethio-ASR-w2v-bert-2.0-uf]
                [--port 7861] [--share]
"""
import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "hf_cache" / "hub"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(PROJECT_ROOT / "tmp_cache" / "gradio"))
for k in ("HF_HOME", "HF_HUB_CACHE", "GRADIO_TEMP_DIR"):
    Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

# Reuse the ASR project's post-processing
sys.path.insert(0, "/blue/rcstudents/btulu/Projects/Ethio-ASR/slurm_scripts")
from evaluate_ctc import paper_postprocess, strip_lang_token  # type: ignore

import numpy as np
import torch
import gradio as gr
from transformers import VitsModel, AutoTokenizer, AutoProcessor, AutoModelForCTC
from uroman import Uroman


EXAMPLES = [
    "ሰላም ለዓለም",
    "አማርኛ ቋንቋ ነው",
    "የሰው ልጆች ሁሉ ነጻ ሆነው ይወለዳሉ።",
    "ኢትዮጵያ ቆንጆ አገር ናት",
    "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ነች",
    "እንኳን ደህና መጣችሁ።",
    "ስለ ድጋፍ እናመሰግናለን።",
    "ዛሬ የአየር ሁኔታ ጥሩ ነው።",
]


def load_models(tts_id: str, asr_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[app] loading TTS {tts_id}...", flush=True)
    tts = VitsModel.from_pretrained(tts_id).to(device).eval()
    tts_tok = AutoTokenizer.from_pretrained(tts_id)
    print(f"[app] TTS sr={tts.config.sampling_rate}", flush=True)

    print(f"[app] loading ASR {asr_id}...", flush=True)
    asr_proc = AutoProcessor.from_pretrained(asr_id)
    asr_model = AutoModelForCTC.from_pretrained(asr_id, torch_dtype=dtype).to(device).eval()
    print(f"[app] loaded on {device}", flush=True)
    return {
        "tts": tts, "tts_tok": tts_tok,
        "asr_proc": asr_proc, "asr_model": asr_model,
        "device": device, "dtype": dtype,
        "uroman": Uroman(),
    }


def synthesize(state, text):
    if not text or not text.strip():
        return None, "—", "—", "—"
    rom = state["uroman"].romanize_string(text.strip())
    t0 = time.time()
    inputs = state["tts_tok"](rom, return_tensors="pt").to(state["device"])
    with torch.no_grad():
        wav = state["tts"](**inputs).waveform
    wav = wav.cpu().squeeze().float().numpy()
    sr = state["tts"].config.sampling_rate
    synth_time = time.time() - t0

    # Round-trip ASR
    t0 = time.time()
    asr_inputs = state["asr_proc"](wav, sampling_rate=sr, return_tensors="pt", padding=True)
    in_kwargs = {}
    if "input_features" in asr_inputs:
        in_kwargs["input_features"] = asr_inputs["input_features"].to(state["device"], dtype=state["dtype"])
    if "input_values" in asr_inputs:
        in_kwargs["input_values"] = asr_inputs["input_values"].to(state["device"], dtype=state["dtype"])
    if "attention_mask" in asr_inputs:
        in_kwargs["attention_mask"] = asr_inputs["attention_mask"].to(state["device"])
    with torch.no_grad():
        logits = state["asr_model"](**in_kwargs).logits
    pred = state["asr_proc"].batch_decode(logits.argmax(-1))[0]
    stripped, _ = strip_lang_token(pred)
    asr_clean = paper_postprocess(stripped.lower())
    asr_time = time.time() - t0

    duration_str = f"{len(wav)/sr:.2f} s  ·  synth {synth_time:.2f}s + asr {asr_time:.2f}s"
    return (sr, wav), rom, asr_clean or "—", duration_str


def build_ui(state, tts_id: str, asr_id: str):
    def _gen(text):
        return synthesize(state, text)

    title_md = f"""
# Ethio-TTS Demo (Amharic)

Convert **Amharic (Ge'ez script)** text to speech, then round-trip the
audio through our **Ethio-ASR** model to check intelligibility.

* **TTS:** `{tts_id}` (zero-shot, 36M params, 16 kHz)
* **ASR (round-trip check):** `{asr_id}`
* **Romanization:** [uroman](https://github.com/isi-nlp/uroman) — auto.
"""

    with gr.Blocks(title="Ethio-TTS Demo") as demo:
        gr.Markdown(title_md)

        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(
                    label="Amharic text (Ge'ez script)",
                    value=EXAMPLES[0],
                    lines=3,
                    placeholder="ሰላም ለዓለም",
                )
                gr.Examples(EXAMPLES, inputs=text_in, label="Click an example")
                btn = gr.Button("Synthesize", variant="primary")
            with gr.Column(scale=2):
                audio_out = gr.Audio(label="Synthesized audio", type="numpy",
                                     interactive=False, autoplay=False)
                rom_out = gr.Textbox(label="Romanized form (uroman)",
                                     interactive=False, lines=2)
                asr_out = gr.Textbox(label="Round-trip ASR transcription",
                                     interactive=False, lines=2)
                dur_out = gr.Textbox(label="Stats", interactive=False)

        btn.click(_gen, text_in, [audio_out, rom_out, asr_out, dur_out])
        gr.Markdown(
            "**Round-trip WER** is the gap between *input* Amharic text "
            "and the ASR transcription of the synthesized audio. Lower is "
            "better. Zero-shot MMS-TTS-Amh produces intelligible speech "
            "but with a noticeable foreign accent and limited prosody. "
            "Fine-tuning on WAXAL closes some of this gap."
        )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts_model", default="facebook/mms-tts-amh")
    ap.add_argument("--asr_model", default="boazsew/Ethio-ASR-w2v-bert-2.0-uf")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    state = load_models(args.tts_model, args.asr_model)
    demo = build_ui(state, args.tts_model, args.asr_model)
    demo.queue(default_concurrency_limit=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
