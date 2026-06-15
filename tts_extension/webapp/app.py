#!/usr/bin/env python3
"""Gradio TTS demo — multi-language + 4 voice profiles via OpenVoice v2.

Pipeline:
  text -> MMS-TTS-{lang} -> OpenVoice ToneColorConverter(profile) -> audio

Profiles: Young Male / Young Female / Old Male / Old Female
(plus 'default' = the unmodified MMS-TTS voice).

Reference voices were auto-picked from WAXAL Amharic speakers using an
F0-based age proxy (see slurm_scripts/pick_reference_voices.py).
"""
import argparse
import os
import sys
import time
from pathlib import Path

PR = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
os.environ.setdefault("HF_HOME", str(PR / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PR / "hf_cache" / "hub"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(PR / "tmp_cache" / "gradio"))
for k in ("HF_HOME", "HF_HUB_CACHE", "GRADIO_TEMP_DIR"):
    Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PR / "slurm_scripts"))
sys.path.insert(0, "/blue/rcstudents/btulu/Projects/Ethio-ASR/slurm_scripts")

import numpy as np
import torch
import gradio as gr
from transformers import AutoProcessor, AutoModelForCTC

from voice_pipeline import VoicePipeline, LANG_TO_TTS, PROFILES
from evaluate_ctc import paper_postprocess, strip_lang_token  # type: ignore


LANGUAGES = {
    "Amharic (አማርኛ)":     {"code": "amh", "examples": [
        "ሰላም ለዓለም",
        "አማርኛ ቋንቋ ነው",
        "የሰው ልጆች ሁሉ ነጻ ሆነው ይወለዳሉ።",
        "ኢትዮጵያ ቆንጆ አገር ናት",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ነች"]},
    "Tigrinya (ትግርኛ)":    {"code": "tir", "examples": [
        "ሰላም ንዓለም",
        "ትግርኛ ቋንቋ እዩ",
        "ኩሎም ሰባት ብነጻ ይውለዱ።",
        "ኢትዮጵያ ጽብቕቲ ሃገር እያ",
        "መቐለ ኣብ ትግራይ ዘሎ ከተማ እዩ"]},
    "Oromo (Afaan Oromoo)": {"code": "orm", "examples": [
        "Nagaa addunyaaf",
        "Afaan Oromoo afaan keenya.",
        "Namni hundi walqixa dhalata.",
        "Itoophiyaan biyya bareedduu dha.",
        "Finfinneen magaalaa guddoo Itoophiyaa ti."]},
    "Sidaama (Sidaamu Afoo)": {"code": "sid", "examples": [
        "Salaamu hanafote",
        "Sidaamu afoo ninka afoo'iho.",
        "Manchu hudi'ne kalaqamunni wolaphinoho.",
        "Itophiya seekkote gobba'iho.",
        "Hawaasaa Sidaamira lowo katama'iho."]},
}

VOICE_LABELS = {
    "Default (MMS-TTS)":  "default",
    "Young Male 👨":       "young_male",
    "Young Female 👩":     "young_female",
    "Old Male 👴":         "old_male",
    "Old Female 👵":       "old_female",
}

ASR_MODEL = "boazsew/Ethio-ASR-w2v-bert-2.0-uf"


def load_asr():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[app] loading ASR {ASR_MODEL}", flush=True)
    proc = AutoProcessor.from_pretrained(ASR_MODEL)
    model = AutoModelForCTC.from_pretrained(ASR_MODEL, torch_dtype=dtype).to(device).eval()
    return {"proc": proc, "model": model, "device": device, "dtype": dtype}


def asr_recognise(audio_array, sr, asr):
    inputs = asr["proc"](audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
    in_kwargs = {}
    if "input_features" in inputs:
        in_kwargs["input_features"] = inputs["input_features"].to(asr["device"], dtype=asr["dtype"])
    if "input_values" in inputs:
        in_kwargs["input_values"] = inputs["input_values"].to(asr["device"], dtype=asr["dtype"])
    if "attention_mask" in inputs:
        in_kwargs["attention_mask"] = inputs["attention_mask"].to(asr["device"])
    with torch.no_grad():
        logits = asr["model"](**in_kwargs).logits
    pred = asr["proc"].batch_decode(logits.argmax(-1))[0]
    stripped, lang_tok = strip_lang_token(pred)
    return paper_postprocess(stripped.lower()), lang_tok


def build_ui(pipeline: VoicePipeline, asr):
    def _gen(text, language_label, voice_label, tau):
        if not text or not text.strip():
            return None, "—", "—", "—"
        cfg = LANGUAGES[language_label]
        lang = cfg["code"]
        profile = VOICE_LABELS[voice_label]
        t0 = time.time()
        try:
            wav, sr, rom = pipeline.synthesize(text.strip(), lang,
                                                profile=profile, tau=float(tau))
        except Exception as e:
            return None, "—", "—", f"ERROR: {e}"
        synth_dt = time.time() - t0
        # Round-trip ASR
        t0 = time.time()
        try:
            asr_text, asr_lang = asr_recognise(wav.astype(np.float32), sr, asr)
        except Exception as e:
            asr_text, asr_lang = f"ASR error: {e}", None
        asr_dt = time.time() - t0
        info = (f"audio {len(wav)/sr:.2f}s @ {sr}Hz · synth {synth_dt:.2f}s + asr {asr_dt:.2f}s · "
                f"voice={profile} · LID={asr_lang or '?'}")
        return (sr, wav), rom, asr_text or "—", info

    def _set_lang(lang):
        examples = LANGUAGES[lang]["examples"]
        return examples[0], gr.update(samples=[[e] for e in examples])

    with gr.Blocks(title="Ethio-TTS — 4 voices × 4 languages") as demo:
        gr.Markdown(f"""
# Ethio-TTS Demo — 4 voices, 4 languages

Synthesize Amharic, Tigrinya, Oromo or Sidaama with one of four voice
profiles (Young/Old × Male/Female) via **MMS-TTS + OpenVoice v2** tone-color
conversion.

* **Base TTS**: `facebook/mms-tts-{{amh,tir,orm,sid}}` (one model per lang)
* **Voice conversion**: OpenVoice v2 ToneColorConverter, conditioned on
  reference clips auto-curated from WAXAL via an F0-based age proxy
* **Romanizer**: uroman
* **Round-trip ASR (intelligibility check)**: `{ASR_MODEL}`

Pick a language → pick a voice → click Synthesize.
""")
        with gr.Row():
            with gr.Column(scale=2):
                lang = gr.Dropdown(list(LANGUAGES), value="Amharic (አማርኛ)",
                                   label="Language")
                voice = gr.Radio(list(VOICE_LABELS), value="Young Female 👩",
                                  label="Voice profile")
                text_in = gr.Textbox(
                    label="Text",
                    value=LANGUAGES["Amharic (አማርኛ)"]["examples"][0],
                    lines=3,
                )
                example_set = gr.Dataset(
                    components=[text_in],
                    samples=[[e] for e in LANGUAGES["Amharic (አማርኛ)"]["examples"]],
                    label="Click an example",
                )
                with gr.Accordion("Advanced", open=False):
                    tau = gr.Slider(0.05, 1.0, value=0.30, step=0.05,
                                     label="Conversion strength τ (lower = closer to ref voice)")
                btn = gr.Button("Synthesize", variant="primary")
            with gr.Column(scale=2):
                audio_out = gr.Audio(label="Synthesized audio", type="numpy",
                                     interactive=False, autoplay=False)
                rom_out = gr.Textbox(label="Romanized form (uroman)",
                                     interactive=False, lines=2)
                asr_out = gr.Textbox(label="Round-trip ASR transcription",
                                     interactive=False, lines=2)
                info_out = gr.Textbox(label="Run info", interactive=False, lines=2)

        lang.change(_set_lang, lang, [text_in, example_set])
        example_set.click(lambda x: x[0], example_set, text_in)
        btn.click(_gen, [text_in, lang, voice, tau],
                  [audio_out, rom_out, asr_out, info_out])

        gr.Markdown(
            "**About the voices:** reference clips were auto-selected from "
            "WAXAL Amharic speakers — within each gender, the speaker with "
            "the highest median F0 became 'young' and the lowest became 'old'. "
            "F0 is an imperfect age proxy; future work would use explicit "
            "age annotations (e.g., from Mozilla Common Voice)."
        )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    pipeline = VoicePipeline()
    # Warm-load Amharic + caches
    pipeline._get_tts("amh")
    pipeline._ensure_source_se("amh")
    pipeline._ensure_target_ses()
    asr = load_asr()
    demo = build_ui(pipeline, asr)
    demo.queue(default_concurrency_limit=2).launch(
        server_name=args.host, server_port=args.port,
        share=args.share, show_api=False,
    )


if __name__ == "__main__":
    main()
