#!/usr/bin/env python3
"""Combined Ethiopian-language demo: TTS and ASR in one Gradio app.

Tab 1 - Text-to-Speech:
   text (any of 4 langs) -> MMS-TTS -> OpenVoice tone-color (4 voices)
Tab 2 - Speech-to-Text:
   audio (file / mic / preloaded WAXAL sample) -> Ethio-ASR w2v-bert-2.0

One SLURM job, one GPU, one share URL.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PR_TTS = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
PR_ASR = Path("/blue/rcstudents/btulu/Projects/Ethio-ASR")

os.environ.setdefault("HF_HOME", str(PR_TTS / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PR_TTS / "hf_cache" / "hub"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(PR_TTS / "tmp_cache" / "gradio"))
for k in ("HF_HOME", "HF_HUB_CACHE", "GRADIO_TEMP_DIR"):
    Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

# import helpers from both projects
sys.path.insert(0, str(PR_TTS / "slurm_scripts"))
sys.path.insert(0, str(PR_ASR / "slurm_scripts"))

import numpy as np
import torch
import librosa
import gradio as gr
from transformers import AutoProcessor, AutoModelForCTC

from voice_pipeline import VoicePipeline, LANG_TO_TTS, PROFILES
from evaluate_ctc import paper_postprocess, strip_lang_token  # type: ignore

ASR_MODEL = "boazsew/Ethio-ASR-w2v-bert-2.0-uf"

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

ENGINES = {
    "Phase 1 — OpenVoice tone-color":              "phase1",
    "Phase 2 — fine-tuned VITS (Amharic only)":     "phase2",
}

# ---------- Model loading ----------
def load_asr():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[combined] loading ASR {ASR_MODEL}", flush=True)
    proc = AutoProcessor.from_pretrained(ASR_MODEL)
    model = AutoModelForCTC.from_pretrained(ASR_MODEL, torch_dtype=dtype).to(device).eval()
    return {"proc": proc, "model": model, "device": device, "dtype": dtype}


def cache_validation_samples(out_dir: Path, per_lang: int = 1) -> list[dict]:
    """Cache one WAXAL validation sample per language as ref WAVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = out_dir / "manifest.json"
    if manifest_file.exists():
        return json.loads(manifest_file.read_text())

    print(f"[combined] caching validation samples to {out_dir}", flush=True)
    os.environ["HF_HUB_CACHE"] = str(PR_ASR / "hf_cache" / "hub")
    from datasets import load_dataset, Audio
    import soundfile as sf
    ds = load_dataset("badrex/waxalNLP-ethiopic-final", split="validation",
                      verification_mode="no_checks")
    langs = ds["language"]; durs = ds["audio_duration"]
    lang_name = {"amh": "Amharic", "tir": "Tigrinya", "orm": "Oromo",
                  "wal": "Wolaytta", "sid": "Sidaama"}
    seen = {c: 0 for c in lang_name}
    keep_idx = []
    keep_meta = []
    for i, (l, d) in enumerate(zip(langs, durs)):
        l = (l or "").lower()
        if l not in seen or seen[l] >= per_lang:
            continue
        if d is None or not (1.5 <= d <= 25.0):
            continue
        keep_idx.append(i)
        keep_meta.append((l, d, seen[l]))
        seen[l] += 1
        if all(v >= per_lang for v in seen.values()):
            break
    sub = ds.select(keep_idx).cast_column("audio", Audio(sampling_rate=16000))
    manifest = []
    for j, ex in enumerate(sub):
        l, dur, idx = keep_meta[j]
        path = out_dir / f"{l}_{idx:02d}.wav"
        sf.write(str(path), ex["audio"]["array"], 16000)
        manifest.append({
            "language_code": l,
            "language_name": lang_name[l],
            "filename": path.name,
            "duration": round(dur, 2),
            "reference_transcription": ex.get("transcription", ""),
        })
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def asr_recognise(audio_array, sr, asr):
    if sr != 16000:
        audio_array = librosa.resample(audio_array.astype(np.float32),
                                       orig_sr=sr, target_sr=16000)
        sr = 16000
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if np.abs(audio_array).max() > 1.5:
        audio_array = audio_array.astype(np.float32) / 32768.0
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
    return paper_postprocess(stripped.lower()), lang_tok, pred


# ---------- UI ----------
def build_ui(pipeline, asr, samples_dir, samples):
    sample_lookup = {
        f"{s['language_name']} — {s['filename']} ({s['duration']}s)":
            (str(samples_dir / s['filename']), s['reference_transcription'])
        for s in samples
    }
    sample_labels = list(sample_lookup)

    # --- TTS handler ---
    def tts_gen(text, language_label, voice_label, engine_label, tau):
        if not text or not text.strip():
            return None, "—", "—", "—"
        cfg = LANGUAGES[language_label]
        lang = cfg["code"]
        profile = VOICE_LABELS[voice_label]
        engine = ENGINES[engine_label]
        t0 = time.time()
        note = ""
        try:
            if engine == "phase2":
                result = pipeline.synthesize_phase2(text.strip(), lang, profile)
                if result is None:
                    note = (f"  ⚠ no Phase-2 model for {lang}/{profile} (only "
                             f"Amharic voices are fine-tuned); falling back to Phase 1")
                    wav, sr, rom = pipeline.synthesize(text.strip(), lang,
                                                       profile=profile, tau=float(tau))
                else:
                    wav, sr, rom = result
            else:  # phase1 (OpenVoice) or default MMS
                wav, sr, rom = pipeline.synthesize(text.strip(), lang,
                                                    profile=profile, tau=float(tau))
        except Exception as e:
            return None, "—", "—", f"ERROR: {e}"
        synth_dt = time.time() - t0
        t0 = time.time()
        asr_text, asr_lang, _ = asr_recognise(wav.astype(np.float32), sr, asr)
        asr_dt = time.time() - t0
        info = (f"audio {len(wav)/sr:.2f}s @ {sr}Hz · synth {synth_dt:.2f}s + asr {asr_dt:.2f}s · "
                f"engine={engine} · voice={profile} · LID={asr_lang or '?'}{note}")
        return (sr, wav), rom, asr_text or "—", info

    def tts_set_lang(lang):
        return LANGUAGES[lang]["examples"][0], gr.update(samples=[[e] for e in LANGUAGES[lang]["examples"]])

    # --- ASR handlers ---
    def asr_file(audio_path):
        if not audio_path:
            return "—", "—", "—", "—"
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        t0 = time.time()
        clean, lang, raw = asr_recognise(y, sr, asr)
        dt = time.time() - t0
        return (lang or "?"), raw or "—", clean or "—", f"{len(y)/sr:.2f}s  ·  {dt:.2f}s asr"

    def asr_mic(audio_tuple):
        if audio_tuple is None:
            return "—", "—", "—", "—"
        sr, y = audio_tuple
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        t0 = time.time()
        clean, lang, raw = asr_recognise(y, sr, asr)
        dt = time.time() - t0
        return (lang or "?"), raw or "—", clean or "—", f"{len(y)/sr:.2f}s  ·  {dt:.2f}s asr"

    def asr_sample(label):
        if not label or label not in sample_lookup:
            return None, "—", "—", "—", "—", "—"
        path, ref = sample_lookup[label]
        y, sr = librosa.load(path, sr=None, mono=True)
        t0 = time.time()
        clean, lang, raw = asr_recognise(y, sr, asr)
        dt = time.time() - t0
        return path, ref, (lang or "?"), raw or "—", clean or "—", f"{len(y)/sr:.2f}s  ·  {dt:.2f}s asr"

    with gr.Blocks(title="Ethio Speech — TTS + ASR") as demo:
        gr.Markdown(f"""
# Ethio Speech Demo — TTS + ASR

* **TTS**: 4 Ethiopian languages × 4 voice profiles
  (MMS-TTS + OpenVoice v2 tone-color conversion).
* **ASR**: 5 Ethiopian languages with our reproduced
  `{ASR_MODEL}` model (Phase-2 fine-tunes coming).
""")

        # --- TAB 1: TTS ---
        with gr.Tab("🔊 Text → Speech (TTS)"):
            with gr.Row():
                with gr.Column(scale=2):
                    tts_lang = gr.Dropdown(list(LANGUAGES),
                                            value="Amharic (አማርኛ)",
                                            label="Language")
                    tts_voice = gr.Radio(list(VOICE_LABELS),
                                          value="Young Female 👩",
                                          label="Voice profile")
                    tts_engine = gr.Radio(list(ENGINES),
                                           value="Phase 1 — OpenVoice tone-color",
                                           label="Engine (A/B compare Phase 1 vs Phase 2)")
                    tts_text = gr.Textbox(
                        label="Text",
                        value=LANGUAGES["Amharic (አማርኛ)"]["examples"][0],
                        lines=3,
                    )
                    tts_examples = gr.Dataset(
                        components=[tts_text],
                        samples=[[e] for e in LANGUAGES["Amharic (አማርኛ)"]["examples"]],
                        label="Click an example",
                    )
                    with gr.Accordion("Advanced", open=False):
                        tts_tau = gr.Slider(0.05, 1.0, value=0.30, step=0.05,
                                             label="Conversion strength τ (lower = closer to ref voice)")
                    tts_btn = gr.Button("Synthesize", variant="primary")
                with gr.Column(scale=2):
                    tts_audio = gr.Audio(label="Synthesized audio", type="numpy",
                                          interactive=False, autoplay=False)
                    tts_rom = gr.Textbox(label="Romanized (uroman)",
                                          interactive=False, lines=2)
                    tts_asr = gr.Textbox(label="Round-trip ASR transcription",
                                          interactive=False, lines=2)
                    tts_info = gr.Textbox(label="Run info", interactive=False, lines=2)
            tts_lang.change(tts_set_lang, tts_lang, [tts_text, tts_examples])
            tts_examples.click(lambda x: x[0], tts_examples, tts_text)
            tts_btn.click(tts_gen,
                          [tts_text, tts_lang, tts_voice, tts_engine, tts_tau],
                          [tts_audio, tts_rom, tts_asr, tts_info])

        # --- TAB 2: ASR pre-loaded sample ---
        with gr.Tab("🎤 Speech → Text (samples)"):
            gr.Markdown("Pick a WAXAL validation utterance and click **Transcribe**.")
            s_dropdown = gr.Dropdown(choices=sample_labels,
                                      value=sample_labels[0] if sample_labels else None,
                                      label="Sample (one per language)")
            s_player = gr.Audio(label="Selected audio", type="filepath",
                                 interactive=False)
            s_btn = gr.Button("Transcribe sample", variant="primary")
            with gr.Row():
                s_lang = gr.Textbox(label="Predicted language", interactive=False)
                s_dur = gr.Textbox(label="Duration", interactive=False)
            s_ref = gr.Textbox(label="Reference (gold)", interactive=False, lines=2)
            s_raw = gr.Textbox(label="Raw prediction", interactive=False, lines=2)
            s_pp = gr.Textbox(label="Post-processed prediction", interactive=False, lines=2)
            s_btn.click(asr_sample, s_dropdown,
                        [s_player, s_ref, s_lang, s_raw, s_pp, s_dur])

        # --- TAB 3: ASR upload ---
        with gr.Tab("📁 Speech → Text (upload)"):
            gr.Markdown("Upload a WAV / FLAC / MP3 file (any sample rate; will be resampled to 16 kHz).")
            up_audio = gr.Audio(label="Audio file", type="filepath", sources=["upload"])
            up_btn = gr.Button("Transcribe upload", variant="primary")
            with gr.Row():
                u_lang = gr.Textbox(label="Predicted language", interactive=False)
                u_dur = gr.Textbox(label="Duration", interactive=False)
            u_raw = gr.Textbox(label="Raw prediction", interactive=False, lines=2)
            u_pp = gr.Textbox(label="Post-processed prediction", interactive=False, lines=2)
            up_btn.click(asr_file, up_audio, [u_lang, u_raw, u_pp, u_dur])

        # --- TAB 4: ASR mic ---
        with gr.Tab("🎙️ Speech → Text (microphone)"):
            gr.Markdown("Record a short clip in your browser and click Transcribe.")
            mic_audio = gr.Audio(label="Microphone", type="numpy", sources=["microphone"])
            mic_btn = gr.Button("Transcribe recording", variant="primary")
            with gr.Row():
                m_lang = gr.Textbox(label="Predicted language", interactive=False)
                m_dur = gr.Textbox(label="Duration", interactive=False)
            m_raw = gr.Textbox(label="Raw prediction", interactive=False, lines=2)
            m_pp = gr.Textbox(label="Post-processed prediction", interactive=False, lines=2)
            mic_btn.click(asr_mic, mic_audio, [m_lang, m_raw, m_pp, m_dur])

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7863)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    pipeline = VoicePipeline()
    pipeline._get_tts("amh")
    pipeline._ensure_source_se("amh")
    pipeline._ensure_target_ses()
    asr = load_asr()

    samples_dir = PR_TTS / "webapp" / "samples_asr"
    samples = cache_validation_samples(samples_dir, per_lang=1)

    demo = build_ui(pipeline, asr, samples_dir, samples)
    demo.queue(default_concurrency_limit=2).launch(
        server_name=args.host, server_port=args.port,
        share=args.share, show_api=False,
    )


if __name__ == "__main__":
    main()
