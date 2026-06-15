#!/usr/bin/env python3
"""Gradio web demo for Ethio-ASR reproduction.

Loads one of our trained CTC models and exposes:
  - Pre-loaded WAXAL validation samples (one per language) for quick clicks
  - Audio file upload
  - Browser microphone recording

Prints the language ID (predicted by the model's [LANG] token), the raw
transcription, and the paper-post-processed transcription.

Usage:
  python app.py [--model_dir PATH] [--port 7860] [--host 0.0.0.0]
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-ASR")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "hf_cache" / "hub"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(PROJECT_ROOT / "tmp_cache" / "gradio"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp_cache" / "matplotlib"))
for k in ("HF_HOME", "HF_HUB_CACHE", "GRADIO_TEMP_DIR", "MPLCONFIGDIR"):
    Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

# Reuse the eval-time post-processing from our reproduction code
sys.path.insert(0, str(PROJECT_ROOT / "Ethio-ASR" / "reproduction_uf" / "slurm_scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "slurm_scripts"))
from evaluate_ctc import paper_postprocess, strip_lang_token  # type: ignore

import numpy as np
import torch
import gradio as gr
import librosa
from transformers import AutoProcessor, AutoModelForCTC

DEFAULT_MODEL = str(PROJECT_ROOT / "models" / "facebook" / "w2v-bert-2.0-09062026-212456")

# Mapping from short codes (used in WAXAL) to human-readable language names
LANG_CODE_TO_NAME = {
    "amh": "Amharic",
    "tir": "Tigrinya",
    "orm": "Oromo",
    "wal": "Wolaytta",
    "sid": "Sidaama",
}
NAME_TO_LANG_CODE = {v: k for k, v in LANG_CODE_TO_NAME.items()}

SAMPLES_DIR = PROJECT_ROOT / "Ethio-ASR" / "reproduction_uf" / "webapp" / "samples"


def load_model(model_dir: str):
    """Load processor and CTC model from disk."""
    print(f"[app] loading model from {model_dir}", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCTC.from_pretrained(model_dir, torch_dtype=dtype).to(device).eval()
    print(f"[app] model loaded in {time.time()-t0:.1f}s on {device} ({dtype})", flush=True)
    return processor, model, device, dtype


def cache_validation_samples(out_dir: Path, per_lang: int = 1) -> list[dict]:
    """Cache one validation sample per language as .wav files for the demo.

    Strategy: read the `language` and `audio_duration` columns first
    (no audio decode), select per-language indices, then materialise
    only those rows. ~100x faster than iterating the whole dataset.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = out_dir / "manifest.json"
    if manifest_file.exists():
        import json
        return json.loads(manifest_file.read_text())

    print(f"[app] caching {per_lang} validation sample(s) per language to {out_dir}", flush=True)
    from datasets import load_dataset, Audio
    import soundfile as sf
    import json

    ds = load_dataset("badrex/waxalNLP-ethiopic-final", split="validation",
                      verification_mode="no_checks")

    # Get language + duration as plain lists (no audio decode)
    langs = ds["language"]
    durs = ds["audio_duration"] if "audio_duration" in ds.column_names else [None] * len(ds)

    chosen_indices = []
    chosen_meta = []
    seen = {code: 0 for code in LANG_CODE_TO_NAME}
    for i, (lang, dur) in enumerate(zip(langs, durs)):
        lang = (lang or "").lower()
        if lang not in seen or seen[lang] >= per_lang:
            continue
        # WAXAL validation has WIDE duration ranges per language
        # (Amharic samples are all 10-30s for example).  Accept up to 25s.
        if dur is not None and (dur < 1.5 or dur > 25.0):
            continue
        chosen_indices.append(i)
        chosen_meta.append((lang, dur, seen[lang]))
        seen[lang] += 1
        if all(v >= per_lang for v in seen.values()):
            break

    # Materialise only the chosen rows
    print(f"[app] materialising {len(chosen_indices)} rows", flush=True)
    sub = ds.select(chosen_indices).cast_column("audio", Audio(sampling_rate=16000))

    manifest = []
    for j, ex in enumerate(sub):
        lang, dur, idx = chosen_meta[j]
        path = out_dir / f"{lang}_{idx:02d}.wav"
        sf.write(str(path), ex["audio"]["array"], ex["audio"]["sampling_rate"])
        manifest.append({
            "language_code": lang,
            "language_name": LANG_CODE_TO_NAME[lang],
            "filename": path.name,
            "duration": round(dur if dur else len(ex["audio"]["array"])/ex["audio"]["sampling_rate"], 2),
            "reference_transcription": ex.get("transcription", ""),
        })

    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[app] cached {len(manifest)} samples", flush=True)
    return manifest


def transcribe(audio_array: np.ndarray, sample_rate: int,
               processor, model, device, dtype) -> dict:
    """Run inference and return predicted language + raw + cleaned transcriptions."""
    if audio_array is None:
        return {"language": "—", "raw": "—", "clean": "—", "duration_s": 0.0}

    # ensure mono float32 at 16k
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)
    # normalize int16 input to [-1,1] if needed
    if np.abs(audio_array).max() > 1.5:
        audio_array = audio_array / 32768.0
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    t0 = time.time()
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
    in_kwargs = {}
    if "input_values" in inputs:
        in_kwargs["input_values"] = inputs["input_values"].to(device, dtype=dtype)
    if "input_features" in inputs:
        in_kwargs["input_features"] = inputs["input_features"].to(device, dtype=dtype)
    if "attention_mask" in inputs:
        in_kwargs["attention_mask"] = inputs["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(**in_kwargs).logits
    pred_ids = logits.argmax(dim=-1)
    raw = processor.batch_decode(pred_ids)[0]
    after_lang, lang_token = strip_lang_token(raw)
    cleaned = paper_postprocess(after_lang.lower())
    lang_label = "—"
    if lang_token:
        lang_label = LANG_CODE_TO_NAME.get(lang_token, lang_token).title()

    print(f"[app] transcribe ({len(audio_array)/16000:.2f}s audio) in {time.time()-t0:.2f}s", flush=True)
    return {
        "language": lang_label,
        "raw": raw or "—",
        "clean": cleaned or "—",
        "duration_s": round(len(audio_array) / 16000, 2),
    }


def build_ui(processor, model, device, dtype, samples: list[dict],
             model_name: str):

    def _transcribe_file(audio_path):
        if not audio_path:
            return "—", "—", "—", "—"
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        out = transcribe(y, sr, processor, model, device, dtype)
        return out["language"], out["raw"], out["clean"], f"{out['duration_s']} s"

    def _transcribe_mic(audio_tuple):
        # gr.Audio(type="numpy") returns (sample_rate, numpy_array)
        if audio_tuple is None:
            return "—", "—", "—", "—"
        sr, y = audio_tuple
        out = transcribe(y, sr, processor, model, device, dtype)
        return out["language"], out["raw"], out["clean"], f"{out['duration_s']} s"

    sample_lookup = {
        f"{s['language_name']} — {s['filename']} ({s['duration']}s)":
            (str(SAMPLES_DIR / s['filename']), s['reference_transcription'])
        for s in samples
    }
    sample_labels = list(sample_lookup)

    def _transcribe_sample(label):
        if not label or label not in sample_lookup:
            return None, "—", "—", "—", "—", "—"
        path, ref = sample_lookup[label]
        y, sr = librosa.load(path, sr=None, mono=True)
        out = transcribe(y, sr, processor, model, device, dtype)
        return path, ref, out["language"], out["raw"], out["clean"], f"{out['duration_s']} s"

    title_md = f"""
# Ethio-ASR Demo (reproduction)

Multilingual speech recognition for **Amharic, Tigrinya, Oromo, Sidaama, Wolaytta**.

**Model:** `{model_name}`  ·  Device: `{device}` ({dtype})
"""

    with gr.Blocks(title="Ethio-ASR Demo") as demo:
        gr.Markdown(title_md)

        with gr.Tab("Pre-loaded validation samples"):
            gr.Markdown("Pick a WAXAL validation utterance and click **Transcribe**.")
            sample_dropdown = gr.Dropdown(choices=sample_labels,
                                          value=sample_labels[0] if sample_labels else None,
                                          label="Sample (one per language)")
            sample_player = gr.Audio(label="Selected audio", type="filepath",
                                     interactive=False)
            sample_btn = gr.Button("Transcribe sample", variant="primary")
            with gr.Row():
                s_lang = gr.Textbox(label="Predicted language", interactive=False)
                s_dur = gr.Textbox(label="Duration", interactive=False)
            s_ref = gr.Textbox(label="Reference (gold)", interactive=False,
                               lines=2)
            s_raw = gr.Textbox(label="Raw prediction", interactive=False, lines=2)
            s_pp = gr.Textbox(label="Post-processed prediction", interactive=False,
                              lines=2)
            sample_btn.click(_transcribe_sample, sample_dropdown,
                             [sample_player, s_ref, s_lang, s_raw, s_pp, s_dur])

        with gr.Tab("Upload audio"):
            gr.Markdown("Upload a WAV/FLAC/MP3 file. Resampled to 16 kHz mono.")
            up_audio = gr.Audio(label="Audio file", type="filepath", sources=["upload"])
            up_btn = gr.Button("Transcribe upload", variant="primary")
            with gr.Row():
                u_lang = gr.Textbox(label="Predicted language", interactive=False)
                u_dur = gr.Textbox(label="Duration", interactive=False)
            u_raw = gr.Textbox(label="Raw prediction", interactive=False, lines=2)
            u_pp = gr.Textbox(label="Post-processed prediction", interactive=False,
                              lines=2)
            up_btn.click(_transcribe_file, up_audio, [u_lang, u_raw, u_pp, u_dur])

        with gr.Tab("Record from microphone"):
            gr.Markdown("Record a short clip (≤ ~30 s).")
            mic_audio = gr.Audio(label="Microphone", type="numpy", sources=["microphone"])
            mic_btn = gr.Button("Transcribe recording", variant="primary")
            with gr.Row():
                m_lang = gr.Textbox(label="Predicted language", interactive=False)
                m_dur = gr.Textbox(label="Duration", interactive=False)
            m_raw = gr.Textbox(label="Raw prediction", interactive=False, lines=2)
            m_pp = gr.Textbox(label="Post-processed prediction", interactive=False,
                              lines=2)
            mic_btn.click(_transcribe_mic, mic_audio, [m_lang, m_raw, m_pp, m_dur])

        gr.Markdown(
            "Post-processing = paper-faithful Ge'ez homophone normalisation "
            "and punctuation removal (Section 5.2 of Abdullah et al., 2026)."
        )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=DEFAULT_MODEL,
                    help=f"Path to trained model (default: {DEFAULT_MODEL})")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0",
                    help="Bind address (use 0.0.0.0 to allow remote access)")
    ap.add_argument("--share", action="store_true",
                    help="Get a public gradio.live URL (requires internet)")
    ap.add_argument("--samples", type=int, default=1,
                    help="Pre-cached samples per language (default 1)")
    args = ap.parse_args()

    processor, model, device, dtype = load_model(args.model_dir)
    samples = cache_validation_samples(SAMPLES_DIR, per_lang=args.samples)
    demo = build_ui(processor, model, device, dtype, samples,
                    model_name=Path(args.model_dir).name)
    demo.queue(default_concurrency_limit=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
