#!/usr/bin/env python3
"""Zero-shot smoke test for facebook/mms-tts-amh.

Romanize Amharic text with uroman, synthesize, save WAVs to disk.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
OUT_DIR = PROJECT_ROOT / "tmp_cache" / "zero_shot_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer
from uroman import Uroman

print("[smoke] loading model...")
model = VitsModel.from_pretrained("facebook/mms-tts-amh")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-amh")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
sr = model.config.sampling_rate
print(f"[smoke] model on {device}, sample_rate={sr} Hz, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

uroman = Uroman()

texts = [
    "ሰላም ለዓለም",
    "አማርኛ ቋንቋ ነው",
    "የሰው ልጆች ሁሉ ነጻ ሆነው ይወለዳሉ።",
    "ኢትዮጵያ ቆንጆ አገር ናት",
    "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ነች",
]

for i, ge in enumerate(texts):
    rom = uroman.romanize_string(ge)
    print(f"\n[{i}] GE'EZ: {ge}")
    print(f"[{i}] ROMAN: {rom}")
    inputs = tokenizer(rom, return_tensors="pt").to(device)
    print(f"[{i}] tokens: {inputs.input_ids.shape}")
    with torch.no_grad():
        wav = model(**inputs).waveform
    wav = wav.cpu().squeeze().float().numpy()
    out_path = OUT_DIR / f"sample_{i:02d}_zero_shot.wav"
    sf.write(str(out_path), wav, sr)
    print(f"[{i}] wrote {out_path}  ({len(wav)/sr:.2f}s)")

print(f"\n[smoke] all samples saved under {OUT_DIR}")
