"""End-to-end TTS + Voice-cloning pipeline for Ethio-TTS.

Architecture:
  text (Ge'ez) --uroman--> romanized text
  romanized text --MMS-TTS-{lang}--> audio at 16 kHz (default MMS voice)
  audio --OpenVoice v2 ToneColorConverter (src_se, tgt_se)--> audio at 22050 Hz
         in the timbre of the chosen reference voice.

We pre-compute and cache:
  - tgt_se for each of {lang} x {young_male, young_female, old_male, old_female}
  - src_se for each MMS-TTS-{lang} voice (extracted from a representative
    MMS-TTS sample, computed once on init)
"""
import os
import sys
import json
import tempfile
from pathlib import Path

PR = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
sys.path.insert(0, str(PR / "slurm_scripts"))

import numpy as np
import torch
import librosa
import soundfile as sf
from transformers import VitsModel, AutoTokenizer
from uroman import Uroman

from openvoice_minimal import load_converter

LANG_TO_TTS = {
    "amh": "facebook/mms-tts-amh",
    "tir": "facebook/mms-tts-tir",
    "orm": "facebook/mms-tts-orm",
    "sid": "facebook/mms-tts-sid",
}
PROFILES = ["young_male", "young_female", "old_male", "old_female"]
REF_DIR = PR / "reference_voices"
SE_CACHE = PR / "openvoice_cache" / "ses"
SE_CACHE.mkdir(parents=True, exist_ok=True)

# A short "anchor" Amharic phrase used to extract a SOURCE speaker embedding
# representative of MMS-TTS-{lang} default voice (one per language).
ANCHOR = {
    "amh": "ሰላም ለዓለም። ይህ የቋንቋ ናሙና ነው።",
    "tir": "ሰላም ንዓለም። እዚ ናይ ቋንቋ ናሙና እዩ።",
    "orm": "Nagaa addunyaaf. Kun fakkeenya afaanii ti.",
    "sid": "Salaamu hanafote. Konni afoo'iho fakkeenya.",
}


PHASE2_DIR = PR / "models" / "phase2"

class VoicePipeline:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[pipeline] device={self.device}", flush=True)
        self.uroman = Uroman()
        self.tts_models = {}             # lang -> (model, tokenizer)
        self.tcc = load_converter(device=self.device)
        self.target_ses = {}             # (lang, profile) -> tgt_se Tensor
        self.source_ses = {}             # lang -> src_se Tensor
        self.phase2_models = {}          # (lang, profile) -> (VitsModel, tokenizer)

    # ---------- Phase 2 (fine-tuned VITS) ----------
    def _get_phase2(self, lang: str, profile: str):
        """Return (VitsModel, tokenizer) for a fine-tuned voice, or None if
        no Phase-2 model exists for this (lang, profile)."""
        key = (lang, profile)
        if key in self.phase2_models:
            return self.phase2_models[key]
        cand = PHASE2_DIR / f"{lang}_{profile}"
        if not (cand / "model.safetensors").exists():
            return None
        print(f"[pipeline] loading Phase 2 model {cand.name}", flush=True)
        m = VitsModel.from_pretrained(str(cand)).to(self.device).eval()
        t = AutoTokenizer.from_pretrained(str(cand))
        self.phase2_models[key] = (m, t)
        return (m, t)

    def synthesize_phase2(self, geez_text: str, lang: str, profile: str):
        """Direct fine-tuned VITS synthesis (no OpenVoice). Returns
        (wav, sr, romanized_text) or None if no Phase-2 model exists."""
        pair = self._get_phase2(lang, profile)
        if pair is None:
            return None
        m, tok = pair
        rom = self.uroman.romanize_string(geez_text.strip())
        inputs = tok(rom, return_tensors="pt").to(self.device)
        with torch.no_grad():
            wav = m(**inputs).waveform
        return (wav.cpu().squeeze().float().numpy(),
                m.config.sampling_rate,
                rom)

    # ---------- TTS model lazy-load ----------
    def _get_tts(self, lang: str):
        if lang not in self.tts_models:
            repo = LANG_TO_TTS[lang]
            print(f"[pipeline] loading {repo}", flush=True)
            m = VitsModel.from_pretrained(repo).to(self.device).eval()
            t = AutoTokenizer.from_pretrained(repo)
            self.tts_models[lang] = (m, t)
        return self.tts_models[lang]

    # ---------- Speaker embedding cache ----------
    def _ensure_target_ses(self):
        for lang in LANG_TO_TTS:
            for prof in PROFILES:
                key = f"{lang}_{prof}"
                if key in self.target_ses:
                    continue
                cache = SE_CACHE / f"tgt_{key}.pt"
                if cache.exists():
                    self.target_ses[key] = torch.load(cache, map_location=self.device)
                    continue
                ref_wav = REF_DIR / lang / prof / "ref.wav"
                if not ref_wav.exists():
                    print(f"[pipeline] missing reference: {ref_wav}")
                    continue
                se = self.tcc.extract_se([str(ref_wav)])
                torch.save(se, cache)
                self.target_ses[key] = se.to(self.device)
                print(f"[pipeline] cached tgt_se {key}")

    def _ensure_source_se(self, lang: str):
        if lang in self.source_ses:
            return
        cache = SE_CACHE / f"src_{lang}.pt"
        if cache.exists():
            self.source_ses[lang] = torch.load(cache, map_location=self.device)
            return
        # Synthesize anchor text with MMS-TTS-{lang} -> wav -> extract_se
        wav, sr = self._synth_raw(ANCHOR[lang], lang)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                         dir=str(PR / "tmp_cache")) as tf:
            sf.write(tf.name, wav, sr)
            se = self.tcc.extract_se([tf.name])
        os.unlink(tf.name)
        torch.save(se, cache)
        self.source_ses[lang] = se.to(self.device)
        print(f"[pipeline] cached src_se {lang}")

    # ---------- Synth + convert ----------
    def _synth_raw(self, geez_text: str, lang: str):
        m, tok = self._get_tts(lang)
        rom = self.uroman.romanize_string(geez_text)
        inputs = tok(rom, return_tensors="pt").to(self.device)
        with torch.no_grad():
            wav = m(**inputs).waveform
        wav = wav.cpu().squeeze().float().numpy()
        return wav, m.config.sampling_rate

    def synthesize(self, geez_text: str, lang: str, profile: str = None,
                   tau: float = 0.3):
        """Return (wav, sr, romanized_text). If profile is None, returns
        the raw default MMS-TTS voice. Otherwise applies tone-color
        conversion to the chosen profile."""
        wav, sr = self._synth_raw(geez_text, lang)
        rom = self.uroman.romanize_string(geez_text)
        if profile is None or profile == "default":
            return wav, sr, rom
        self._ensure_source_se(lang)
        self._ensure_target_ses()
        key = f"{lang}_{profile}"
        if key not in self.target_ses:
            return wav, sr, rom + f"  [WARN: voice {profile} not available]"

        # Write to temp wav (OpenVoice expects a file path; resamples to 22050)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                         dir=str(PR / "tmp_cache")) as tf:
            sf.write(tf.name, wav, sr)
            converted = self.tcc.convert(
                audio_src_path=tf.name,
                src_se=self.source_ses[lang],
                tgt_se=self.target_ses[key],
                tau=tau,
                message="",
            )
        os.unlink(tf.name)
        out_sr = self.tcc.hps.data.sampling_rate
        return converted, out_sr, rom


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="amh", choices=list(LANG_TO_TTS))
    ap.add_argument("--profile", default=None,
                    choices=["default"] + PROFILES + [None])
    ap.add_argument("--out", default=str(PR / "tmp_cache" / "out.wav"))
    args = ap.parse_args()
    p = VoicePipeline()
    wav, sr, rom = p.synthesize(args.text, args.lang, args.profile)
    sf.write(args.out, wav, sr)
    print(f"[main] wrote {args.out} ({len(wav)/sr:.2f}s @ {sr}Hz) rom={rom!r}")


if __name__ == "__main__":
    main()
