#!/usr/bin/env python3
"""Download Mozilla Common Voice Amharic clips that have age + gender metadata.

CV stores age as "teens", "twenties", "thirties", ..., "eighties".
We bucket into young = teens/twenties/thirties, old = fifties/sixties/seventies/eighties.

Output: writes 4 reference clips per language to
  reference_voices/amh/{young_male,young_female,old_male,old_female}/ref.wav
plus a manifest with the speaker IDs.
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

PR = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
os.environ.setdefault("HF_HOME", str(PR / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PR / "hf_cache" / "hub"))

import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio
from huggingface_hub import login

token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
if token:
    login(token=token, add_to_git_credential=False)

YOUNG = {"teens", "twenties", "thirties"}
OLD   = {"fifties", "sixties", "seventies", "eighties"}

OUT_BASE = PR / "reference_voices"
REF_TARGET_SEC = 30.0   # aim for ~30s of voice per profile

DATASET = "mozilla-foundation/common_voice_17_0"
SPLIT_PREFS = ["train", "validation", "test"]


def pick_clips(ds, age_bucket: str, gender: str, target_sec: float):
    """Pick clips totaling ~target_sec seconds for (age, gender)."""
    pool = []
    for ex in ds:
        if ex.get("gender") != gender:
            continue
        if ex.get("age") not in age_bucket:
            continue
        # Use sentence length as a quality proxy (very short clips often have OOV)
        sent = ex.get("sentence", "") or ""
        if not (10 <= len(sent) <= 200):
            continue
        pool.append(ex)
    return pool


def materialise(pool, label, out_dir: Path, target_sec: float):
    """Decode audio, concatenate enough to hit target_sec, write a single ref.wav."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pool:
        return None
    chunks = []
    acc = 0.0
    speaker_ids = []
    for ex in pool:
        try:
            wav = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = ex["audio"]["sampling_rate"]
        except Exception as e:
            continue
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        dur = len(wav) / sr
        if dur < 1.5 or dur > 12.0:
            continue
        # Trim silence
        wav, _ = librosa.effects.trim(wav, top_db=30)
        if len(wav)/sr < 1.0:
            continue
        # Light loudness normalize
        peak = float(np.max(np.abs(wav))) or 1.0
        wav = wav * (0.95 / peak)
        chunks.append(wav)
        acc += len(wav) / sr
        speaker_ids.append(ex.get("client_id", "?")[:12])
        if acc >= target_sec:
            break
    if not chunks:
        return None
    silence = np.zeros(int(0.3 * 16000), dtype=np.float32)
    audio = np.concatenate(sum(([c, silence] for c in chunks), []))
    out_path = out_dir / "ref.wav"
    sf.write(str(out_path), audio, 16000)
    return {"path": str(out_path), "duration_s": round(len(audio)/16000, 2),
            "n_clips": len(chunks), "speakers": speaker_ids}


def main():
    out_dir = OUT_BASE / "amh"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cv] loading {DATASET} amh train (this can be slow first time)...", flush=True)
    # Use streaming=False but trust_remote_code maybe needed for CV
    ds = load_dataset(DATASET, "am", split="train+validation+test",
                      verification_mode="no_checks")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Index metadata cheaply (no audio decode)
    print(f"[cv] dataset size: {len(ds)}", flush=True)
    ages_all = ds["age"]
    genders_all = ds["gender"]
    from collections import Counter
    print("[cv] age distribution:", dict(Counter([a or "—" for a in ages_all])))
    print("[cv] gender distribution:", dict(Counter([g or "—" for g in genders_all])))

    # Pre-pick indices per profile WITHOUT decoding audio
    profiles = {
        "young_male":   (YOUNG, "male_masculine"),
        "young_female": (YOUNG, "female_feminine"),
        "old_male":     (OLD,   "male_masculine"),
        "old_female":   (OLD,   "female_feminine"),
    }
    # CV gender field uses old labels too
    GENDER_FALLBACK = {
        "male_masculine": ["male", "male_masculine"],
        "female_feminine": ["female", "female_feminine"],
    }

    manifest = {}
    for prof_name, (age_set, gender) in profiles.items():
        valid_genders = GENDER_FALLBACK.get(gender, [gender])
        idxs = [i for i, (a, g) in enumerate(zip(ages_all, genders_all))
                if a in age_set and g in valid_genders]
        print(f"[cv] {prof_name}: {len(idxs)} candidate clips")
        if not idxs:
            manifest[prof_name] = None
            continue
        sub = ds.select(idxs)
        result = materialise(sub, prof_name, out_dir / prof_name, REF_TARGET_SEC)
        manifest[prof_name] = result
        print(f"[cv] {prof_name}: {result}")

    import json
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[cv] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
