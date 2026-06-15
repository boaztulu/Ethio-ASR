#!/usr/bin/env python3
"""Pick 4 reference voices (young M / young F / old M / old F) per language
from WAXAL using an F0-based age proxy.

WAXAL has speaker gender but no age metadata.  We use mean fundamental
frequency (F0) as a coarse age proxy:
  - Higher F0  -> typically younger voice
  - Lower F0   -> typically older voice
within each gender.

For each language:
  1. Pull top-N speakers by total duration of WAXAL clips (after our
     earlier 1-25s filter for Amharic, or on-the-fly for others).
  2. Compute mean F0 over a 10s sample of clean speech per speaker.
  3. Within each gender, pick speaker with highest F0 (young) and
     lowest F0 (old).
  4. Concatenate up to 30 s of that speaker's cleanest clips ->
     reference_voices/{lang}/{young_male,young_female,old_male,old_female}/ref.wav

Limitation: F0 is correlated with age but not perfectly; the resulting
voices reflect speakers who sound 'younger' or 'older' rather than
verified-by-metadata age cohorts.
"""
import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

PR = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
os.environ.setdefault("HF_HOME", str(PR / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", "/blue/rcstudents/btulu/Projects/Ethio-ASR/hf_cache/hub")

import numpy as np
import librosa
import soundfile as sf
from datasets import load_dataset, Audio
from huggingface_hub import login

token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
if token:
    login(token=token, add_to_git_credential=False)

REF_TARGET_SEC = 30.0
TOPN_SPEAKERS = 40           # consider this many top-duration speakers
PER_SPEAKER_PROBE_SEC = 8.0  # estimate F0 on this much audio per speaker
LANGS = ["amh", "tir", "orm", "sid"]  # MMS-TTS-supported langs
OUT = PR / "reference_voices"
OUT.mkdir(parents=True, exist_ok=True)


def estimate_f0(audio: np.ndarray, sr: int) -> float:
    """Return median F0 (Hz) over voiced frames, or 0 if no voiced frames."""
    f0, voiced_flag, _ = librosa.pyin(
        audio.astype(np.float32),
        fmin=70, fmax=400, sr=sr, frame_length=2048
    )
    if f0 is None:
        return 0.0
    voiced_f0 = f0[~np.isnan(f0)]
    if voiced_f0.size == 0:
        return 0.0
    return float(np.median(voiced_f0))


def collect_per_speaker_clips(ds, lang_code: str, min_dur: float = 2.0,
                              max_dur: float = 12.0):
    """Return dict speaker_id -> list of (index, duration, gender)."""
    langs = ds["language"]
    spids = ds["speaker_id"]
    durs = ds["audio_duration"]
    genders = ds["gender"]
    out = defaultdict(list)
    for i, (l, sp, d, g) in enumerate(zip(langs, spids, durs, genders)):
        if (l or "").lower() != lang_code:
            continue
        if d is None or not (min_dur <= d <= max_dur):
            continue
        if not sp:
            continue
        if not g:                  # need gender
            continue
        out[sp].append((i, d, g.lower()))
    return out


def materialise_ref(ds, indices, out_dir: Path):
    """Decode the chosen indices, trim silence, concatenate to ref.wav."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = ds.select(indices).cast_column("audio", Audio(sampling_rate=16000))
    chunks = []
    acc = 0.0
    for ex in sub:
        wav = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = ex["audio"]["sampling_rate"]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        wav, _ = librosa.effects.trim(wav, top_db=30)
        if len(wav)/16000 < 1.0:
            continue
        peak = float(np.max(np.abs(wav))) or 1.0
        wav = wav * (0.95 / peak)
        chunks.append(wav)
        acc += len(wav)/16000
        if acc >= REF_TARGET_SEC:
            break
    if not chunks:
        return None
    silence = np.zeros(int(0.3 * 16000), dtype=np.float32)
    audio = np.concatenate(sum(([c, silence] for c in chunks), []))
    sf.write(str(out_dir / "ref.wav"), audio, 16000)
    return round(len(audio)/16000, 2)


def main():
    print("[refs] loading WAXAL")
    ds = load_dataset("badrex/waxalNLP-ethiopic-final",
                      verification_mode="no_checks")
    # Combine train+validation pools (validation has many usable Amh)
    big_ds_train = ds["train"]
    big_ds_val = ds["validation"]

    manifest = {}
    for lang in LANGS:
        print(f"\n=== {lang} ===")
        lang_dir = OUT / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        per_spk_train = collect_per_speaker_clips(big_ds_train, lang)
        per_spk_val = collect_per_speaker_clips(big_ds_val, lang)
        per_spk = defaultdict(list)
        for d in (per_spk_train, per_spk_val):
            for sp, clips in d.items():
                per_spk[sp].extend([(i, dur, g, "train") for i, dur, g in d[sp]])

        if not per_spk:
            print(f"  [{lang}] no speakers")
            manifest[lang] = None
            continue

        # Top-N by total duration
        ranked = sorted(per_spk.items(),
                        key=lambda kv: -sum(c[1] for c in kv[1]))
        topn = ranked[:TOPN_SPEAKERS]
        print(f"  considering top {len(topn)} speakers")

        # Sample clips to probe F0 (one short clip per speaker, sum to ~8s)
        per_spk_probe = {}
        for sp, clips in topn:
            chosen = []
            acc = 0.0
            # Sort clips by duration ascending so we pick a couple short ones
            for i, dur, g, _ in sorted(clips, key=lambda x: x[1]):
                chosen.append(i)
                acc += dur
                if acc >= PER_SPEAKER_PROBE_SEC:
                    break
            per_spk_probe[sp] = (chosen, clips[0][2])  # gender from any clip

        # Decode probes and compute F0
        all_probe_idxs = sorted({i for v, _ in per_spk_probe.values() for i in v})
        ds_for_probe = (big_ds_train if any(c[3]=="train" for _, c, *_ in per_spk_probe.values() for _ in [None])
                        else big_ds_train)
        # Use train for probes (most data)
        probe_sub = big_ds_train.select(all_probe_idxs).cast_column(
            "audio", Audio(sampling_rate=16000))

        idx_to_probe = dict(zip(all_probe_idxs, probe_sub))
        spk_f0 = {}
        for sp, (idxs, g) in per_spk_probe.items():
            try:
                # Concat audio of this speaker's probe clips
                bits = []
                for i in idxs:
                    if i in idx_to_probe:
                        ex = idx_to_probe[i]
                        bits.append(np.asarray(ex["audio"]["array"], dtype=np.float32))
                if not bits:
                    continue
                audio = np.concatenate(bits)
                f0 = estimate_f0(audio, 16000)
                spk_f0[sp] = (f0, g)
            except Exception as e:
                print(f"   F0 fail for {sp}: {e}")

        if not spk_f0:
            print(f"  [{lang}] F0 estimation produced nothing")
            manifest[lang] = None
            continue

        # Split by gender and pick young (highest F0) + old (lowest F0)
        by_gender = defaultdict(list)
        for sp, (f0, g) in spk_f0.items():
            if g in ("m", "male"):
                by_gender["male"].append((f0, sp))
            elif g in ("f", "female"):
                by_gender["female"].append((f0, sp))
        for g, lst in by_gender.items():
            lst.sort()  # ascending F0

        for g_name, g_label in [("male", "male"), ("female", "female")]:
            print(f"  {g_name}: {len(by_gender[g_label])} speakers")

        chosen = {}
        for g_label, g_human in [("male", "male"), ("female", "female")]:
            lst = by_gender[g_label]
            if not lst:
                continue
            # Lowest F0 -> sounds older
            old_f0, old_sp = lst[0]
            # Highest F0 -> sounds younger
            young_f0, young_sp = lst[-1]
            chosen[f"old_{g_human}"]   = (old_sp, old_f0)
            chosen[f"young_{g_human}"] = (young_sp, young_f0)
            print(f"  {g_human}: young_F0={young_f0:.1f}Hz ({young_sp[:8]}...), "
                  f"old_F0={old_f0:.1f}Hz ({old_sp[:8]}...)")

        manifest[lang] = {}
        for profile, (sp, f0) in chosen.items():
            clips = per_spk[sp]
            indices = [c[0] for c in sorted(clips, key=lambda x: -x[1])][:8]  # longer clips first
            dur = materialise_ref(big_ds_train, indices, OUT / lang / profile)
            manifest[lang][profile] = {
                "speaker_id": sp,
                "median_f0_hz": round(f0, 1),
                "ref_duration_s": dur,
                "path": str((OUT / lang / profile / "ref.wav").relative_to(OUT)),
            }

    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[refs] DONE -> {OUT}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
