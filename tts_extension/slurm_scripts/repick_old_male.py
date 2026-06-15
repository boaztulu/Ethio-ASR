#!/usr/bin/env python3
"""Re-pick an Amharic 'old_male' speaker with enough <20s clips for VITS
fine-tuning.

The original Phase-1 pick (Nr83c8HD...) had only 14 usable clips because
most of his recordings exceed our 20s training cap. We re-rank male
speakers by (clips with 1<=d<=20s), then within the top-30 pick the
one with the LOWEST median F0 (proxy for older voice).
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

PR = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
os.environ.setdefault("HF_HUB_CACHE", "/blue/rcstudents/btulu/Projects/Ethio-ASR/hf_cache/hub")

import numpy as np
import librosa
from datasets import load_dataset, Audio
from huggingface_hub import login

token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
if token:
    login(token=token, add_to_git_credential=False)


def estimate_f0(audio, sr=16000):
    f0, _, _ = librosa.pyin(audio.astype(np.float32),
                             fmin=70, fmax=300, sr=sr, frame_length=2048)
    voiced = f0[~np.isnan(f0)] if f0 is not None else None
    return float(np.median(voiced)) if voiced is not None and voiced.size else 0.0


def main():
    print("[repick] loading WAXAL")
    ds = load_dataset("badrex/waxalNLP-ethiopic-final",
                      verification_mode="no_checks")
    train, val = ds["train"], ds["validation"]

    male_speakers = defaultdict(list)   # spk -> list of (split, index, dur)
    for split_name, split_ds in (("train", train), ("validation", val)):
        for i, (l, sp, d, g) in enumerate(zip(
                split_ds["language"], split_ds["speaker_id"],
                split_ds["audio_duration"], split_ds["gender"])):
            if (l or "").lower() != "amh":
                continue
            if (g or "").lower() not in ("m", "male"):
                continue
            if d is None or not (1.5 <= d <= 20.0):
                continue
            if not sp:
                continue
            male_speakers[sp].append((split_name, i, d))

    # Rank by number of <20s clips
    ranked = sorted(male_speakers.items(),
                    key=lambda kv: -len(kv[1]))[:30]
    print(f"[repick] {len(ranked)} candidate male speakers (top 30 by clip count)")
    for sp, clips in ranked[:8]:
        total_min = sum(c[2] for c in clips) / 60
        print(f"  {sp[:12]}... {len(clips)} clips, {total_min:.1f} min")

    # Probe F0 for each: take 4 clips total ~8s
    print(f"\n[repick] estimating F0 (8s per speaker)...")
    spk_f0 = {}
    for sp, clips in ranked:
        # Take 4 short clips
        chosen = sorted(clips, key=lambda x: x[2])[:4]
        try:
            audio_bits = []
            for split_name, idx, _ in chosen:
                ds_for = train if split_name == "train" else val
                ex = ds_for.select([idx]).cast_column("audio", Audio(sampling_rate=16000))[0]
                audio_bits.append(np.asarray(ex["audio"]["array"], dtype=np.float32))
            audio = np.concatenate(audio_bits)
            f0 = estimate_f0(audio)
            spk_f0[sp] = (f0, len(clips))
            print(f"  {sp[:12]}... F0={f0:.1f} Hz, {len(clips)} clips")
        except Exception as e:
            print(f"  {sp[:12]}... F0 fail: {e}")

    # Pick speaker with LOWEST F0 (within the well-stocked top-30)
    ranked_f0 = sorted(spk_f0.items(), key=lambda kv: kv[1][0])
    print(f"\n[repick] lowest-F0 candidates:")
    for sp, (f0, n) in ranked_f0[:5]:
        print(f"  {sp[:12]}... F0={f0:.1f} Hz, {n} clips")

    chosen_sp = ranked_f0[0][0]
    chosen_f0 = ranked_f0[0][1][0]
    print(f"\n[repick] CHOSEN old_male: {chosen_sp} F0={chosen_f0:.1f} Hz")

    # Update manifest + write a fresh ref.wav
    manifest_path = PR / "reference_voices" / "manifest.json"
    manifest = json.load(open(manifest_path))
    manifest["amh"]["old_male"] = {
        "speaker_id": chosen_sp,
        "median_f0_hz": round(chosen_f0, 1),
        "ref_duration_s": None,   # filled below
        "path": "amh/old_male/ref.wav",
        "note": "re-picked for Phase 2 (more usable <20s clips)",
    }

    # Build ref.wav from this speaker's clips
    clips = male_speakers[chosen_sp]
    chosen_clips = sorted(clips, key=lambda x: -x[2])[:10]   # longer first
    refs = []
    acc = 0.0
    for split_name, idx, _ in chosen_clips:
        ds_for = train if split_name == "train" else val
        ex = ds_for.select([idx]).cast_column("audio", Audio(sampling_rate=16000))[0]
        wav = np.asarray(ex["audio"]["array"], dtype=np.float32)
        wav, _ = librosa.effects.trim(wav, top_db=30)
        if len(wav)/16000 < 1.0:
            continue
        peak = float(np.max(np.abs(wav))) or 1.0
        wav = wav * (0.95 / peak)
        refs.append(wav)
        acc += len(wav) / 16000
        if acc >= 30.0:
            break
    import soundfile as sf
    silence = np.zeros(int(0.3 * 16000), dtype=np.float32)
    audio = np.concatenate(sum(([c, silence] for c in refs), []))
    out_dir = PR / "reference_voices" / "amh" / "old_male"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "ref.wav"), audio, 16000)
    manifest["amh"]["old_male"]["ref_duration_s"] = round(len(audio)/16000, 2)
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"[repick] wrote ref.wav ({len(audio)/16000:.1f}s) + updated manifest")


if __name__ == "__main__":
    main()
