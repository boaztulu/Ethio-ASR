#!/usr/bin/env python3
"""Filter WAXAL Amharic data for TTS training.

Reads badrex/waxalNLP-ethiopic-final, keeps Amharic train+validation,
applies duration filter (1.5-12s ideal for TTS), groups by speaker_id,
and writes:
  - filtered_audio/<split>/<speaker_id>/<utt_id>.wav  (16 kHz mono)
  - filtered_audio/metadata_train.csv
  - filtered_audio/metadata_val.csv
  - filtered_audio/speaker_stats.json
"""
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
OUT_DIR = PROJECT_ROOT / "filtered_audio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import soundfile as sf
from datasets import load_dataset, Audio
from uroman import Uroman

MIN_DUR = 1.0
MAX_DUR = 25.0
MIN_SPEAKER_MINUTES = 3.0    # lower bar: many WAXAL speakers have only minutes each
TARGET_LANG = "amh"

print(f"[filter] loading badrex/waxalNLP-ethiopic-final", flush=True)
ds_full = load_dataset("badrex/waxalNLP-ethiopic-final",
                       verification_mode="no_checks")

uroman = Uroman()


def process_split(split: str):
    ds = ds_full[split]
    print(f"\n[filter] {split}: {len(ds)} total samples", flush=True)

    # Read columns we need WITHOUT decoding audio
    langs = ds["language"]
    speakers = ds["speaker_id"]
    durs = ds["audio_duration"]
    texts = ds["transcription"]
    ids = ds["id"]

    # First pass: find Amharic indices with duration in range
    amh_indices = [i for i, (l, d) in enumerate(zip(langs, durs))
                   if (l or "").lower() == TARGET_LANG
                   and d is not None and MIN_DUR <= d <= MAX_DUR]
    print(f"[filter] {split}: {len(amh_indices)} Amharic samples in {MIN_DUR}-{MAX_DUR}s", flush=True)

    # Compute speaker -> total seconds and decide which speakers to keep
    sp_totals = {}
    for i in amh_indices:
        sp = speakers[i] or "unknown"
        sp_totals[sp] = sp_totals.get(sp, 0.0) + durs[i]
    keep_speakers = {sp for sp, t in sp_totals.items()
                     if t >= MIN_SPEAKER_MINUTES * 60}
    print(f"[filter] {split}: {len(keep_speakers)} speakers with >={MIN_SPEAKER_MINUTES} min", flush=True)

    final_indices = [i for i in amh_indices if (speakers[i] or "unknown") in keep_speakers]
    print(f"[filter] {split}: keeping {len(final_indices)} samples after speaker filter", flush=True)

    if not final_indices:
        return [], {}

    # Materialise only the rows we'll keep
    print(f"[filter] {split}: materialising {len(final_indices)} rows + decoding audio", flush=True)
    sub = ds.select(final_indices).cast_column("audio", Audio(sampling_rate=16000))

    split_dir = OUT_DIR / split
    rows = []
    t0 = time.time()
    for j, ex in enumerate(sub):
        sp = ex.get("speaker_id") or "unknown"
        utt_id = ex["id"]
        text = ex["transcription"]
        rom = uroman.romanize_string(text)
        sp_dir = split_dir / sp
        sp_dir.mkdir(parents=True, exist_ok=True)
        wav_path = sp_dir / f"{utt_id}.wav"
        sf.write(str(wav_path), ex["audio"]["array"], 16000)
        rows.append({
            "utt_id": utt_id,
            "wav_path": str(wav_path.relative_to(OUT_DIR)),
            "speaker_id": sp,
            "duration": round(len(ex["audio"]["array"]) / 16000, 2),
            "text_geez": text,
            "text_roman": rom,
        })
        if j > 0 and j % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {j}/{len(final_indices)} ({j/elapsed:.1f}/s)", flush=True)

    # Speaker totals for this split
    sp_used = {}
    for r in rows:
        sp_used[r["speaker_id"]] = sp_used.get(r["speaker_id"], 0.0) + r["duration"]

    return rows, sp_used


train_rows, train_sp = process_split("train")
val_rows, val_sp = process_split("validation")

# Write metadata CSVs (LJSpeech-style)
import csv
for split_name, rows in [("train", train_rows), ("val", val_rows)]:
    path = OUT_DIR / f"metadata_{split_name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["wav_path", "speaker_id", "duration", "text_roman", "text_geez"])
        for r in rows:
            writer.writerow([r["wav_path"], r["speaker_id"], r["duration"],
                             r["text_roman"], r["text_geez"]])
    print(f"[filter] wrote {path} ({len(rows)} rows)")

# Speaker stats
total_sp = {}
for d in (train_sp, val_sp):
    for k, v in d.items():
        total_sp[k] = total_sp.get(k, 0.0) + v
ranked = sorted(total_sp.items(), key=lambda kv: -kv[1])
stats = {
    "n_speakers": len(total_sp),
    "total_hours": round(sum(total_sp.values()) / 3600, 2),
    "min_speaker_minutes_threshold": MIN_SPEAKER_MINUTES,
    "duration_filter_seconds": [MIN_DUR, MAX_DUR],
    "top_15_speakers": [{"speaker": sp, "minutes": round(s / 60, 2)}
                         for sp, s in ranked[:15]],
}
with open(OUT_DIR / "speaker_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"\n[filter] DONE")
print(f"  speakers kept: {stats['n_speakers']}")
print(f"  total hours:   {stats['total_hours']}")
if ranked:
    print(f"  top speaker:   {ranked[0][0]} ({ranked[0][1]/60:.1f} min)")
else:
    print("  no speakers kept - filter too strict")
