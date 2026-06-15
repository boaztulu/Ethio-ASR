#!/usr/bin/env python3
"""Build one HF DatasetDict per voice profile for ylacombe's fine-tune script.

For each of the 4 Amharic reference speakers (from Phase 1 manifest),
extract that speaker's WAXAL train+validation clips, romanize the text
with uroman, and save as a local DatasetDict at:

  models/datasets/amh/{profile}/
    train/
    validation/
    dataset_info.json

The trainer will then take dataset_name=this_path, text_column_name=text,
audio_column_name=audio, train_split_name=train, eval_split_name=validation.
"""
import json
import os
import sys
from pathlib import Path

PR = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
os.environ.setdefault("HF_HUB_CACHE", "/blue/rcstudents/btulu/Projects/Ethio-ASR/hf_cache/hub")

import numpy as np
from datasets import load_dataset, Audio, Dataset, DatasetDict
from uroman import Uroman

MANIFEST = json.load(open(PR / "reference_voices" / "manifest.json"))
OUT_BASE = PR / "models" / "voice_datasets"
OUT_BASE.mkdir(parents=True, exist_ok=True)

LANG = "amh"   # Phase 2 start with Amharic only
EVAL_FRAC = 0.10
MIN_DUR = 1.0
MAX_DUR = 20.0   # MMS-TTS prefers <=20s


def main():
    uroman = Uroman()
    print(f"[prep] loading WAXAL", flush=True)
    ds = load_dataset("badrex/waxalNLP-ethiopic-final",
                      verification_mode="no_checks")
    train, val = ds["train"], ds["validation"]

    # Build a speaker -> indices map across train + val for the target language
    splits = {"train": train, "validation": val}

    for profile, info in MANIFEST[LANG].items():
        spkr = info["speaker_id"]
        out_dir = OUT_BASE / LANG / profile
        if (out_dir / "train" / "dataset_info.json").exists() and \
           (out_dir / "validation" / "dataset_info.json").exists():
            print(f"[prep] {profile} already prepped, skipping")
            continue
        print(f"\n=== {LANG}/{profile} (speaker {spkr[:12]}...) ===", flush=True)

        all_rows = []
        for split_name, split_ds in splits.items():
            langs = split_ds["language"]
            spids = split_ds["speaker_id"]
            durs = split_ds["audio_duration"]
            texts = split_ds["transcription"]
            ids = split_ds["id"]

            indices = [i for i, (l, sp, d) in enumerate(zip(langs, spids, durs))
                       if (l or "").lower() == LANG and sp == spkr
                       and d is not None and MIN_DUR <= d <= MAX_DUR]
            print(f"[prep] {split_name}: {len(indices)} usable clips")
            if not indices:
                continue
            sub = split_ds.select(indices).cast_column(
                "audio", Audio(sampling_rate=16000))
            for ex in sub:
                ge = ex["transcription"]
                rom = uroman.romanize_string(ge)
                rom = " ".join(rom.split())   # collapse whitespace
                if not rom or len(rom) < 3:
                    continue
                all_rows.append({
                    "id": ex["id"],
                    "audio": ex["audio"],          # dict with array+sr
                    "text": rom,
                    "text_geez": ge,
                    "speaker_id": ex["speaker_id"] or "unknown",
                    "duration": ex["audio_duration"],
                })

        if not all_rows:
            print(f"[prep] {profile}: NO ROWS — skipping")
            continue

        # 90/10 train/val split
        rng = np.random.default_rng(seed=42)
        idx = np.arange(len(all_rows))
        rng.shuffle(idx)
        n_eval = max(5, int(EVAL_FRAC * len(all_rows)))
        eval_idx = set(idx[:n_eval].tolist())
        train_rows = [r for i, r in enumerate(all_rows) if i not in eval_idx]
        val_rows = [r for i, r in enumerate(all_rows) if i in eval_idx]
        print(f"[prep] {profile}: train={len(train_rows)} val={len(val_rows)}")

        ds_train = Dataset.from_list(train_rows).cast_column(
            "audio", Audio(sampling_rate=16000))
        ds_val = Dataset.from_list(val_rows).cast_column(
            "audio", Audio(sampling_rate=16000))
        dd = DatasetDict({"train": ds_train, "validation": ds_val})
        out_dir.mkdir(parents=True, exist_ok=True)
        # save_to_disk has been flaky earlier with multiprocessing; use single
        dd.save_to_disk(str(out_dir), num_proc=1)
        print(f"[prep] saved -> {out_dir}")

    print("\n[prep] DONE")


if __name__ == "__main__":
    main()
