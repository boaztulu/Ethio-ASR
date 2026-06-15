#!/usr/bin/env python3
"""Download the WAXAL Ethiopian dataset.

Two phases:
  1. snapshot_download with 8 parallel workers - fast, resumable
  2. load_dataset + save_to_disk for the training pipeline
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-ASR")
DATA_DIR = PROJECT_ROOT / "data" / "waxal_ethiopic"
DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "hf_cache" / "datasets"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "hf_cache" / "hub"))

from huggingface_hub import snapshot_download, login
from datasets import load_dataset

token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
if token:
    login(token=token, add_to_git_credential=False)
    print(f"[download] HF login OK (token {token[:8]}…)")

REPO = "badrex/waxalNLP-ethiopic-final"

# === Phase 1: parallel snapshot_download ===
print(f"[download] === Phase 1: parallel snapshot_download for {REPO} ===")
t0 = time.time()
snap_dir = snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    max_workers=8,
    allow_patterns=["*.parquet", "*.json", "*.md", ".gitattributes"],
)
print(f"[download] Snapshot complete in {time.time()-t0:.0f}s -> {snap_dir}")
os.system(f"du -sh {snap_dir}")
os.system(f"ls {snap_dir}/data | head -5; echo ... ; ls {snap_dir}/data | wc -l")

# === Phase 2: load_dataset + save_to_disk ===
# Now use load_dataset (will be cached so should be fast).  Saving to
# DatasetDict on disk gives a clean format our train_model.py expects when
# use_custom_dataset: true.
print(f"\n[download] === Phase 2: load_dataset + save_to_disk -> {DATA_DIR} ===")
t0 = time.time()
ds = load_dataset(REPO, verification_mode="no_checks")
print(f"[download] load_dataset in {time.time()-t0:.0f}s")

for split, d in ds.items():
    print(f"  {split}: {len(d)} samples")
    if d.column_names:
        print(f"    columns: {d.column_names}")

print(f"[download] Saving to {DATA_DIR} (num_proc=8)...")
t0 = time.time()
ds.save_to_disk(str(DATA_DIR), num_proc=8)
print(f"[download] save_to_disk in {time.time()-t0:.0f}s")

print(f"[download] Final on-disk size:")
os.system(f"du -sh {DATA_DIR}")
print("[download] DONE")
