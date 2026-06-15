#!/usr/bin/env python3
"""Push the 4 per-voice DatasetDicts to HF Hub as PRIVATE datasets."""
import os
import sys
from pathlib import Path

PR = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
os.environ.setdefault("HF_HUB_CACHE", str(PR / "hf_cache" / "hub"))

from datasets import DatasetDict
from huggingface_hub import HfApi, create_repo, login

token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
login(token=token, add_to_git_credential=False)

USER = "boazsew"
PROFILES = ["young_male", "young_female", "old_male", "old_female"]
LANG = "amh"

for prof in PROFILES:
    src = PR / "models" / "voice_datasets" / LANG / prof
    if not src.exists():
        print(f"[push] skip {prof}: no local data")
        continue
    repo = f"{USER}/waxal-{LANG}-{prof}"
    print(f"\n=== {repo} ===")
    dd = DatasetDict.load_from_disk(str(src))
    for k, ds in dd.items():
        print(f"  {k}: {len(ds)} examples, columns {ds.column_names[:6]}")
    create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
    dd.push_to_hub(repo, private=True)
    print(f"  -> https://huggingface.co/datasets/{repo}")

print("\nALL DONE")
