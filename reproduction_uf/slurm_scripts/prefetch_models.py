#!/usr/bin/env python3
"""Pre-download all pretrained models we'll use, so training jobs don't waste GPU time on downloads."""
import os
from pathlib import Path

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-ASR")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "hf_cache" / "hub"))

from huggingface_hub import snapshot_download, login

token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
if token:
    login(token=token, add_to_git_credential=False)

MODELS = [
    # Paper baselines (4 models)
    "ajesujoba/AfriHuBERT",            # 94M
    "facebook/mms-300m",               # 300M
    "facebook/mms-1b",                 # 1B
    "facebook/w2v-bert-2.0",           # 600M
    # Candidate improvements (3 new models)
    "facebook/wav2vec2-xls-r-1b",      # 1B - alternative to MMS-1B
    "openai/whisper-large-v3",         # 1.6B - encoder-decoder for FT
    # 3rd candidate = retrain w2v-bert-2.0 with tuned LR, no new download needed
    # Also fetch paper's published checkpoints for direct comparison
    "badrex/Ethio-ASR-multilingual-94M",
    "badrex/Ethio-ASR-multilingual-300M",
    "badrex/Ethio-ASR-multilingual-1B",
    "badrex/Ethio-ASR-multilingual-600M",
]

for m in MODELS:
    print(f"\n=== Fetching {m} ===")
    try:
        path = snapshot_download(
            repo_id=m,
            allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors", "*.model", "*tokenizer*", "*processor*", "*.npy"],
            ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*"],  # skip non-PyTorch
        )
        print(f"  -> {path}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n[done] All models prefetched.")
os.system(f"du -sh {PROJECT_ROOT / 'hf_cache'}")
