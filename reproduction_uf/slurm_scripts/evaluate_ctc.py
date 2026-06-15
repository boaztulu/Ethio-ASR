#!/usr/bin/env python3
"""Evaluate a trained CTC model on the WAXAL test split.

Computes WER and CER per language and overall (matching Paper Table 3).
Also evaluates LID accuracy if model emits language tokens (Table 5).

Usage:
  python evaluate_ctc.py --model_dir <path-to-trained-model> [--dataset <path>] [--split test] [--out_json <path>]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-ASR")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "hf_cache" / "hub"))

import torch
import numpy as np
import evaluate
from datasets import DatasetDict, Audio, load_dataset
from transformers import AutoProcessor, AutoModelForCTC, AutoFeatureExtractor


LANGUAGES = ["amharic", "tigrinya", "oromo", "wolaytta", "sidaama"]

# Same character cleanup approach as the paper's preprocessing.py (lowercase + NFC)
LANG_TOKEN_RE = re.compile(r"\[(?:amharic|tigrinya|oromo|wolaytta|sidaama|amh|tir|orm|wal|sid)\]\|?", re.IGNORECASE)


# ===== Paper post-processing (Section 5.2) =====
# "we apply punctuation removal and normalize homophones in the Ge'ez script"
# Following Nigatu et al. EMNLP 2025 [ref 32] - the standard homophone
# normalization used in Ethiopic NLP.  Each row maps a homophone family to
# a single canonical form (column 1 of each row).
_GEEZ_HOMOPHONE_GROUPS = [
    # ha-series (ሀ, ሐ, ኀ are all /ha/-like)
    "ሀሐኀ", "ሁሑኁ", "ሂሒኂ", "ሃሓኃ", "ሄሔኄ", "ህሕኅ", "ሆሖኆ",
    # sa-series (ሰ vs ሠ)
    "ሰሠ", "ሱሡ", "ሲሢ", "ሳሣ", "ሴሤ", "ስሥ", "ሶሦ", "ሷሧ",
    # a-series (አ vs ዐ)
    "አዐ", "ኡዑ", "ኢዒ", "ኣዓ", "ኤዔ", "እዕ", "ኦዖ",
    # ts'a-series (ጸ vs ፀ)
    "ጸፀ", "ጹፁ", "ጺፂ", "ጻፃ", "ጼፄ", "ጽፅ", "ጾፆ",
]
_HOMOPHONE_TABLE = str.maketrans(
    {ch: grp[0] for grp in _GEEZ_HOMOPHONE_GROUPS for ch in grp[1:]}
)

# Ethiopic + Latin punctuation to strip after normalization
_PUNCT_TABLE = str.maketrans(
    "",
    "",
    "።፣፤፥፦፧፨፠፡፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼"
    ".,;:?!\"'()[]{}\\/|<>-_=+*&^%$#@~`",
)


def paper_postprocess(text: str) -> str:
    """Apply paper's eval-time post-processing (Section 5.2)."""
    # Homophone normalization (Ge'ez)
    text = text.translate(_HOMOPHONE_TABLE)
    # Punctuation removal
    text = text.translate(_PUNCT_TABLE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_lang_token(text: str) -> tuple[str, str | None]:
    """Strip a leading [LANG] token from text; return (clean_text, predicted_lang_or_None)."""
    text = text.strip()
    m = re.match(r"\[(amharic|tigrinya|oromo|wolaytta|sidaama|amh|tir|orm|wal|sid)\]\|?\s*", text, re.IGNORECASE)
    if m:
        lang_token = m.group(1).lower()
        if len(lang_token) == 3:
            lang_token = {"amh":"amharic","tir":"tigrinya","orm":"oromo","wal":"wolaytta","sid":"sidaama"}[lang_token]
        return text[m.end():].strip(), lang_token
    return text, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="Path to trained model directory (contains model + processor)")
    ap.add_argument("--dataset", default="badrex/waxalNLP-ethiopic-final",
                    help="HF dataset name OR local DatasetDict path")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Limit eval samples (0 = all)")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--apply_postproc", action="store_true",
                    help="Apply paper's post-processing (punct removal + Ge'ez homophone norm)")
    args = ap.parse_args()

    out_json = args.out_json or os.path.join(args.model_dir, f"eval_{args.split}.json")
    print(f"[eval] model_dir   = {args.model_dir}")
    print(f"[eval] dataset     = {args.dataset}")
    print(f"[eval] split       = {args.split}")
    print(f"[eval] out_json    = {out_json}")

    # === Load model + processor ===
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForCTC.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    model.to(args.device).eval()

    # === Load eval dataset ===
    if os.path.isdir(args.dataset):
        ds = DatasetDict.load_from_disk(args.dataset)
    else:
        ds = load_dataset(args.dataset, verification_mode="no_checks")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = ds[args.split]
    if args.max_samples > 0:
        eval_ds = eval_ds.select(range(args.max_samples))
    print(f"[eval] Evaluating on {len(eval_ds)} samples")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    preds_per_lang = defaultdict(list)
    refs_per_lang = defaultdict(list)
    lid_correct = 0
    lid_total = 0

    # === Inference loop ===
    n = 0
    for i in range(0, len(eval_ds), args.batch_size):
        batch = eval_ds[i:i + args.batch_size]
        audio = [a["array"] for a in batch["audio"]]
        sr = batch["audio"][0]["sampling_rate"]
        text_ref = batch.get("transcription", batch.get("transcript", batch.get("text", [""] * len(audio))))
        lang_ref = batch.get("language", ["unknown"] * len(audio))

        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        # Move to device + bfloat16 features
        in_kwargs = {}
        if "input_values" in inputs:
            in_kwargs["input_values"] = inputs["input_values"].to(args.device, dtype=torch.bfloat16)
        if "input_features" in inputs:
            in_kwargs["input_features"] = inputs["input_features"].to(args.device, dtype=torch.bfloat16)
        if "attention_mask" in inputs:
            in_kwargs["attention_mask"] = inputs["attention_mask"].to(args.device)

        with torch.no_grad():
            logits = model(**in_kwargs).logits
        pred_ids = logits.argmax(dim=-1)
        pred_str = processor.batch_decode(pred_ids)

        for p, r, l in zip(pred_str, text_ref, lang_ref):
            pred_clean, pred_lang = strip_lang_token(p)
            # Also strip LID token from references if they have one (training format does this)
            ref_clean, _ = strip_lang_token(r)
            # apply same lowercase as preprocess
            ref_clean = ref_clean.lower()
            pred_clean = pred_clean.lower()
            # Paper post-processing: punctuation removal + Ge'ez homophone norm
            if args.apply_postproc:
                pred_clean = paper_postprocess(pred_clean)
                ref_clean = paper_postprocess(ref_clean)
            lang_key = l.lower() if isinstance(l, str) else "unknown"
            preds_per_lang[lang_key].append(pred_clean)
            refs_per_lang[lang_key].append(ref_clean)
            if pred_lang is not None:
                lid_total += 1
                if pred_lang == lang_key:
                    lid_correct += 1

        n += len(audio)
        if n % 200 < args.batch_size:
            print(f"[eval] processed {n}/{len(eval_ds)}")

    # === Per-language WER / CER ===
    results = {"per_language": {}, "averages": {}, "lid": {}}
    micro_pairs = []
    macro_wer, macro_cer = [], []
    for lang in sorted(preds_per_lang):
        preds = preds_per_lang[lang]
        refs = refs_per_lang[lang]
        if not preds:
            continue
        wer = wer_metric.compute(predictions=preds, references=refs)
        cer = cer_metric.compute(predictions=preds, references=refs)
        results["per_language"][lang] = {"wer": wer, "cer": cer, "n": len(preds)}
        print(f"[eval] {lang:10s}  n={len(preds):5d}  WER={wer*100:6.2f}%  CER={cer*100:6.2f}%")
        macro_wer.append(wer)
        macro_cer.append(cer)
        micro_pairs.extend(zip(preds, refs))

    # Average across languages
    if macro_wer:
        results["averages"]["macro_wer"] = sum(macro_wer) / len(macro_wer)
        results["averages"]["macro_cer"] = sum(macro_cer) / len(macro_cer)
    if micro_pairs:
        mp, mr = zip(*micro_pairs)
        results["averages"]["micro_wer"] = wer_metric.compute(predictions=list(mp), references=list(mr))
        results["averages"]["micro_cer"] = cer_metric.compute(predictions=list(mp), references=list(mr))

    if lid_total > 0:
        results["lid"] = {"accuracy": lid_correct / lid_total, "n": lid_total}
        print(f"[eval] LID accuracy: {lid_correct}/{lid_total} = {lid_correct/lid_total*100:.2f}%")

    print("[eval] === Final averages ===")
    for k, v in results["averages"].items():
        print(f"  {k}: {v*100:.2f}%" if isinstance(v, float) else f"  {k}: {v}")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[eval] Saved -> {out_json}")


if __name__ == "__main__":
    main()
