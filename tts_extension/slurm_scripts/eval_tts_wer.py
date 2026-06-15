#!/usr/bin/env python3
"""Evaluate a TTS model by synthesising sentences, running them through
our trained Amharic ASR, and computing WER against the original text.

This is the "round-trip" intelligibility test: lower WER => TTS is
producing speech that an ASR model trained on real Amharic recognises
as the original text.

Usage:
  python eval_tts_wer.py [--tts_model facebook/mms-tts-amh] \
                        [--asr_model boazsew/Ethio-ASR-w2v-bert-2.0-uf] \
                        [--n_samples 100] [--out_json results.json]
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")

import numpy as np
import torch
import soundfile as sf
import evaluate
from transformers import (
    VitsModel, AutoTokenizer,
    AutoProcessor, AutoModelForCTC,
)
from uroman import Uroman


# Paper-faithful post-processing for Amharic ASR output
sys.path.insert(0, "/blue/rcstudents/btulu/Projects/Ethio-ASR/slurm_scripts")
from evaluate_ctc import paper_postprocess, strip_lang_token  # type: ignore


def synth(tts_model, tokenizer, uroman, ge_text, device):
    rom = uroman.romanize_string(ge_text)
    inputs = tokenizer(rom, return_tensors="pt").to(device)
    with torch.no_grad():
        wav = tts_model(**inputs).waveform
    return wav.cpu().squeeze().float().numpy(), tts_model.config.sampling_rate, rom


def asr_recognize(audio_array, sr, asr_proc, asr_model, device, dtype):
    inputs = asr_proc(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
    in_kwargs = {}
    if "input_values" in inputs:
        in_kwargs["input_values"] = inputs["input_values"].to(device, dtype=dtype)
    if "input_features" in inputs:
        in_kwargs["input_features"] = inputs["input_features"].to(device, dtype=dtype)
    if "attention_mask" in inputs:
        in_kwargs["attention_mask"] = inputs["attention_mask"].to(device)
    with torch.no_grad():
        logits = asr_model(**in_kwargs).logits
    pred = asr_proc.batch_decode(logits.argmax(-1))[0]
    text, _ = strip_lang_token(pred)
    return paper_postprocess(text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts_model", default="facebook/mms-tts-amh")
    ap.add_argument("--asr_model", default="boazsew/Ethio-ASR-w2v-bert-2.0-uf")
    ap.add_argument("--n_samples", type=int, default=100,
                    help="Number of test sentences (random subset of validation)")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--save_audio_dir", default=None,
                    help="Optional: save synth wavs to this dir")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}")

    print(f"[eval] loading TTS: {args.tts_model}")
    tts = VitsModel.from_pretrained(args.tts_model).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.tts_model)
    uroman = Uroman()

    print(f"[eval] loading ASR: {args.asr_model}")
    asr_proc = AutoProcessor.from_pretrained(args.asr_model)
    asr_model = AutoModelForCTC.from_pretrained(
        args.asr_model, torch_dtype=torch.bfloat16).to(device).eval()
    dtype = torch.bfloat16

    # Load Amharic validation sentences
    from datasets import load_dataset
    os.environ["HF_HUB_CACHE"] = "/blue/rcstudents/btulu/Projects/Ethio-ASR/hf_cache/hub"
    print("[eval] loading WAXAL validation")
    ds = load_dataset("badrex/waxalNLP-ethiopic-final", split="validation",
                      verification_mode="no_checks")
    langs = ds["language"]
    durs = ds["audio_duration"]
    texts = ds["transcription"]
    # Pick Amharic samples with short-medium duration => their texts are TTS-friendly
    candidates = [i for i, (l, d, t) in enumerate(zip(langs, durs, texts))
                  if (l or "").lower() == "amh"
                  and d is not None and 3.0 <= d <= 12.0
                  and t and 5 <= len(t.split()) <= 40]
    print(f"[eval] {len(candidates)} eligible Amharic sentences")
    rng = np.random.default_rng(seed=42)
    pick = rng.choice(candidates, size=min(args.n_samples, len(candidates)),
                      replace=False)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    refs, hyps, durs_synth = [], [], []
    save_dir = Path(args.save_audio_dir) if args.save_audio_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for k, idx in enumerate(pick):
        ge_text = texts[int(idx)]
        ref = paper_postprocess(ge_text.lower())
        wav, sr, rom = synth(tts, tok, uroman, ge_text, device)
        if save_dir:
            sf.write(str(save_dir / f"synth_{k:03d}.wav"), wav, sr)
        hyp = asr_recognize(wav, sr, asr_proc, asr_model, device, dtype)
        refs.append(ref)
        hyps.append(hyp)
        durs_synth.append(len(wav) / sr)
        if k < 5:
            print(f"  [{k}] ge: {ge_text[:60]}...")
            print(f"  [{k}] hyp: {hyp[:60]}...")
        if (k + 1) % 20 == 0:
            print(f"  processed {k+1}/{len(pick)}  ({(k+1)/(time.time()-t0):.1f}/s)")

    wer = wer_metric.compute(predictions=hyps, references=refs)
    cer = cer_metric.compute(predictions=hyps, references=refs)

    results = {
        "tts_model": args.tts_model,
        "asr_model": args.asr_model,
        "n_samples": len(refs),
        "round_trip_wer": wer,
        "round_trip_cer": cer,
        "mean_synth_duration_s": float(np.mean(durs_synth)),
        "total_synth_duration_s": float(np.sum(durs_synth)),
    }
    print("\n=== Round-trip TTS eval ===")
    print(json.dumps(results, indent=2))

    out = args.out_json or str(PROJECT_ROOT / f"results_{args.tts_model.replace('/','_')}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] saved -> {out}")


if __name__ == "__main__":
    main()
