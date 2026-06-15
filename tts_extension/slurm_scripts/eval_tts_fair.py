#!/usr/bin/env python3
"""Fair comparison: for the SAME N Amharic sentences:
  (A) recognise the REAL audio with our ASR -> WER_real (lower bound)
  (B) synthesize the text with MMS-TTS-Amh -> recognise -> WER_synth
The gap = TTS-induced intelligibility loss.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
os.environ.setdefault("HF_HUB_CACHE", "/blue/rcstudents/btulu/Projects/Ethio-ASR/hf_cache/hub")

sys.path.insert(0, "/blue/rcstudents/btulu/Projects/Ethio-ASR/slurm_scripts")
from evaluate_ctc import paper_postprocess, strip_lang_token  # type: ignore

import numpy as np
import torch
import soundfile as sf
import evaluate
from datasets import load_dataset, Audio
from transformers import VitsModel, AutoTokenizer, AutoProcessor, AutoModelForCTC
from uroman import Uroman


def asr_predict(audio_array, sr, proc, model, device, dtype):
    inputs = proc(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
    in_kwargs = {}
    if "input_features" in inputs:
        in_kwargs["input_features"] = inputs["input_features"].to(device, dtype=dtype)
    if "input_values" in inputs:
        in_kwargs["input_values"] = inputs["input_values"].to(device, dtype=dtype)
    if "attention_mask" in inputs:
        in_kwargs["attention_mask"] = inputs["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(**in_kwargs).logits
    pred = proc.batch_decode(logits.argmax(-1))[0]
    text, _ = strip_lang_token(pred)
    return paper_postprocess(text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts_model", default="facebook/mms-tts-amh")
    ap.add_argument("--asr_model", default="boazsew/Ethio-ASR-w2v-bert-2.0-uf")
    ap.add_argument("--n_samples", type=int, default=300)
    ap.add_argument("--min_dur", type=float, default=2.0)
    ap.add_argument("--max_dur", type=float, default=25.0)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[eval-fair] device={device} dtype={dtype}")

    print(f"[eval-fair] loading TTS {args.tts_model}")
    tts = VitsModel.from_pretrained(args.tts_model).to(device).eval()
    tts_tok = AutoTokenizer.from_pretrained(args.tts_model)
    uroman = Uroman()

    print(f"[eval-fair] loading ASR {args.asr_model}")
    asr_proc = AutoProcessor.from_pretrained(args.asr_model)
    asr_model = AutoModelForCTC.from_pretrained(args.asr_model, torch_dtype=dtype).to(device).eval()

    print(f"[eval-fair] loading WAXAL validation")
    ds = load_dataset("badrex/waxalNLP-ethiopic-final", split="validation",
                      verification_mode="no_checks")
    langs = ds["language"]; durs = ds["audio_duration"]; texts = ds["transcription"]
    candidates = [i for i, (l, d, t) in enumerate(zip(langs, durs, texts))
                  if (l or "").lower() == "amh"
                  and d is not None and args.min_dur <= d <= args.max_dur
                  and t and 4 <= len(t.split()) <= 80]
    print(f"[eval-fair] {len(candidates)} Amharic candidates in {args.min_dur}-{args.max_dur}s")

    rng = np.random.default_rng(42)
    pick = list(rng.choice(candidates, size=min(args.n_samples, len(candidates)),
                           replace=False))
    pick_sorted = sorted(pick)

    sub = ds.select(pick_sorted).cast_column("audio", Audio(sampling_rate=16000))

    wer_metric = evaluate.load("wer"); cer_metric = evaluate.load("cer")
    refs, hyps_real, hyps_synth = [], [], []
    synth_durs, real_durs = [], []

    t0 = time.time()
    for k, ex in enumerate(sub):
        ref_text = ex["transcription"]
        ref = paper_postprocess(ref_text.lower())

        # (A) Real audio
        wav_real = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr_real = ex["audio"]["sampling_rate"]
        hyp_real = asr_predict(wav_real, sr_real, asr_proc, asr_model, device, dtype)
        real_durs.append(len(wav_real)/sr_real)

        # (B) Synth audio
        rom = uroman.romanize_string(ref_text)
        inp = tts_tok(rom, return_tensors="pt").to(device)
        with torch.no_grad():
            wav_synth = tts(**inp).waveform.cpu().squeeze().float().numpy()
        hyp_synth = asr_predict(wav_synth, tts.config.sampling_rate, asr_proc, asr_model, device, dtype)
        synth_durs.append(len(wav_synth)/tts.config.sampling_rate)

        refs.append(ref); hyps_real.append(hyp_real); hyps_synth.append(hyp_synth)
        if (k+1) % 25 == 0:
            print(f"  {k+1}/{len(pick_sorted)}  ({(k+1)/(time.time()-t0):.1f}/s)")

    wer_real = wer_metric.compute(predictions=hyps_real, references=refs)
    cer_real = cer_metric.compute(predictions=hyps_real, references=refs)
    wer_synth = wer_metric.compute(predictions=hyps_synth, references=refs)
    cer_synth = cer_metric.compute(predictions=hyps_synth, references=refs)

    results = {
        "tts_model": args.tts_model,
        "asr_model": args.asr_model,
        "n_samples": len(refs),
        "duration_filter_s": [args.min_dur, args.max_dur],
        "real_audio":  {"wer": wer_real,  "cer": cer_real,  "mean_dur_s": float(np.mean(real_durs))},
        "synth_audio": {"wer": wer_synth, "cer": cer_synth, "mean_dur_s": float(np.mean(synth_durs))},
        "intelligibility_gap_wer": wer_synth - wer_real,
    }
    print("\n=== Fair TTS vs Real (Amharic, ASR-evaluated) ===")
    print(json.dumps(results, indent=2))
    out = args.out_json or str(PROJECT_ROOT / "results_fair_comparison.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[eval-fair] saved -> {out}")


if __name__ == "__main__":
    main()
