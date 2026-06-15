#!/usr/bin/env python3
"""Generate a final markdown report comparing all results.

Sources:
  - /blue/.../eval_results/paper_published/*.json   - paper's checkpoints (their numbers)
  - /blue/.../models/<exp>/eval_test.json           - our trained models
  - 2603.23654v1.pdf Table 3                       - paper-reported numbers (hardcoded)
"""
import glob
import json
import os
from pathlib import Path

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-ASR")
LANGS = ["amharic", "tigrinya", "oromo", "wolaytta", "sidaama"]

# Paper Table 3 - their reported WERs.
PAPER_TABLE_3 = {
    "Whisper-small (300M)":              [157.85, 174.31, 152.99, 184.67, 182.32],
    "Whisper-medium (786M)":             [195.05, 198.06, 192.64, 247.30, 232.91],
    "Whisper-large-v3 (1.6B)":           [153.03, 166.20, 128.65, 151.04, 144.08],
    "Seamless-M4T-v2 (2B)":              [103.75, 100.00, 100.00, 100.00, 100.00],
    "MMS-1B-all":                        [57.53, 70.48, 41.53, 104.22, 37.64],
    "OmniASR-CTC-300M v2":               [49.15, 58.11, 40.77, 52.90, 41.08],
    "OmniASR-CTC-1B v2":                 [37.44, 50.15, 31.34, 46.35, 37.26],
    "OmniASR-CTC-3B v2":                 [32.41, 45.91, 27.91, 43.44, 35.38],
    "OmniASR-CTC-7B v2":                 [32.48, 46.21, 27.79, 44.58, 35.21],
    "OmniASR-LLM-300M v2":               [30.95, 46.10, 27.33, 41.43, 34.10],
    "OmniASR-LLM-1B v2":                 [27.65, 42.87, 25.28, 40.37, 33.21],
    "OmniASR-LLM-3B v2":                 [26.83, 42.32, 24.80, 40.36, 32.91],
    "OmniASR-LLM-7B v2":                 [25.12, 40.69, 23.59, 39.22, 32.46],
    "Ethio-ASR (afrihubert, 94M)":       [30.95, 42.42, 27.57, 40.44, 34.02],
    "Ethio-ASR (mms-300m)":              [30.19, 41.62, 26.41, 39.10, 32.66],
    "Ethio-ASR (mms-1b)":                [26.14, 37.63, 23.69, 37.51, 31.02],
    "Ethio-ASR (w2v-bert-2.0, 600M)":    [22.92, 35.22, 24.44, 38.19, 31.65],
}


def load_jsons(pattern):
    out = {}
    for f in glob.glob(pattern):
        with open(f) as fh:
            out[Path(f).stem] = json.load(fh)
    return out


def percent(v):
    return f"{v*100:5.2f}" if isinstance(v, float) else str(v)


def main():
    print("# Ethio-ASR Reproduction & Improvement Report")
    print()
    print(f"Working dir: `{PROJECT_ROOT}`")
    print(f"Dataset: `badrex/waxalNLP-ethiopic-final`")
    print()

    # Paper-reported baselines (Table 3 of arXiv:2603.23654v1)
    print("## Paper Table 3 (reported in paper, for reference)")
    print()
    cols = ["Model"] + [l.title()[:3].upper() for l in LANGS] + ["Avg"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for name, wers in PAPER_TABLE_3.items():
        avg = sum(wers) / len(wers)
        print(f"| {name} | " + " | ".join(f"{w:.2f}" for w in wers) + f" | {avg:.2f} |")
    print()

    # Our re-eval of paper's published checkpoints
    print("## Our eval of paper's published checkpoints (sanity check vs Table 3)")
    print()
    published = load_jsons(str(PROJECT_ROOT / "eval_results" / "paper_published" / "*.json"))
    if published:
        print("| Model | " + " | ".join(l[:3].upper() for l in LANGS) + " | macro WER | micro WER |")
        print("|" + "|".join(["---"] * (len(LANGS) + 3)) + "|")
        for name, r in sorted(published.items()):
            row = [name]
            for lang in LANGS:
                w = r.get("per_language", {}).get(lang, {}).get("wer")
                row.append(percent(w))
            row.append(percent(r.get("averages", {}).get("macro_wer")))
            row.append(percent(r.get("averages", {}).get("micro_wer")))
            print("| " + " | ".join(row) + " |")
    else:
        print("_(no published eval results yet)_")
    print()

    # Our trained models
    print("## Our trained models")
    print()
    ours = {}
    for d in glob.glob(str(PROJECT_ROOT / "models" / "*")):
        for cand in [os.path.join(d, "eval_test.json"),
                     *glob.glob(os.path.join(d, "**", "eval_test.json"), recursive=True)]:
            if os.path.exists(cand):
                with open(cand) as f:
                    ours[Path(d).name] = json.load(f)
                break
    if ours:
        print("| Model | " + " | ".join(l[:3].upper() for l in LANGS) + " | macro WER | micro WER | LID acc |")
        print("|" + "|".join(["---"] * (len(LANGS) + 4)) + "|")
        for name, r in sorted(ours.items()):
            row = [name]
            for lang in LANGS:
                w = r.get("per_language", {}).get(lang, {}).get("wer")
                row.append(percent(w))
            row.append(percent(r.get("averages", {}).get("macro_wer")))
            row.append(percent(r.get("averages", {}).get("micro_wer")))
            lid = r.get("lid", {}).get("accuracy")
            row.append(percent(lid) if lid is not None else "-")
            print("| " + " | ".join(row) + " |")
    else:
        print("_(no training results yet)_")
    print()

    # Best so far
    if ours:
        best_name, best_macro = None, None
        for name, r in ours.items():
            mw = r.get("averages", {}).get("macro_wer")
            if mw is not None and (best_macro is None or mw < best_macro):
                best_macro, best_name = mw, name
        if best_name:
            paper_best = min(sum(v) / len(v) for v in PAPER_TABLE_3.values())
            print(f"## Best so far: **{best_name}** at macro WER {best_macro*100:.2f}%")
            print(f"Paper's best (any model in Table 3): {paper_best:.2f}%")


if __name__ == "__main__":
    main()
