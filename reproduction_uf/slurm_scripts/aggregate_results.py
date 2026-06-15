#!/usr/bin/env python3
"""Aggregate eval_*.json across all trained models into a paper-style WER table.

Output:
  - Markdown table (printed)
  - CSV (saved)
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path

LANG_ORDER = ["amharic", "tigrinya", "oromo", "wolaytta", "sidaama"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="/blue/rcstudents/btulu/Projects/Ethio-ASR/models")
    ap.add_argument("--out_csv", default="/blue/rcstudents/btulu/Projects/Ethio-ASR/results.csv")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.models_dir, "*"))):
        if not os.path.isdir(d):
            continue
        # Find eval json (could be at top or in 'best/' subdir)
        for fname in [f"eval_{args.split}.json"]:
            for cand in [os.path.join(d, fname), os.path.join(d, "best", fname),
                         *glob.glob(os.path.join(d, "*", fname))]:
                if os.path.exists(cand):
                    with open(cand) as f:
                        e = json.load(f)
                    name = os.path.basename(d)
                    row = {"model": name}
                    for lang in LANG_ORDER:
                        v = e.get("per_language", {}).get(lang, {}).get("wer")
                        row[f"WER_{lang}"] = f"{v*100:.2f}" if v is not None else "-"
                    row["WER_avg_macro"] = f"{e.get('averages', {}).get('macro_wer', 0)*100:.2f}"
                    row["WER_avg_micro"] = f"{e.get('averages', {}).get('micro_wer', 0)*100:.2f}"
                    row["LID_acc"] = f"{e.get('lid', {}).get('accuracy', 0)*100:.2f}" if e.get("lid") else "-"
                    rows.append(row)
                    break

    if not rows:
        print("No eval results found.")
        return

    # Markdown table
    cols = ["model"] + [f"WER_{l}" for l in LANG_ORDER] + ["WER_avg_macro", "WER_avg_micro", "LID_acc"]
    print("\n## Results table (WER %, lower is better)")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")

    # CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved CSV -> {args.out_csv}")


if __name__ == "__main__":
    main()
