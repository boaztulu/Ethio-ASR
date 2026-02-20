#!/usr/bin/env python3
import os
import sys
import json
import argparse
import re

from datasets import load_dataset, load_from_disk
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import evaluate
from tqdm import tqdm

# add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from scripts.evaluate_finetuned_model_mutlilingual import process_text

# add project root for geez normalizer
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)
from post_processing.normalization import GeezNormalizer
geez_normalizer = GeezNormalizer()

lang2code = {
    "amh": "amh_Ethi",
    "tir": "tir_Ethi",
    "wal": "wal_Latn",
    "sid": "sid_Latn",
    "orm": "orm_Latn",
}

# languages that use ge'ez script
geez_langs = {"amh", "tir"}


def normalize(text: str, lang: str) -> str:
    """apply geez normalization for script languages, then process_text for all"""
    if lang in geez_langs:
        text = geez_normalizer.normalize(text)
    return process_text(text)


def compute_per_lang_metrics(predictions, references, languages, wer_metric, cer_metric):
    """compute per-language and macro average WER/CER"""
    lang_preds, lang_refs = {}, {}
    for pred, ref, lang in zip(predictions, references, languages):
        lang_preds.setdefault(lang, []).append(pred)
        lang_refs.setdefault(lang, []).append(ref)

    per_lang = {}
    for lang in lang_preds:
        lp = [p for p, r in zip(lang_preds[lang], lang_refs[lang]) if r]
        lr = [r for r in lang_refs[lang] if r]
        if not lp:
            continue
        per_lang[lang] = {
            'wer': wer_metric.compute(predictions=lp, references=lr),
            'cer': cer_metric.compute(predictions=lp, references=lr),
            'samples': len(lp)
        }

    macro_wer = sum(m['wer'] for m in per_lang.values()) / len(per_lang) if per_lang else 0
    macro_cer = sum(m['cer'] for m in per_lang.values()) / len(per_lang) if per_lang else 0
    return per_lang, macro_wer, macro_cer


def parse_args():
    parser = argparse.ArgumentParser(description="evaluate omni ASR model")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["omniASR_LLM_300M_v2", "omniASR_LLM_1B_v2", "omniASR_LLM_3B_v2"])
    parser.add_argument("--output_dir", type=str, default="transcripts/omni_asr")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_lid", action="store_true", help="pass language id to model")
    parser.add_argument("--debug", action="store_true", help="shuffle and take 100 samples for debugging")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    if os.path.exists(args.dataset_path):
        ds = load_from_disk(args.dataset_path)[args.split]
    else:
        ds = load_dataset(args.dataset_path)[args.split]

    print(f"loaded {len(ds)} samples (all languages)")

    if args.debug:
        ds = ds.shuffle(seed=42).select(range(100))
        print("debug mode: using 100 shuffled samples")

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"evaluating model: {model_name}")
        print(f"{'='*60}")

        pipeline = ASRInferencePipeline(model_card=model_name)

        results = {}
        predictions, references, languages = [], [], []

        for i, sample in tqdm(enumerate(ds), total=len(ds)):
            lang = sample["language"]
            audio = {
                "waveform": sample["audio"]["array"],
                "sample_rate": sample["audio"]["sampling_rate"]
            }

            # pass lang code to model only if use_lid is set
            lang_arg = [lang2code[lang]] if args.use_lid else None
            raw_pred = pipeline.transcribe([audio], lang=lang_arg, batch_size=args.batch_size)[0]

            pred = normalize(raw_pred, lang)
            ref = normalize(sample["transcription"], lang)

            results[sample.get("id", str(i))] = {
                "language": lang,
                "true_transcription": ref,
                "pred_transcription": pred,
            }

            print(f"sample {i} [{lang}]:")
            print(f"  prediction: {pred}")
            print(f"  reference:  {ref}")
            print("-" * 75)

            predictions.append(pred)
            references.append(ref)
            languages.append(lang)

        # filter empty refs
        filtered = [(p, r) for p, r in zip(predictions, references) if r]
        filtered_preds, filtered_refs = zip(*filtered) if filtered else ([], [])

        wer = wer_metric.compute(predictions=filtered_preds, references=filtered_refs)
        cer = cer_metric.compute(predictions=filtered_preds, references=filtered_refs)
        score = (1 - (0.5 * wer + 0.5 * cer)) * 100

        per_lang, macro_wer, macro_cer = compute_per_lang_metrics(
            predictions, references, languages, wer_metric, cer_metric
        )

        print(f"\nresults for {model_name}:")
        print(f"  micro WER: {wer*100:.2f}%")
        print(f"  micro CER: {cer*100:.2f}%")
        print(f"  macro WER: {macro_wer*100:.2f}%")
        print(f"  macro CER: {macro_cer*100:.2f}%")
        print(f"  score:     {score:.2f}%")
        print(f"\n  per-language:")
        for lang, m in per_lang.items():
            print(f"    {lang}: WER={m['wer']*100:.2f}%  CER={m['cer']*100:.2f}%  samples={m['samples']}")

        suffix = f"{model_name}_{args.split}{'_debug' if args.debug else ''}"

        # save transcriptions json
        json_path = os.path.join(args.output_dir, f"{suffix}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\ntranscriptions saved to {json_path}")

        # save metrics json
        metrics = {
            'model': model_name, 'split': args.split,
            'use_lid': args.use_lid, 'samples': len(predictions),
            'micro_wer': wer, 'micro_cer': cer,
            'macro_wer': macro_wer, 'macro_cer': macro_cer,
            'score': score, 'per_language': per_lang
        }
        metrics_path = os.path.join(args.output_dir, f"{suffix}.metrics")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()