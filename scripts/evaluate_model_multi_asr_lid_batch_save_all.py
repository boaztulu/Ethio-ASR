#!/usr/bin/env python3

import os
import sys

# do this before importing torch
os.environ['NUMBA_CACHE_DIR'] = '/tmp/'
os.environ['NUMBA_DISABLE_JIT'] = '1'

from pathlib import Path
import torch
import json
import argparse

from string import punctuation
from transformers import AutoProcessor, AutoModelForCTC
from datasets import load_from_disk, Audio, Dataset, load_dataset
import jiwer
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import evaluate
import re
from typing import Dict, List, Any, Tuple
import logging


def load_env_file(env_path: str):
    """load environment variables from .env file"""
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value.strip("'").strip('"')

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
env_path = project_root / '.env'
load_env_file(env_path)


# add root directory to Python path
sys.path.insert(0, str(project_root))
logging.info(f"Project root added to Python path: {project_root}")


# Ge'ez Normalizer 
import post_processing.normalization
geez_normalizer = post_processing.normalization.GeezNormalizer()


# to map languages to integers for LID accuracy calculation
lang_mapping_dict = {
    '[AMH]': 0,
    '[TIR]': 1,
    '[ORM]': 2,
    '[SID]': 3,
    '[WAL]': 4,
    '[NO_LID_TOKEN]': 5,
}


def setup_logging():
    """configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


# def debug_tokenizer(processor):
#     """debug tokenizer information"""
#     print("=== TOKENIZER DEBUG ===")
#     print(f"tokenizer type: {type(processor.tokenizer)}")
#     if hasattr(processor.tokenizer, 'vocab_size'):
#         print(f"vocab size: {processor.tokenizer.vocab_size}")
#     if hasattr(processor.tokenizer, 'get_vocab'):
#         vocab = processor.tokenizer.get_vocab()
#         pad_tokens = [k for k, v in vocab.items() if 'pad' in k.lower()]
#         print(f"pad-related tokens: {pad_tokens[:5]}")
#     print("=====================")

def extract_lid_token(prediction: str) -> tuple[str, str]:
    """extract lid token and transcript from prediction."""
    
    match = re.search(r'\[[A-Z]+\]', prediction)
    if match:
        LID_token = match.group(0)
        transcript = prediction[match.end():].strip()
    else:
        LID_token = '[NO_LID_TOKEN]'
        transcript = prediction.strip()

    return LID_token, transcript


ModelComponents = Tuple[AutoModelForCTC, AutoProcessor, torch.device]


def load_model(model_path: str) -> ModelComponents:
    """load the trained ASR model and processor from local path"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    logging.info(f"loading model from {model_path}...")
    model = AutoModelForCTC.from_pretrained(model_path).to(device)
    
    model.eval()
    logging.info(f"model loaded on device: {device}")

    return model, processor, device


def transcribe_batch(audio_arrays: List, 
                     model: AutoModelForCTC, 
                     processor: AutoProcessor, 
                     device: torch.device) -> List[Tuple[str, str]]:
    """
    transcribe a batch of audio arrays, returns list of (LID_token, transcription)
    for each transcription.

    Args:
        audio_arrays: list of audio arrays
        model: ASR model
        processor: ASR processor
        device: device to run the model on
    Returns:
        list of (LID_token, transcription) tuples
    """

    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=False # 
    )
    
    # get a list of (LID_token, transcription) for each transcription
    LID_tokens_and_transcriptions = [extract_lid_token(t) for t in transcriptions]

    return LID_tokens_and_transcriptions


def evaluate_split(dataset: Dataset,
                   model: AutoModelForCTC,
                   processor: AutoProcessor,
                   device: torch.device,
                   split_name: str,
                   use_lid: bool,
                   batch_size: int = 16,
                   text_column: str = 'transcription') -> Dict[str, Any]:
    """
    evaluate a dataset split and calculate WER/CER.

    Args:
        dataset: dataset
        model: ASR model
        processor: ASR processor
        device: device to run the model on
    """
    eval_dataset = dataset[split_name]


    if not eval_dataset:
        print(f"no data found for {split_name} split.")
        return {}

    # shuffle dataset
    logging.info(f"shuffling {split_name} split...")
    eval_dataset = eval_dataset.shuffle(seed=42)

    # limit to first few samples for debugging
    num_samples = len(eval_dataset) 
    logging.info(f"limiting {split_name} split to first {num_samples} samples...")
    eval_dataset = eval_dataset.select(range(num_samples))

    print(f"processing {split_name} split. {len(eval_dataset)} samples were found.")

    # if split_name == 'validation':
    #     debug_tokenizer(processor)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    LID_accuracy_metric = evaluate.load("accuracy")

    raw_predictions, raw_references, predictions, references = [], [], [], []
    true_LID_tokens, pred_LID_tokens = [], []
    sample_ids, genders = [], []

    logging.info(f"resampling {split_name} split to 16kHz...")
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))

    logging.info(f"shuffling {split_name} split...")
    eval_dataset = eval_dataset.shuffle(seed=42)

    # batch processing
    for batch_start in tqdm(range(0, len(eval_dataset), batch_size),
                            desc=f"transcribing {split_name} split..."):
        batch = eval_dataset.select(range(batch_start, min(batch_start + batch_size, len(eval_dataset))))

        audio_arrays = [s['audio']['array'] for s in batch]
        results = transcribe_batch(audio_arrays, model, processor, device)

        for s, (pred_LID, raw_pred_transcript) in zip(batch, results):
            
            raw_predictions.append(raw_pred_transcript)
            raw_references.append(s[text_column])

            true_LID = '[' + s['language'].upper() + ']'

            pred_text = post_processing.normalization.process_text(raw_pred_transcript)
            raw_ref_text = s[text_column]
            ref_text = post_processing.normalization.process_text(raw_ref_text)

            # in case Ge'ez script, normalize the prediction and reference
            if true_LID in {'[AMH]', '[TIR]'}:
                pred_text = geez_normalizer.normalize(pred_text)
                ref_text = geez_normalizer.normalize(ref_text)

            # override lid if flag is off
            final_pred_LID = pred_LID if use_lid else '[NO_LID_TOKEN]'

            sample_ids.append(s['id'])
            genders.append(s.get('gender', 'unknown'))

            true_LID_tokens.append(lang_mapping_dict[true_LID])
            pred_LID_tokens.append(
                lang_mapping_dict.get(final_pred_LID, lang_mapping_dict['[NO_LID_TOKEN]'])
            )

            predictions.append(pred_text)
            references.append(ref_text)

    # filter empty refs
    filtered_preds = [p for p, r in zip(predictions, references) if r]
    filtered_refs = [r for r in references if r]
    filtered_raw_preds = [p for p, r in zip(raw_predictions, raw_references) if r]
    filtered_raw_refs = [r for r in raw_references if r]

    if not filtered_preds:
        print("no valid predictions found!")
        return {'split': split_name, 'wer': 1.0, 'cer': 1.0, 'score': 0.0,
                'lid_accuracy': 0.0, 'samples': len(eval_dataset),
                'raw_predictions': raw_predictions, 'raw_references': raw_references,
                'predictions': predictions, 'references': references,
                'sample_ids': sample_ids, 'genders': genders,
                'true_LID_tokens': true_LID_tokens, 'pred_LID_tokens': pred_LID_tokens}

    wer = wer_metric.compute(predictions=filtered_preds, references=filtered_refs)
    cer = cer_metric.compute(predictions=filtered_preds, references=filtered_refs)
    error_rate = (0.5 * wer) + (0.5 * cer)
    score = (1 - error_rate) * 100

    LID_accuracy = LID_accuracy_metric.compute(
        predictions=pred_LID_tokens, references=true_LID_tokens
    )

    # per-language metrics
    id_to_lang = {v: k for k, v in lang_mapping_dict.items()}
    lang_preds, lang_refs = {}, {}
    
    for pred, ref, lang_id in zip(predictions, references, true_LID_tokens):
        lang = id_to_lang[lang_id]
        lang_preds.setdefault(lang, []).append(pred)
        lang_refs.setdefault(lang, []).append(ref)

    per_lang_metrics = {}

    for lang in lang_preds:
        lang_pred_texts = [p for p, r in zip(lang_preds[lang], lang_refs[lang]) if r]
        lang_ref_texts = [r for r in lang_refs[lang] if r]
        
        if not lang_pred_texts:
            continue

        lang_wer = wer_metric.compute(predictions=lang_pred_texts, references=lang_ref_texts)
        lang_cer = cer_metric.compute(predictions=lang_pred_texts, references=lang_ref_texts)

        per_lang_metrics[lang] = {'wer': lang_wer, 'cer': lang_cer, 'samples': len(lang_pred_texts)}

    macro_wer = sum(m['wer'] for m in per_lang_metrics.values()) / len(per_lang_metrics)
    macro_cer = sum(m['cer'] for m in per_lang_metrics.values()) / len(per_lang_metrics)

    print(f"\n{split_name} results:")
    print(f"  micro WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"  micro CER: {cer:.4f} ({cer*100:.2f}%)")
    print(f"  macro WER: {macro_wer:.4f} ({macro_wer*100:.2f}%)")
    print(f"  macro CER: {macro_cer:.4f} ({macro_cer*100:.2f}%)")
    print(f"  score: {score:.2f}%")
    print(f"  LID Accuracy: {LID_accuracy['accuracy']*100:.2f}%")
    print(f"\n  per-language metrics:")
    for lang, m in per_lang_metrics.items():
        print(f"    {lang}: WER={m['wer']*100:.2f}%  CER={m['cer']*100:.2f}%  samples={m['samples']}")

    return {
        'split': split_name, 'wer': wer, 'cer': cer, 'score': score,
        'macro_wer': macro_wer, 'macro_cer': macro_cer,
        'per_lang_metrics': per_lang_metrics,
        'lid_accuracy': LID_accuracy['accuracy'], 'samples': len(eval_dataset),
        'predictions': filtered_preds, 'references': filtered_refs,
        'raw_predictions': filtered_raw_preds, 'raw_references': filtered_raw_refs,
        'sample_ids': sample_ids, 'genders': genders,
        'true_LID_tokens': true_LID_tokens, 'pred_LID_tokens': pred_LID_tokens
    }


def save_transcriptions(result: Dict, dataset: Dataset, split_name: str,
                        use_lid: bool, experiment_name: str):
    """save transcriptions and lid predictions to json"""
    id_to_char = {v: k for k, v in lang_mapping_dict.items()}
    output = {}

    for i, sample_id in enumerate(result['sample_ids']):
        true_lang = id_to_char[result['true_LID_tokens'][i]]
        pred_lang = id_to_char[result['pred_LID_tokens'][i]] if use_lid else 'NO_LID_TOKEN'
        output[sample_id + '_' + true_lang.lower()] = {
            "true_language": true_lang,
            "gender": result['genders'][i],
            "pred_language": pred_lang,
            "true_transcription": result['references'][i],
            "pred_transcription": result['predictions'][i],
            "raw_pred_transcription": result['raw_predictions'][i],
            "raw_ref_transcription": result['raw_references'][i],
        }

    out_path = Path(f"json_outputs_fleurs/{experiment_name}_{split_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logging.info(f"transcriptions saved to {out_path}")


def save_metrics(result: Dict, experiment_name: str, split_name: str):
    """save evaluation metrics to json next to the transcriptions file"""
    metrics = {
        'split': result['split'],
        'samples': result['samples'],
        'micro_wer': result['wer'],
        'micro_cer': result['cer'],
        'macro_wer': result['macro_wer'],
        'macro_cer': result['macro_cer'],
        'score': result['score'],
        'lid_accuracy': result['lid_accuracy'],
        'per_language': {
            lang: {'wer': m['wer'], 'cer': m['cer'], 'samples': m['samples']}
            for lang, m in result['per_lang_metrics'].items()
        }
    }
    out_path = Path(f"json_outputs_fleurs/{experiment_name}_{split_name}.metrics")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info(f"metrics saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="evaluate ASR model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--use_lid", action="store_true", help="enable language id prediction")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--text_column", type=str, default="transcription")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    logging.info("starting evaluation script...")

    logging.info(f"loading dataset from {args.dataset_path}...")

    # try to read from disk if it's a local path, otherwise load from HF hub
    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)

    logging.info("loading trained model...")
    model, processor, device = load_model(args.model_path)
    logging.info(f"model loaded on device: {device}")

    result = evaluate_split(
        dataset, model, processor, device,
        args.split, args.use_lid, args.batch_size, args.text_column
    )

    save_transcriptions(result, dataset, args.split, args.use_lid, args.experiment_name)
    save_metrics(result, args.experiment_name, args.split)

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    summary_df = pd.DataFrame([{
        'Split': result['split'],
        'Samples': result['samples'],
        'micro WER (%)': f"{result['wer']*100:.2f}%",
        'micro CER (%)': f"{result['cer']*100:.2f}%",
        'macro WER (%)': f"{result['macro_wer']*100:.2f}%",
        'macro CER (%)': f"{result['macro_cer']*100:.2f}%",
        'Score (%)': f"{result['score']:.2f}%",
        'LID Accuracy (%)': f"{result['lid_accuracy']*100:.2f}%",
    }])
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()