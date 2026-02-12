# src/training/metrics.py
from typing import Dict, List
import numpy as np

def preprocess_logits_for_metrics(logits, labels):
    # only keep predicted token ids, not full logit tensor
    return logits.argmax(dim=-1)

def compute_metrics(pred, processor, wer_metric, cer_metric):
    # get logits and ids
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # replace padding tokens
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # decode predictions and references
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # compute metrics once (batch)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    # debug: print a few samples with manually computed WER/CER
    for i in range(min(10, len(pred_str))):
        # compute per-sample metrics without using the evaluate module
        sample_wer = _simple_wer(pred_str[i], label_str[i])
        sample_cer = _simple_cer(pred_str[i], label_str[i])
        print(f"Sample {i}:")
        print(f"Prediction: {pred_str[i]}")
        print(f"Reference: {label_str[i]}")
        print(f"WER: {sample_wer*100:.4f}%")
        print(f"CER: {sample_cer*100:.4f}%")
        print("-"*75)

    combined_error = (0.5 * wer) + (0.5 * cer)
    score = (1 - combined_error) * 100

    return {
        "wer": wer,
        "cer": cer,
        "score": score,
    }

def _simple_wer(pred: str, ref: str) -> float:
    # simple wer calculation for debug purposes
    pred_words = pred.split()
    ref_words = ref.split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    distance = _levenshtein(pred_words, ref_words)
    return distance / len(ref_words)

def _simple_cer(pred: str, ref: str) -> float:
    # simple cer calculation for debug purposes
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    distance = _levenshtein(list(pred), list(ref))
    return distance / len(ref)

def _levenshtein(a: list, b: list) -> int:
    # levenshtein distance
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = range(len(b) + 1)
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]