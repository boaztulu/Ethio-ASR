# src/training/metrics.py
from typing import Dict, List
import numpy as np


def compute_metrics(pred, processor, wer_metric, cer_metric):
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER) metrics.
    
    Args:
        pred: Prediction object containing predictions and label_ids
        processor: Wav2Vec2Processor for decoding predictions
        
    Returns:
        Dictionary of evaluation metrics
    """

    
    # Get logits and ids
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    # Replace padding tokens
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and references
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # print a few predicted samples and references
    for i in range(min(10, len(pred_str))):
        print(f"Sample {i}:")
        sample_wer = wer_metric.compute(predictions=[pred_str[i]], references=[label_str[i]])
        sample_cer = cer_metric.compute(predictions=[pred_str[i]], references=[label_str[i]])
        print(f"Prediction: {pred_str[i]}")
        print(f"Reference: {label_str[i]}")
        print(f"WER: {sample_wer*100:.4f}%")
        print(f"CER: {sample_cer*100:.4f}%")
        print("-"*75)
    
    # Compute metrics
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    # combined score    
    combined_error = (0.5 * wer) + (0.5 * cer)

    score = (1 - combined_error) * 100
    
    return {
        "wer": wer, 
        "cer": cer,
        "score": score,
    }