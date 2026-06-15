#!/usr/bin/env python3
"""Fine-tune Whisper-large-v3 with LoRA on the WAXAL Ethiopian dataset.

This is the encoder-decoder candidate (vs CTC) - the paper's Whisper baselines
were zero-shot only and performed poorly. We expect fine-tuning to dramatically
improve WER, potentially below the paper's 30.48% best.
"""
import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-ASR")
load_dotenv(PROJECT_ROOT / ".env")

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "hf_cache" / "hub"))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

import torch
import numpy as np
import evaluate
from datasets import DatasetDict, Audio, load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============ Config ============
MODEL_NAME = "openai/whisper-large-v3"
DATA_REPO = "badrex/waxalNLP-ethiopic-final"
OUTPUT_DIR = PROJECT_ROOT / "models" / "whisper-large-v3-lora-ethio"
LANG_TO_WHISPER = {
    "amharic": "amharic",
    "tigrinya": "amharic",   # not in whisper; closest related script
    "oromo": "swahili",      # not in whisper; another African
    "sidaama": "swahili",
    "wolaytta": "swahili",
}

PER_DEVICE_BATCH = 8
GRAD_ACCUM = 4              # effective batch 32
LR = 1e-4
MAX_STEPS = 30000
WARMUP = 3000
EVAL_STEPS = 1000
SAVE_STEPS = 1000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ Data ============
log.info(f"Loading dataset from HF: {DATA_REPO}")
ds = load_dataset(DATA_REPO, verification_mode="no_checks")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
# Sample a smaller eval set so training is fast
ds["validation"] = ds["validation"].shuffle(seed=42).select(range(2000))

log.info(f"Train: {len(ds['train'])}  Validation: {len(ds['validation'])}")

# ============ Model / Processor ============
log.info(f"Loading processor from {MODEL_NAME}")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)

log.info(f"Loading model from {MODEL_NAME}")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)

# generation defaults
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.suppress_tokens = []

# LoRA
log.info("Wrapping with LoRA")
lora_cfg = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.05, bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ============ Preprocessing ============
# Set prefix tokens ONCE before mapping (set_prefix_tokens mutates state and
# races with num_proc>1).  Use amharic since it's the only Whisper-supported
# Ethiopian language; for others we tag at runtime via input language tokens
# embedded in the text targets.
processor.tokenizer.set_prefix_tokens(language="amharic", task="transcribe")

LANG_TAG_MAP = {"amh": "amharic", "tir": "tigrinya",
                "orm": "oromo", "wal": "wolaytta", "sid": "sidaama"}

def prepare(batch):
    audio = batch["audio"]
    inputs = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    )
    batch["input_features"] = inputs.input_features[0]
    # Prepend a language tag inside the text so decoder learns LID jointly.
    lang_code = batch.get("language", "amh").lower()
    lang_word = LANG_TAG_MAP.get(lang_code, "amharic")
    text = batch.get("transcription", batch.get("transcript", batch.get("text", "")))
    text = f"<{lang_word}> {text}"
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch

log.info("Encoding datasets (single-proc to avoid tokenizer races)")
train_ds = ds["train"].map(prepare, num_proc=1,
                           remove_columns=ds["train"].column_names,
                           desc="encode train")
eval_ds = ds["validation"].map(prepare, num_proc=1,
                               remove_columns=ds["validation"].column_names,
                               desc="encode val")

# ============ Collator ============
class WhisperCollator:
    def __init__(self, processor):
        self.p = processor
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.p.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.p.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Don't include leading BOS in labels
        if (labels[:, 0] == self.p.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

collator = WhisperCollator(processor)

# ============ Metrics ============
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer, "score": (1 - (wer + cer) / 2) * 100}

# ============ Training ============
args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=WARMUP,
    max_steps=MAX_STEPS,
    gradient_checkpointing=True,
    bf16=True, fp16=False,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    logging_steps=25,
    report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
    predict_with_generate=True,
    generation_max_length=225,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="score",
    greater_is_better=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

log.info("Starting Whisper LoRA fine-tuning")
trainer.train()

log.info(f"Saving final model + processor to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR / "final")
processor.save_pretrained(OUTPUT_DIR / "final")

log.info("Final evaluation")
metrics = trainer.evaluate()
log.info(f"Final: {metrics}")
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
log.info("Done")
