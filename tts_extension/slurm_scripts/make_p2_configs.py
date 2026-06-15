#!/usr/bin/env python3
"""Generate ylacombe-style training configs for 4 Amharic voice fine-tunes."""
import json
from pathlib import Path

PR = Path("/blue/rcstudents/btulu/Projects/Ethio-TTS")
CFG_DIR = PR / "configs" / "phase2"
CFG_DIR.mkdir(parents=True, exist_ok=True)

LANG = "amh"
PROFILES = ["young_male", "young_female", "old_male", "old_female"]
# Per-language test sentence used to monitor full-generation samples
ANCHOR_TEXT_ROMAN = "salaame lealame, atu degenta amaarenyaa nesh."

for prof in PROFILES:
    out = {
        # project metadata
        "project_name": f"mms-tts-amh-{prof}",
        "push_to_hub": False,
        "report_to": ["tensorboard"],
        "overwrite_output_dir": True,
        "output_dir": str(PR / "models" / "phase2" / f"amh_{prof}"),
        # data
        "dataset_name": f"boazsew/waxal-{LANG}-{prof}",
        "audio_column_name": "audio",
        "text_column_name": "text",
        "train_split_name": "train",
        "eval_split_name": "validation",
        "speaker_id_column_name": None,             # single-speaker
        "override_speaker_embeddings": True,
        "max_duration_in_seconds": 20.0,
        "min_duration_in_seconds": 1.0,
        "max_tokens_length": 500,
        "full_generation_sample_text": ANCHOR_TEXT_ROMAN,
        # model
        "model_name_or_path": str(PR / "models" / "mms-tts-with-disc" / LANG),
        # training
        "preprocessing_num_workers": 4,
        "do_train": True,
        "num_train_epochs": 200,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "per_device_train_batch_size": 16,
        "learning_rate": 2e-5,
        "adam_beta1": 0.8,
        "adam_beta2": 0.99,
        "warmup_ratio": 0.01,
        "group_by_length": False,
        # eval
        "do_eval": True,
        "eval_steps": 50,
        "per_device_eval_batch_size": 16,
        "max_eval_samples": 15,
        "do_step_schedule_per_epoch": True,
        # loss weights (ylacombe defaults)
        "weight_disc": 3,
        "weight_fmaps": 1,
        "weight_gen": 1,
        "weight_kl": 1.5,
        "weight_duration": 1,
        "weight_mel": 35,
        # precision
        "fp16": False,
        "bf16": True,
        "seed": 456,
    }
    path = CFG_DIR / f"{prof}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
