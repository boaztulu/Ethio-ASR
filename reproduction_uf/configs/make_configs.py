#!/usr/bin/env python3
"""Generate all training configs from a single source of truth.

Run this once to regenerate configs/*.yaml.
"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path("/blue/rcstudents/btulu/Projects/Ethio-ASR")
CFG_DIR = PROJECT_ROOT / "configs"
# Use HF hub directly - parquet files are cached in $HF_HOME/hub.
# Avoids fragile save_to_disk multiprocess pickling on this cluster.
DATASET_PATH = "badrex/waxalNLP-ethiopic-final"
USE_CUSTOM_DATASET = False
OUTPUT_DIR = str(PROJECT_ROOT / "models")

# Full character set from the paper (407 chars).  Single source of truth.
CHARACTER_SET = (
    " !#$%&'*+,-.0123456789=?@abcdefghijklmnopqrstuvwxyz"
    "ሀሁሂሃሄህሆሇለሉሊላሌልሎሏሐሑሒሓሔሕሖሗመሙሚማሜምሞሟሠሡሢሣሤሥሦሧረሩሪራሬርሮሯ"
    "ሰሱሲሳሴስሶሷሸሹሺሻሼሽሾሿቀቁቂቃቄቅቆቇቈቊቋቌቍቐቑቒቓቔቕቖቘቚቛቜቝበቡቢባቤብቦቧቨቩቪቫቬቭቮቯ"
    "ተቱቲታቴትቶቷቸቹቺቻቼችቾቿኀኁኂኃኄኅኆኇኈኊኋኌኍነኑኒናኔንኖኗኘኙኚኛኜኝኞኟአኡኢኣኤእኦኧ"
    "ከኩኪካኬክኮኯኰኲኳኴኵኸኹኺኻኼኽኾዀዂዃዄዅወዉዊዋዌውዎዏዐዑዒዓዔዕዖዘዙዚዛዜዝዞዟዠዡዢዣዤዥዦዧ"
    "የዩዪያዬይዮዯደዱዲዳዴድዶዷዸዹዺዻዼዽዾዿጀጁጂጃጄጅጆጇገጉጊጋጌግጎጏጐጒጓጔጕጘጙጚጛጜጝጞጟ"
    "ጠጡጢጣጤጥጦጧጨጩጪጫጬጭጮጯጰጱጲጳጴጵጶጷጸጹጺጻጼጽጾጿፀፁፂፃፄፅፆፇፈፉፊፋፌፍፎፏፐፑፒፓፔፕፖፗፘፙፚ"
    "፠፡።፣፤፥፦፧፨፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼€"
)
assert len(CHARACTER_SET) == 407, f"got {len(CHARACTER_SET)}"


def base_cfg():
    """Common settings shared by all configs."""
    return {
        "project": "Ethio-ASR-reproduction",
        "output_dir": OUTPUT_DIR,
        "seed": 42,
        # Model
        "freeze_feature_encoder": True,
        "add_final_layer_adapter": True,
        # Training (will be overridden per-model)
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "num_epochs": 7,
        "max_steps": 36800,
        "learning_rate": 3e-5,
        "warmup_ratio": 0.1,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
        "save_steps": 800,
        "eval_steps": 800,
        "logging_steps": 25,
        "save_total_limit": 2,
        # Data
        "use_custom_dataset": USE_CUSTOM_DATASET,
        "dataset_path": DATASET_PATH,
        "train_split": "train",
        "eval_split": "validation",
        "language": "all",
        "sample": False,
        "sample_size": 197634,
        # Text
        "add_language_tokens": True,
        "apply_accent_replacements": True,
        "character_set": CHARACTER_SET,
    }


def cfg(name, pretrained_model, lr, *, bf16=True, batch_size=4, grad_accum=8,
        max_steps=36800, add_adapter=True):
    c = base_cfg()
    c["pretrained_model"] = pretrained_model
    c["learning_rate"] = lr
    c["batch_size"] = batch_size
    c["gradient_accumulation_steps"] = grad_accum
    c["bf16"] = bf16
    c["fp16"] = False  # never use fp16 on B200; bf16 or fp32
    c["max_steps"] = max_steps
    c["add_final_layer_adapter"] = add_adapter
    c["project"] = f"Ethio-ASR-{name}"
    return c


# === Paper baselines (4 models) ===
configs = {
    # AfriHuBERT (hubert-based). Paper used float32; B200 bf16 is fine.
    # MMS doesn't have native add_adapter so we disable.
    "01_paper_afrihubert.yaml":   cfg("paper-afrihubert",
                                      "ajesujoba/AfriHuBERT",
                                      lr=3e-4, batch_size=8, grad_accum=4,
                                      add_adapter=False),

    "02_paper_mms-300m.yaml":     cfg("paper-mms-300m",
                                      "facebook/mms-300m",
                                      lr=3e-4, batch_size=8, grad_accum=4,
                                      add_adapter=False),

    "03_paper_mms-1b.yaml":       cfg("paper-mms-1b",
                                      "facebook/mms-1b",
                                      lr=7e-5, batch_size=4, grad_accum=8,
                                      add_adapter=False),

    "04_paper_w2v-bert-2.0.yaml": cfg("paper-w2v-bert-2.0",
                                      "facebook/w2v-bert-2.0",
                                      lr=3e-5, batch_size=8, grad_accum=4,
                                      add_adapter=True),

    # === Candidate improvements (3 new models) ===

    # XLS-R-1B: alternative 1B multilingual encoder (trained on 128 langs).
    "05_new_xls-r-1b.yaml":       cfg("new-xls-r-1b",
                                      "facebook/wav2vec2-xls-r-1b",
                                      lr=7e-5, batch_size=4, grad_accum=8,
                                      add_adapter=True),

    # w2v-bert-2.0 extended: same model as paper best, but trained longer
    # (60k steps) + slightly tuned LR. Try to push past 30.48% WER.
    "06_new_w2v-bert-extended.yaml": cfg(
        "new-w2v-bert-extended",
        "facebook/w2v-bert-2.0",
        lr=5e-5, batch_size=8, grad_accum=4,
        max_steps=60000, add_adapter=True,
    ),

    # Slot for whisper-large-v3 (handled by a separate script, not by
    # train_model.py which is CTC-only). See slurm_scripts/train_whisper.py.
}

for fname, c in configs.items():
    path = CFG_DIR / fname
    with open(path, "w") as f:
        yaml.safe_dump(c, f, default_flow_style=False, allow_unicode=True,
                       sort_keys=False, width=10**6)
    print(f"wrote {path}")

print(f"\nTotal: {len(configs)} CTC configs.  Whisper handled separately.")
