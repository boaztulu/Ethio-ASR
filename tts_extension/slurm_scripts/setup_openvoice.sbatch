#!/bin/bash
#SBATCH --job-name=ov-setup
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.err

set -euo pipefail
PR="/blue/rcstudents/btulu/Projects/Ethio-TTS"
module purge
module load pytorch/2.8.0
source "$PR/slurm_scripts/setup_env.sh"

# 1. Download OpenVoice v2 converter checkpoints from HF mirror
#    (original myshell S3 URL is now 404; HF copy at myshell-ai/OpenVoiceV2)
CKPT_DIR="$PR/openvoice_ckpts"
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/converter/checkpoint.pth" ]; then
    echo "[ov] downloading OpenVoice v2 checkpoints from HF..."
    python3 - <<PYEOF
import os
from huggingface_hub import snapshot_download, login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
p = snapshot_download(
    "myshell-ai/OpenVoiceV2",
    repo_type="model",
    local_dir="$CKPT_DIR",
    allow_patterns=["converter/*", "base_speakers/ses/*"],
)
print("downloaded to", p)
PYEOF
    echo "[ov] files:"
    find "$CKPT_DIR" -type f | head -20
fi

# 2. Install minimal extra deps (avoid OpenVoice's pinned old librosa/numpy)
echo "[ov] installing minimal extras into Ethio-TTS pylibs..."
mkdir -p "$PR/pylibs"
PYTHONUSERBASE="$PR/pylibs" pip install --user --quiet --no-deps wavmark 2>&1 | tail -3

# 3. Verify ToneColorConverter loads
python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, "/blue/rcstudents/btulu/Projects/Ethio-TTS/external/OpenVoice")
import torch
from openvoice.api import ToneColorConverter

ckpt_dir = "/blue/rcstudents/btulu/Projects/Ethio-TTS/openvoice_ckpts"
config = os.path.join(ckpt_dir, "converter/config.json")
ckpt   = os.path.join(ckpt_dir, "converter/checkpoint.pth")
assert os.path.exists(config), config
assert os.path.exists(ckpt), ckpt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[ov] device={device}")
tcc = ToneColorConverter(config, device=device, enable_watermark=False)
tcc.load_ckpt(ckpt)
print("[ov] ToneColorConverter loaded OK")
print(f"[ov] sampling_rate={tcc.hps.data.sampling_rate}")
PYEOF
echo "[ov] DONE"
