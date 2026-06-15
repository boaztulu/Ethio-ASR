#!/bin/bash
#SBATCH --job-name=p2-setup
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.err

set -euo pipefail
PR="/blue/rcstudents/btulu/Projects/Ethio-TTS"
FT_REPO="$PR/external/finetune-hf-vits"
module purge
module load pytorch/2.8.0
source "$PR/slurm_scripts/setup_env.sh"

# 1. Build monotonic_align Cython (skip if .so already present)
echo "=== Building monotonic_align ==="
cd "$FT_REPO/monotonic_align"
mkdir -p monotonic_align
if ! ls monotonic_align/*.so 2>/dev/null | head -1; then
    python3 setup.py build_ext --inplace
fi
echo "[so file]" ; find . -name "*.so" -maxdepth 3

# 2. Make the repo importable + ensure utils/ on path
cd "$FT_REPO"
echo "=== Convert discriminator for 4 Ethiopian languages ==="
mkdir -p "$PR/models/mms-tts-with-disc"
export PYTHONPATH="$FT_REPO:${PYTHONPATH:-}"

for LANG in amh tir orm sid; do
    OUT="$PR/models/mms-tts-with-disc/$LANG"
    if [ -f "$OUT/model.safetensors" ] || [ -f "$OUT/pytorch_model.bin" ]; then
        echo "[$LANG] already converted -> skip"
        continue
    fi
    echo "[$LANG] converting..."
    python3 convert_original_discriminator_checkpoint.py \
        --language_code "$LANG" \
        --pytorch_dump_folder_path "$OUT" 2>&1 | tail -8
done

echo ""
echo "=== Inventory ==="
for LANG in amh tir orm sid; do
    OUT="$PR/models/mms-tts-with-disc/$LANG"
    echo "[$LANG]"
    ls -la "$OUT" 2>/dev/null | head -5
done
echo "=== DONE ==="
