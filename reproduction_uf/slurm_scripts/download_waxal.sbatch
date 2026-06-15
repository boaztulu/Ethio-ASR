#!/usr/bin/env bash
#SBATCH --job-name=waxal-dl
#SBATCH --partition=hpg-default
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/waxal_download_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/waxal_download_%j.err

set -euo pipefail

PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-ASR"

# Setup env
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

# Load python with torch + transformers + datasets
module purge
module load pytorch/2.8.0

echo "[job] Starting WAXAL download at $(date)"
echo "[job] On host: $(hostname)"
echo "[job] Free space in /blue/rcstudents:"
df -h /blue/rcstudents | tail -1

python3 "$PROJECT_ROOT/slurm_scripts/download_waxal.py"

echo "[job] Done at $(date)"
echo "[job] Final dataset size:"
du -sh "$PROJECT_ROOT/data/waxal_ethiopic" 2>/dev/null || true
echo "[job] HF cache size:"
du -sh "$PROJECT_ROOT/hf_cache" 2>/dev/null || true
