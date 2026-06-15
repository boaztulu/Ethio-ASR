#!/bin/bash
#SBATCH --job-name=tts-filter
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-TTS"
module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

# Reuse the ASR project's HF cache (already has WAXAL downloaded - 58 GB)
export HF_HUB_CACHE="/blue/rcstudents/btulu/Projects/Ethio-ASR/hf_cache/hub"
export HF_DATASETS_CACHE="/blue/rcstudents/btulu/Projects/Ethio-ASR/hf_cache/datasets"
export HF_HOME="/blue/rcstudents/btulu/Projects/Ethio-ASR/hf_cache"

echo "=== Filter WAXAL Amharic for TTS ==="
python3 -u "$PROJECT_ROOT/slurm_scripts/filter_waxal_amh.py"
