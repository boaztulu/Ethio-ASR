#!/usr/bin/env bash
#SBATCH --job-name=prefetch
#SBATCH --partition=hpg-default
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents-b
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/prefetch_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/prefetch_%j.err

set -euo pipefail
source /blue/rcstudents/btulu/Projects/Ethio-ASR/slurm_scripts/setup_env.sh
module purge
module load pytorch/2.8.0
echo "[job] Starting model prefetch at $(date) on $(hostname)"
python3 /blue/rcstudents/btulu/Projects/Ethio-ASR/slurm_scripts/prefetch_models.py
echo "[job] Done at $(date)"
