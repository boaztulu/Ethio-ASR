#!/bin/bash
#SBATCH --job-name=cv-amh
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.err

set -euo pipefail
PR="/blue/rcstudents/btulu/Projects/Ethio-TTS"
module purge
module load pytorch/2.8.0
source "$PR/slurm_scripts/setup_env.sh"

echo "=== Download Common Voice Amharic for reference voices ==="
python3 -u "$PR/slurm_scripts/get_common_voice_amh.py"
