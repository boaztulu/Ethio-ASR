#!/bin/bash
#SBATCH --job-name=ethio-combined
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.err

set -euo pipefail
PR=/blue/rcstudents/btulu/Projects/Ethio-TTS
module purge
module load pytorch/2.8.0
source "$PR/slurm_scripts/setup_env.sh"

PORT="${PORT:-7863}"
echo "=== Combined TTS+ASR demo ==="
echo "Host: $(hostname)"
nvidia-smi -L

python3 -u "$PR/webapp/combined_app.py" --port "$PORT" --host 0.0.0.0 --share
