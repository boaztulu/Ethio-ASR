#!/bin/bash
#SBATCH --job-name=tts-demo
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-TTS"
module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

PORT="${PORT:-7862}"

echo "=== Ethio-TTS demo (multi-language + voice controls) ==="
echo "Host:  $(hostname)"
echo "Job:   ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo "Port:  $PORT"
nvidia-smi -L

python3 -u "$PROJECT_ROOT/webapp/app.py" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --share
