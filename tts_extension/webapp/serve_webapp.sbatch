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

TTS_MODEL="${TTS_MODEL:-facebook/mms-tts-amh}"
ASR_MODEL="${ASR_MODEL:-boazsew/Ethio-ASR-w2v-bert-2.0-uf}"
PORT="${PORT:-7861}"

echo "=== Ethio-TTS demo ==="
echo "Host:  $(hostname)"
echo "Job:   ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo "TTS:   $TTS_MODEL"
echo "ASR:   $ASR_MODEL"
echo "Port:  $PORT"
nvidia-smi -L

python3 -u "$PROJECT_ROOT/webapp/app.py" \
    --tts_model "$TTS_MODEL" \
    --asr_model "$ASR_MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --share
