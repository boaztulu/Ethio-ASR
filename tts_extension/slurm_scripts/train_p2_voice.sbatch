#!/bin/bash
#
# Fine-tune MMS-TTS-Amharic for ONE voice profile.
# Usage:
#   sbatch --export=ALL,VOICE_CONFIG=/path/to/young_male.json \
#          --job-name=p2-young_male train_p2_voice.sbatch
#
#SBATCH --job-name=p2-voice
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.err

set -euo pipefail
PR="/blue/rcstudents/btulu/Projects/Ethio-TTS"
FT_REPO="$PR/external/finetune-hf-vits"
CONFIG="${VOICE_CONFIG:?Set VOICE_CONFIG=path/to/config.json}"

module purge
module load pytorch/2.8.0
source "$PR/slurm_scripts/setup_env.sh"
export PYTHONPATH="$FT_REPO:${PYTHONPATH:-}"

echo "=== MMS-TTS Phase 2 fine-tune ==="
echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo "Config:  $CONFIG"
nvidia-smi -L

trap 'echo "[SLURM] SIGUSR1"; kill -USR1 $PID 2>/dev/null || true' USR1

cd "$FT_REPO"
python3 -u run_vits_finetuning.py "$CONFIG" &
PID=$!
wait $PID
STATUS=$?
echo "Training exited $STATUS"
exit $STATUS
