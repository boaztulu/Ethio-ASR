#!/bin/bash
#SBATCH --job-name=ethio-whisper
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gpus=1
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-ASR"
module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

echo "=== Whisper-large-v3 LoRA fine-tune ==="
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date -Iseconds)"
nvidia-smi -L || true

trap 'echo "[SLURM] SIGUSR1"; kill -USR1 $PID 2>/dev/null || true' USR1
python3 -u "$PROJECT_ROOT/slurm_scripts/train_whisper.py" &
PID=$!
wait $PID
STATUS=$?
echo "End: $(date -Iseconds)  status=$STATUS"
exit $STATUS
