#!/bin/bash
#
# Train a single CTC ASR model on the WAXAL Ethiopian dataset.
#
# Usage:
#   sbatch --export=ALL,CONFIG=configs/01_paper_afrihubert.yaml \
#          --job-name=ethio-afrihubert \
#          slurm_scripts/train_ctc.sbatch
#
#SBATCH --job-name=ethio-asr
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
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
REPO_DIR="$PROJECT_ROOT/Ethio-ASR"
CONFIG="${CONFIG:?Set CONFIG=path/to/config.yaml when calling sbatch}"

# Resolve config to absolute path
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$PROJECT_ROOT/$CONFIG"
fi

module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

echo "=== Ethio-ASR training (CTC) ==="
echo "Host:      $(hostname)"
echo "Job:       ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo "Started:   $(date -Iseconds)"
echo "Config:    $CONFIG"
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "Free disk:"
df -h "$PROJECT_ROOT" | tail -1

# Forward SIGUSR1 to Python for graceful checkpoint+exit on timeout/requeue
trap 'echo "[SLURM] Forwarding SIGUSR1"; kill -USR1 $PID 2>/dev/null || true' USR1

cd "$REPO_DIR"
python3 -u scripts/train_model.py --config "$CONFIG" &
PID=$!
wait $PID
STATUS=$?

echo "Train exited with status $STATUS"
echo "End: $(date -Iseconds)"
exit $STATUS
