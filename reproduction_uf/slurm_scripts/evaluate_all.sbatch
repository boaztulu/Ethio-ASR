#!/bin/bash
# Evaluate all trained models (or the published paper checkpoints) on WAXAL test.
#
# Usage:
#   sbatch --export=ALL,MODELS_GLOB="/blue/.../models/*/best" evaluate_all.sbatch
#
#SBATCH --job-name=ethio-eval
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-ASR"
module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

MODELS_GLOB="${MODELS_GLOB:-$PROJECT_ROOT/models/*/best}"
echo "=== evaluate models matching: $MODELS_GLOB ==="
for d in $MODELS_GLOB; do
    [ -d "$d" ] || continue
    echo ">>> Evaluating $d"
    python3 "$PROJECT_ROOT/slurm_scripts/evaluate_ctc.py" \
        --model_dir "$d" \
        --dataset "$PROJECT_ROOT/data/waxal_ethiopic" \
        --split test \
        --batch_size 8 || echo "  (eval failed for $d)"
done
echo "=== done ==="
