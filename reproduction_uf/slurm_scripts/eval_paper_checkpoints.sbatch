#!/bin/bash
#SBATCH --job-name=eval-paper-ckpts
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-ASR"
module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

mkdir -p "$PROJECT_ROOT/eval_results/paper_published"

echo "=== Evaluating the paper's published checkpoints on WAXAL test ==="
nvidia-smi -L

# Iterate the 4 multilingual paper checkpoints
for size in 94M 300M 1B 600M; do
    repo="badrex/Ethio-ASR-multilingual-${size}"
    snap_dir=$(ls -d "$PROJECT_ROOT/hf_cache/hub/models--badrex--Ethio-ASR-multilingual-${size}/snapshots/"*/ 2>/dev/null | head -1)
    if [ -z "$snap_dir" ]; then
        echo "skip $size: no snapshot found"
        continue
    fi
    snap_dir="${snap_dir%/}"
    out="$PROJECT_ROOT/eval_results/paper_published/${size}_test.json"
    echo "--- evaluating $repo ($snap_dir) ---"
    python3 "$PROJECT_ROOT/slurm_scripts/evaluate_ctc.py" \
        --model_dir "$snap_dir" \
        --split test \
        --batch_size 8 \
        --out_json "$out" || echo "FAILED: $size"
done

echo "=== done ==="
ls -la "$PROJECT_ROOT/eval_results/paper_published"
