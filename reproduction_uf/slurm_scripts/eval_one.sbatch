#!/bin/bash
#SBATCH --job-name=eval-one
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-ASR"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR=path/to/model when calling sbatch}"
OUT_JSON="${OUT_JSON:-${MODEL_DIR}/eval_test.json}"

module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

echo "=== Eval $MODEL_DIR ==="
echo "Output: $OUT_JSON"
nvidia-smi -L

python3 "$PROJECT_ROOT/slurm_scripts/evaluate_ctc.py" \
    --model_dir "$MODEL_DIR" \
    --dataset "badrex/waxalNLP-ethiopic-final" \
    --split test \
    --batch_size 16 \
    --out_json "$OUT_JSON"
