#!/bin/bash
#SBATCH --job-name=eval-pp-all
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-ASR/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-ASR"
module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

nvidia-smi -L

# Re-eval paper's 4 published checkpoints with post-processing
mkdir -p "$PROJECT_ROOT/eval_results/paper_published_pp"
for size in 94M 300M 1B 600M; do
    snap=$(ls -d "$PROJECT_ROOT/hf_cache/hub/models--badrex--Ethio-ASR-multilingual-${size}/snapshots/"*/ 2>/dev/null | head -1)
    [ -z "$snap" ] && continue
    snap="${snap%/}"
    out="$PROJECT_ROOT/eval_results/paper_published_pp/${size}_test_pp.json"
    echo ""
    echo "=== paper-pub-${size} (with post-proc) ==="
    python3 "$PROJECT_ROOT/slurm_scripts/evaluate_ctc.py" \
        --model_dir "$snap" --split test --batch_size 16 \
        --out_json "$out" --apply_postproc || echo "FAILED: $size"
done

# Re-eval our trained models with post-processing
for d in \
    "$PROJECT_ROOT/models/facebook/mms-300m-09062026-205411" \
    "$PROJECT_ROOT/models/facebook/mms-1b-09062026-203848" \
    "$PROJECT_ROOT/models/facebook/w2v-bert-2.0-09062026-212456" \
    "$PROJECT_ROOT/models/facebook/w2v-bert-2.0-10062026-003217/checkpoint-43200" \
    "$PROJECT_ROOT/models/facebook/wav2vec2-xls-r-1b-09062026-230639" \
    "$PROJECT_ROOT/models/ajesujoba/AfriHuBERT-11062026-234348" ; do
    [ -d "$d" ] || continue
    out="${d}/eval_test_pp.json"
    echo ""
    echo "=== $(basename $(dirname $d))/$(basename $d) (with post-proc) ==="
    python3 "$PROJECT_ROOT/slurm_scripts/evaluate_ctc.py" \
        --model_dir "$d" --split test --batch_size 16 \
        --out_json "$out" --apply_postproc || echo "FAILED: $d"
done

echo ""
echo "=== ALL DONE ==="
