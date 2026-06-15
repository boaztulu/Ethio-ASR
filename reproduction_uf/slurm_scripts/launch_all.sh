#!/bin/bash
#
# Launch all 7 Ethio-ASR training jobs (4 paper baselines + 3 candidates).
# Sized to fit on a single B200 GPU each.
#
# Usage:
#   ./slurm_scripts/launch_all.sh                       # submit all
#   ./slurm_scripts/launch_all.sh 04 05 06 whisper      # subset
#
set -euo pipefail
PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-ASR"
cd "$PROJECT_ROOT"

declare -A JOBS=(
    # paper baselines
    [01]="paper-afrihubert      configs/01_paper_afrihubert.yaml       96G  3-00:00:00"
    [02]="paper-mms300m         configs/02_paper_mms-300m.yaml         128G 3-00:00:00"
    [03]="paper-mms1b           configs/03_paper_mms-1b.yaml           160G 4-00:00:00"
    [04]="paper-w2vbert         configs/04_paper_w2v-bert-2.0.yaml     128G 3-00:00:00"
    # candidate improvements
    [05]="new-xlsr1b            configs/05_new_xls-r-1b.yaml           160G 4-00:00:00"
    [06]="new-w2vbert-ext       configs/06_new_w2v-bert-extended.yaml  128G 4-00:00:00"
)

PICK=("$@")
if [ ${#PICK[@]} -eq 0 ]; then
    PICK=(01 02 03 04 05 06 whisper)
fi

for n in "${PICK[@]}"; do
    if [ "$n" = "whisper" ]; then
        echo ">>> sbatch whisper (separate script)"
        sbatch "$PROJECT_ROOT/slurm_scripts/train_whisper.sbatch"
        continue
    fi
    spec="${JOBS[$n]:-}"
    if [ -z "$spec" ]; then
        echo "skip: no job '$n'"
        continue
    fi
    name=$(awk '{print $1}' <<<"$spec")
    cfg=$(awk '{print $2}' <<<"$spec")
    mem=$(awk '{print $3}' <<<"$spec")
    tlim=$(awk '{print $4}' <<<"$spec")

    echo ">>> sbatch $name (mem=$mem time=$tlim) CONFIG=$cfg"
    sbatch --export=ALL,CONFIG="$cfg" \
           --job-name="ethio-$name" \
           --mem="$mem" \
           --time="$tlim" \
           "$PROJECT_ROOT/slurm_scripts/train_ctc.sbatch"
done

echo ""
echo "=== Current queue ==="
squeue -u "$USER" --noheader | head -30
