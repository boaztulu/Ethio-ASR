#!/usr/bin/env bash
# Submit all paper baseline + candidate training jobs.
#
# Usage:
#   ./slurm_scripts/submit_all_paper.sh           # submit all 6 CTC configs
#   ./slurm_scripts/submit_all_paper.sh 04 05     # submit only configs 04 and 05
#
set -euo pipefail
PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-ASR"
cd "$PROJECT_ROOT"

declare -A NAMES=(
    [01]="paper-afrihubert"
    [02]="paper-mms300m"
    [03]="paper-mms1b"
    [04]="paper-w2vbert"
    [05]="new-xlsr1b"
    [06]="new-w2vbert-ext"
)

PICK=("$@")
if [ ${#PICK[@]} -eq 0 ]; then
    PICK=(01 02 03 04 05 06)
fi

for n in "${PICK[@]}"; do
    cfg=$(ls configs/${n}_*.yaml 2>/dev/null | head -1)
    if [ -z "$cfg" ]; then
        echo "skip: no config matches ${n}_*"
        continue
    fi
    name="${NAMES[$n]:-ethio-$n}"
    echo ">>> sbatch --job-name=$name CONFIG=$cfg"
    sbatch --export=ALL,CONFIG="$cfg" \
           --job-name="$name" \
           slurm_scripts/train_ctc.sbatch
done

echo ""
squeue -u "$USER" --noheader | head -20
