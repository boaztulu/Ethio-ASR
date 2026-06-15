#!/bin/bash
# Launch all 4 Amharic voice fine-tune jobs.
set -euo pipefail
PR="/blue/rcstudents/btulu/Projects/Ethio-TTS"
for prof in young_male young_female old_male old_female; do
    cfg="$PR/configs/phase2/${prof}.json"
    if [ ! -f "$cfg" ]; then
        echo "skip: $cfg missing"
        continue
    fi
    echo ">>> sbatch ${prof}"
    sbatch --export=ALL,VOICE_CONFIG="$cfg" \
           --job-name="p2-amh-${prof}" \
           "$PR/slurm_scripts/train_p2_voice.sbatch"
done
echo ""
squeue -u "$USER" --noheader | head -10
