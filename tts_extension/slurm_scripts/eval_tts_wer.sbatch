#!/bin/bash
#SBATCH --job-name=tts-eval
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.out
#SBATCH --error=/blue/rcstudents/btulu/Projects/Ethio-TTS/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-TTS"
module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

TTS_MODEL="${TTS_MODEL:-facebook/mms-tts-amh}"
ASR_MODEL="${ASR_MODEL:-boazsew/Ethio-ASR-w2v-bert-2.0-uf}"
N_SAMPLES="${N_SAMPLES:-200}"
OUT_JSON="${OUT_JSON:-$PROJECT_ROOT/results_zero_shot_mms.json}"
SAVE_AUDIO="${SAVE_AUDIO:-$PROJECT_ROOT/tmp_cache/eval_audio}"

echo "=== Round-trip TTS->ASR eval ==="
echo "TTS:  $TTS_MODEL"
echo "ASR:  $ASR_MODEL"
echo "N:    $N_SAMPLES"
nvidia-smi -L

python3 -u "$PROJECT_ROOT/slurm_scripts/eval_tts_wer.py" \
    --tts_model "$TTS_MODEL" \
    --asr_model "$ASR_MODEL" \
    --n_samples "$N_SAMPLES" \
    --out_json "$OUT_JSON" \
    --save_audio_dir "$SAVE_AUDIO"
