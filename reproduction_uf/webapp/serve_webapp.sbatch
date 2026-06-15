#!/bin/bash
#
# Launch the Ethio-ASR Gradio demo on a HiPerGator GPU node.
# Default: 12 h on hpg-turin (L4) with 1 GPU; override with sbatch flags.
#
#SBATCH --job-name=ethio-demo
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

# Gradio is not in pylibs by default; install once if missing
python3 -c "import gradio" 2>/dev/null || {
    echo "[serve] installing gradio into pylibs"
    PYTHONUSERBASE=$PROJECT_ROOT/pylibs pip install --user --quiet "gradio>=4.36" "gradio_client>=1.0"
}

PORT="${PORT:-7860}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/facebook/w2v-bert-2.0-09062026-212456}"

echo "=== Ethio-ASR demo ==="
echo "Host:   $(hostname)"
echo "Job:    ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo "Port:   $PORT"
echo "Model:  $MODEL_DIR"
echo ""
echo "After it prints 'Running on local URL: http://0.0.0.0:$PORT',"
echo "open a NEW local terminal and run this on YOUR laptop:"
echo "    ssh -L $PORT:$(hostname):$PORT btulu@hpg.rc.ufl.edu"
echo "then open http://localhost:$PORT in your browser."
echo "(Or run app.py with --share for a public gradio.live URL.)"
echo ""
nvidia-smi -L

python3 -u "$PROJECT_ROOT/Ethio-ASR/reproduction_uf/webapp/app.py" \
    --model_dir "$MODEL_DIR" \
    --port "$PORT" \
    --host 0.0.0.0
