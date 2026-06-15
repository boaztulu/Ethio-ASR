#!/usr/bin/env bash
# Source this to set up env vars for any Ethio-ASR job/script
PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-ASR"

# Load .env (HF/WANDB tokens)
set -a
[ -f "$PROJECT_ROOT/.env" ] && . "$PROJECT_ROOT/.env"
set +a

# Even if .env had WANDB_DISABLED, scrub it - newer transformers errors out
unset WANDB_DISABLED

# Cache paths
export HF_HOME="$PROJECT_ROOT/hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/hf_cache/datasets"
export HF_HUB_CACHE="$PROJECT_ROOT/hf_cache/hub"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/hf_cache/transformers"
export TORCH_HOME="$PROJECT_ROOT/hf_cache/torch"
export NUMBA_CACHE_DIR="$PROJECT_ROOT/tmp_cache/numba"
export LIBROSA_CACHE_DIR="$PROJECT_ROOT/tmp_cache/librosa"
export MPLCONFIGDIR="$PROJECT_ROOT/tmp_cache/matplotlib"
export WANDB_DIR="$PROJECT_ROOT/wandb_cache"
export WANDB_CACHE_DIR="$PROJECT_ROOT/wandb_cache"

# Add our extra-installed packages (jiwer, evaluate, librosa, etc.) via
# PYTHONPATH so we keep both ~/.local (transformers/datasets/wandb) AND
# our $PROJECT_ROOT/pylibs available.  Do NOT set PYTHONUSERBASE here -
# that would hide ~/.local entirely.
EXTRA_SITE="$PROJECT_ROOT/pylibs/lib/python3.13/site-packages"
export PYTHONPATH="$EXTRA_SITE:${PYTHONPATH:-}"
export PATH="$PROJECT_ROOT/pylibs/bin:$PATH"
unset PYTHONUSERBASE

mkdir -p "$NUMBA_CACHE_DIR" "$LIBROSA_CACHE_DIR" "$MPLCONFIGDIR" "$WANDB_DIR" "$HF_HOME"

# Show what's set
echo "[setup_env] HF_HOME=$HF_HOME"
echo "[setup_env] HF_TOKEN starts with: ${HF_TOKEN:0:8}..."
echo "[setup_env] EXTRA_SITE (project pylibs) prepended to PYTHONPATH"
echo "[setup_env] ~/.local (transformers/datasets/wandb) is the user-site"
