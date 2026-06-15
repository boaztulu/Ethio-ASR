#!/usr/bin/env bash
# Source this for the Ethio-TTS project.
# Reuses pylibs from the Ethio-ASR project (saves disk + setup time)
# but uses a separate hf_cache and tmp_cache.

PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-TTS"
ASR_PROJECT_ROOT="/blue/rcstudents/btulu/Projects/Ethio-ASR"

# Load .env (HF tokens, cache vars)
set -a
[ -f "$PROJECT_ROOT/.env" ] && . "$PROJECT_ROOT/.env"
set +a
unset WANDB_DISABLED

# Override cache paths to TTS project (in case .env has stale ASR paths)
export HF_HOME="$PROJECT_ROOT/hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/hf_cache/datasets"
export HF_HUB_CACHE="$PROJECT_ROOT/hf_cache/hub"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/hf_cache/transformers"
export TORCH_HOME="$PROJECT_ROOT/hf_cache/torch"
export NUMBA_CACHE_DIR="$PROJECT_ROOT/tmp_cache/numba"
export LIBROSA_CACHE_DIR="$PROJECT_ROOT/tmp_cache/librosa"
export MPLCONFIGDIR="$PROJECT_ROOT/tmp_cache/matplotlib"
export GRADIO_TEMP_DIR="$PROJECT_ROOT/tmp_cache/gradio"

mkdir -p "$NUMBA_CACHE_DIR" "$LIBROSA_CACHE_DIR" "$MPLCONFIGDIR" \
         "$GRADIO_TEMP_DIR" "$HF_HOME"

# Reuse pylibs from ASR project (transformers 4.55, accelerate 1.13, gradio 5.x, etc.)
EXTRA_SITE="$ASR_PROJECT_ROOT/pylibs/lib/python3.13/site-packages"
export PYTHONPATH="$EXTRA_SITE:${PYTHONPATH:-}"
export PATH="$ASR_PROJECT_ROOT/pylibs/bin:$PATH"
# Some TTS-specific deps may go into a NEW pylibs dir later (Ethio-TTS/pylibs)
TTS_SITE="$PROJECT_ROOT/pylibs/lib/python3.13/site-packages"
if [ -d "$TTS_SITE" ]; then
    export PYTHONPATH="$TTS_SITE:$PYTHONPATH"
    export PATH="$PROJECT_ROOT/pylibs/bin:$PATH"
fi
unset PYTHONUSERBASE

echo "[setup_env-tts] HF_HOME=$HF_HOME"
echo "[setup_env-tts] HF_TOKEN starts with: ${HF_TOKEN:0:8}..."
echo "[setup_env-tts] reuse ASR pylibs (transformers, gradio, etc.)"
