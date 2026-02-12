#!/usr/bin/env bash

# run misc. stuff
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

export HF_HOME="/project_dir/huggingface_cache"

# numba tmp dir
NUMBA_CACHE_DIR='/tmp/numba_cache'
LIBROSA_CACHE_DIR="/tmp/librosa_cache"

# show current working directory
echo "Current working directory: $(pwd)"

# run training script
python3 Ethio-ASR/scripts/train_model.py --config Ethio-ASR/config_files/ASR_train_config_multi_debug.yaml
