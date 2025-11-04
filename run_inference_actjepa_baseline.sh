#!/bin/bash

# Inference script for ACTJEPA baseline policy (frozen ViT, no adapters)
# Run this to compare against ACTJEPAAdapter

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Running inference with ACTJEPA baseline policy..."
echo "Checkpoint directory: ckpt/actjepa_baseline_vitl"
echo "Policy: ACTJEPA (frozen ViT, no adapters)"

python experiments/run_inference_adapter.py \
    --ckpt_dir ./ckpt/actjepa_baseline_vitl \
    --ckpt_name policy_best.ckpt \
    --task_name dobot_peginhole_tac_1029 \
    --vit_ckpt_path ./jepa_ckpt/vitl.pt \
    --vit_model vitl \
    --policy_class ACTJEPA \
    --show_img \
    --robot_port 6001 \
    --hostname 127.0.0.1

echo "Inference completed!"

