#!/bin/bash

# Inference script for ACTJEPAAdapter policy
# Run this to deploy the adapter-trained model on the robot

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Running inference with ACTJEPAAdapter policy..."
echo "Checkpoint directory: ckpt/actjepa_adapter_vitl"
echo "Policy: ACTJEPAAdapter (with learnable adapters)"

python experiments/run_inference_adapter.py \
    --ckpt_dir ./ckpt/actjepa_adapter_vitl \
    --ckpt_name policy_best.ckpt \
    --task_name dobot_peginhole_tac_1029 \
    --vit_ckpt_path ./jepa_ckpt/vitl.pt \
    --vit_model vitl \
    --policy_class ACTJEPAAdapter \
    --show_img \
    --robot_port 6001 \
    --hostname 127.0.0.1 \
    --temporal_agg

echo "Inference completed!"

