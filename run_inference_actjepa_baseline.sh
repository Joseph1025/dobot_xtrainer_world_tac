#!/bin/bash

# Inference script for ACTJEPA baseline policy (frozen ViT, no adapters)
# Run this to compare against ACTJEPAAdapter

cd /home/zexi/Dev/dobot_xtrainer_world_tac

echo "Running inference with ACTJEPA baseline policy..."
echo "Checkpoint directory: ckpt/actjepa_baseline_vitl"
echo "Policy: ACTJEPA (frozen ViT, no adapters)"

python experiments/run_inference_adapter.py \
    --ckpt_dir ./ckpt/actjepa_baseline_vitl \
    --ckpt_name policy_best.ckpt \
    --policy_class ACTJEPA \
    --show_img true \
    --robot_port 6001 \
    --hostname 127.0.0.1

echo "Inference completed!"

