#!/bin/bash

# Inference script for ACTJEPAAdapter policy
# Run this to deploy the adapter-trained model on the robot

cd /home/zexi/Dev/dobot_xtrainer_world_tac

echo "Running inference with ACTJEPAAdapter policy..."
echo "Checkpoint directory: ckpt/actjepa_adapter_vitl"
echo "Policy: ACTJEPAAdapter (with learnable adapters)"

python experiments/run_inference_adapter.py \
    --ckpt_dir ./ckpt/actjepa_adapter_vitl \
    --ckpt_name policy_best.ckpt \
    --policy_class ACTJEPAAdapter \
    --show_img true \
    --robot_port 6001 \
    --hostname 127.0.0.1

echo "Inference completed!"

