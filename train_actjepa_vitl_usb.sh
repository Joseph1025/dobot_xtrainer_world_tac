#!/bin/bash

# Baseline training script for ACTJEPA (frozen ViT-L, no adapters)
# Task: dobot_peginhole_tac_1029 (peg-in-hole with tactile sensors)
# Use this to compare against ACTJEPAAdapter

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

python ModelTrain/model_train.py \
    --policy_class ACTJEPA \
    --task_name dobot_usb_tac_1107 \
    --ckpt_dir ckpt/actjepa_baseline_usb_1108 \
    --num_steps 10000 \
    --vit_model vitl \
    --vit_ckpt_path ./jepa_ckpt/vitl.pt \
    --batch_size 16 \
    --lr 2e-5 \
    --kl_weight 10 \
    --chunk_size 45 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --validate_every 100 \
    --save_every 1000 \
    --seed 0

echo "Baseline training completed!"

