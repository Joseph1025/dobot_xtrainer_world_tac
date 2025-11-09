#!/bin/bash

# Training script for ACT (standard ResNet) with HSA Loss
# Task: dobot_peginhole_tac_1107 (peg-in-hole with tactile sensors)
# HSA: Hard Sample Aware loss using CLIP-like ViT backbone

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

python ModelTrain/model_train.py \
    --policy_class ACT \
    --task_name dobot_gearassemb_tac_1107 \
    --ckpt_dir ckpt/act_hsa_gear_1108 \
    --num_steps 20000 \
    --batch_size 16 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 45 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --validate_every 100 \
    --save_every 5000 \
    --enable_hsa \
    --hsa_weight 1.0 \
    --hsa_temperature 0.1 \
    --hsa_img_size 224 \
    --hsa_feature_dim 768 \
    --robot_type "Nova 2" \
    --wrist_camera left_wrist \
    --seed 42

echo "ACT + HSA training completed!"
echo "This uses standard ResNet backbones for RGB + CLIP-like ViT for HSA loss."

