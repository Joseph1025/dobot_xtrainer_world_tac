#!/bin/bash

# Training script for ACTJEPAAdapter with CLIP (RGB) + V-JEPA (Tactile) + HSA Loss
# This is the unified model architecture combining:
# - CLIP ViT encoder for RGB cameras (trainable or frozen)
# - V-JEPA ViT encoder with adapters for tactile sensors (frozen base, trainable adapters)
# - Hierarchical Spatial Alignment (HSA) loss for tactile-visual alignment

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Task configuration
TASK="dobot_peginhole_tac_1107"  # Change to your task
VIT_CKPT="jepa_ckpt/vitl.pt"  # V-JEPA checkpoint for tactile sensors

python ModelTrain/model_train.py \
    --policy_class ACTJEPAAdapter \
    --task_name $TASK \
    --ckpt_dir ckpt/actjepa_adapter_clip_${TASK} \
    --num_steps 20000 \
    --batch_size 16 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 45 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --validate_every 100 \
    --save_every 5000 \
    --vit_ckpt_path $VIT_CKPT \
    --vit_model vitl \
    --adapter_hidden_dim 512 \
    --adapter_depth 3 \
    --adapter_dropout 0.1 \
    --adapter_scale_init 0.1 \
    --adapter_pooling attention \
    --clip_model ViT-B-16 \
    --clip_pretrained openai \
    --freeze_clip \
    --enable_hsa \
    --hsa_weight 1.0 \
    --hsa_temperature 0.07 \
    --hsa_img_size 224 \
    --hsa_feature_dim 768 \
    --hsa_num_heads 12 \
    --robot_type "Nova 2" \
    --wrist_camera left_wrist \
    --seed 42


