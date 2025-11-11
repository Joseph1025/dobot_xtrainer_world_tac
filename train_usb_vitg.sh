#!/bin/bash

# Training script for ACTJEPAAdapter with ViT-G + HSA Loss
# Task: dobot_usb_tac_1107 (USB insertion with tactile sensors)
# HSA: Hard Sample Aware loss for tactile-visual feature alignment
# NOTE: ViT-G uses 1408-dimensional features (vs 768 for ViT-L)

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

python ModelTrain/model_train.py \
    --policy_class ACTJEPAAdapter \
    --task_name dobot_usb_tac_1107 \
    --ckpt_dir ckpt/actjepa_hsa_usb_vitg_1107 \
    --num_steps 30000 \
    --vit_model vitg \
    --vit_ckpt_path ./jepa_ckpt/e150.pt \
    --batch_size 16 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 45 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --validate_every 100 \
    --save_every 5000 \
    --adapter_hidden_dim 512 \
    --adapter_depth 3 \
    --adapter_dropout 0.1 \
    --adapter_scale_init 0.1 \
    --adapter_pooling attention \
    --enable_hsa \
    --hsa_weight 1.0 \
    --hsa_temperature 0.1 \
    --hsa_img_size 224 \
    --hsa_feature_dim 1408 \
    --hsa_num_heads 16 \
    --robot_type "Nova 2" \
    --wrist_camera left_wrist \
    --seed 42

echo "Training with ViT-G + HSA completed!"
echo "HSA loss should decrease from ~4.0 to ~1.0 during training."
echo "Check ckpt/actjepa_hsa_usb_vitg_1107/ for results."


