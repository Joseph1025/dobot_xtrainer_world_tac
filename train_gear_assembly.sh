#!/bin/bash

# Training script for ACTJEPAAdapter with ViT-L + HSA Loss
# Task: dobot_peginhole_tac_1029 (peg-in-hole with tactile sensors)
# HSA: Hard Sample Aware loss for tactile-visual feature alignment

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

python ModelTrain/model_train.py \
    --policy_class ACTJEPAAdapter \
    --task_name dobot_gearassemb_tac_1107 \
    --ckpt_dir ckpt/actjepa_hsa_gear_1107 \
    --num_steps 20000 \
    --vit_model vitl \
    --vit_ckpt_path ./jepa_ckpt/vitl.pt \
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
    --hsa_feature_dim 768 \
    --robot_type "Nova 2" \
    --wrist_camera left_wrist \
    --seed 42

echo "Training with HSA completed!"
echo "HSA loss should decrease from ~4.0 to ~1.0 during training."
echo "Check ckpt/actjepa_adapter_vitl_hsa_new/ for results."


