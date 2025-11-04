#!/bin/bash

# Training script for ACTJEPAAdapter with ViT-L
# Task: dobot_peginhole_tac_1029 (peg-in-hole with tactile sensors)

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

python ModelTrain/model_train.py \
    --policy_class ACTJEPAAdapter \
    --task_name dobot_peginhole_tac_1029 \
    --ckpt_dir ckpt/actjepa_adapter_vitl \
    --num_steps 50000 \
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
    --seed 42

echo "Training completed!"

