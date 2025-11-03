#!/bin/bash

# Training script for ACTJEPAAdapter with ViT-L
# Task: dobot_peginhole_tac_1029 (peg-in-hole with tactile sensors)

cd /home/zexi/Dev/dobot_xtrainer_world_tac

python ModelTrain/model_train.py \
    --policy_class ACTJEPAAdapter \
    --task_name dobot_peginhole_tac_1029 \
    --ckpt_dir ckpt/actjepa_adapter_vitl \
    --num_steps 50000 \
    --vit_model vitl \
    --vit_ckpt_path /home/zexi/Dev/dobot_xtrainer_world_tac/jepa_ckpts/vitl.pt \
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

