"""
Enhanced Model Training Script with HSA Loss

This script extends the standard ACT training to include HSA (Hard Sample Aware)
loss for tactile-visual feature alignment.

Usage:
    python ModelTrain/model_train_with_hsa.py \
        --task_name dobot_pick_random \
        --ckpt_dir ./ckpt/dobot_pick_hsa \
        --enable_hsa \
        --hsa_weight 1.0 \
        --batch_size 16 \
        --num_steps 30000
"""

import argparse
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR+'/ModelTrain')
sys.path.append(BASE_DIR+'/ModelTrain/detr')
sys.path.append(BASE_DIR+'/robomimic-r2d2')

from module.train_module_with_hsa import train


def arg_config():
    parser = argparse.ArgumentParser()
    
    # Standard ACT arguments
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', 
                       default='./ckpt/dobot_pick_random_hsa', required=False)
    parser.add_argument('--task_name', action='store', type=str, 
                       default='dobot_pick_random_1013', help='task_name', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', 
                       default=16, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', 
                       default=0, required=False)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', 
                       default=30000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', 
                       default=2e-5, required=False)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', action='store', type=int, default=100, 
                       help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=100, 
                       help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=10000, 
                       help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, 
                       help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    
    # ACT model parameters
    parser.add_argument('--kl_weight', action='store', type=int, 
                       help='KL divergence weight, recommended set 10 or 100', 
                       default=10, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, 
                       help='The model predicts the length of the output action sequence at a time', 
                       default=45, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', 
                       default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, 
                       help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store_true', default=True)
    parser.add_argument('--no_encoder', action='store_true', default=False)
    
    # ViTG-related arguments
    parser.add_argument('--use_vitg', action='store_true', default=False, 
                       help='Use ViTG encoder for tactile images instead of ResNet')
    parser.add_argument('--vitg_ckpt_path', action='store', type=str, 
                       help='Path to ViTG checkpoint file (.pt)', required=False)
    
    # HSA Loss arguments
    parser.add_argument('--enable_hsa', action='store_true', default=False,
                       help='Enable HSA (Hard Sample Aware) loss for tactile-visual alignment')
    parser.add_argument('--hsa_weight', action='store', type=float, default=1.0,
                       help='Weight for HSA loss term (default: 1.0)')
    parser.add_argument('--hsa_temperature', action='store', type=float, default=0.07,
                       help='Temperature parameter for HSA contrastive loss (default: 0.07)')
    parser.add_argument('--hsa_img_size', action='store', type=int, default=224,
                       help='Image size for HSA feature extraction (default: 224)')
    parser.add_argument('--hsa_feature_dim', action='store', type=int, default=768,
                       help='Feature dimension for HSA backbone (default: 768)')
    parser.add_argument('--robot_type', action='store', type=str, default='Nova 2',
                       choices=['Nova 2', 'Nova 5'],
                       help='Robot type for forward kinematics (default: Nova 2)')
    parser.add_argument('--wrist_camera', action='store', type=str, default='left_wrist',
                       help='Name of the wrist camera for HSA (default: left_wrist)')
    
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = arg_config()
    
    # Print HSA configuration if enabled
    if args['enable_hsa']:
        print("\n" + "="*60)
        print("HSA Loss Configuration:")
        print(f"  Enabled: {args['enable_hsa']}")
        print(f"  Weight: {args['hsa_weight']}")
        print(f"  Temperature: {args['hsa_temperature']}")
        print(f"  Image Size: {args['hsa_img_size']}")
        print(f"  Feature Dim: {args['hsa_feature_dim']}")
        print(f"  Robot Type: {args['robot_type']}")
        print(f"  Wrist Camera: {args['wrist_camera']}")
        print("="*60 + "\n")
    
    train(args)

