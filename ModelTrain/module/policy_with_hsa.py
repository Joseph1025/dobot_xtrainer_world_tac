"""
Enhanced ACT Policy with HSA (Hard Sample Aware) Loss

This module extends the ACT policy to include tactile-visual feature alignment
using HSA contrastive loss.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

# Add parent directories to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'dobot_control'))

from dobot_control.tactile_feature_extraction import (
    TactileFeatureExtractor,
    ForwardKinematics,
    CameraProjection,
)
from dobot_control.hsa_loss import HSALossWithThirdPerson

# Import the base ACTPolicy
from ModelTrain.module.policy import ACTPolicy


class ACTPolicyWithHSA(ACTPolicy):
    """
    Enhanced ACT Policy with tactile-visual feature alignment via HSA loss.
    
    This extends the standard ACT policy to:
    1. Extract tactile and wrist visual features
    2. Compute HSA contrastive loss for feature alignment
    3. Combine ACT loss (L1 + KL) with HSA loss
    """
    
    def __init__(self, args_override, hsa_config: Optional[Dict] = None):
        """
        Initialize enhanced ACT policy with HSA loss.
        
        Args:
            args_override: Configuration for ACT model
            hsa_config: Configuration for HSA loss, with keys:
                - enable_hsa (bool): Whether to enable HSA loss
                - hsa_weight (float): Weight for HSA loss term
                - temperature (float): Temperature for contrastive loss
                - use_third_person (bool): Whether to use third-person camera
                - feature_dim (int): Feature dimension for CLIP backbone
                - img_size (int): Image size for feature extractor
                - patch_size (int): Patch size for ViT
                - camera_params (dict): Camera intrinsic/extrinsic parameters
                - robot_type (str): Robot type for FK ("Nova 2" or "Nova 5")
                - sensor_offset (np.ndarray): Sensor offset from end-effector
                - wrist_camera_idx (int): Index of wrist camera in camera_names
                - tactile_camera_idx (int): Index of tactile sensor in camera_names
        """
        super().__init__(args_override)
        
        # Parse HSA configuration
        if hsa_config is None:
            hsa_config = {}
        
        self.enable_hsa = hsa_config.get('enable_hsa', False)
        self.hsa_weight = hsa_config.get('hsa_weight', 1.0)
        
        if self.enable_hsa:
            # Initialize tactile feature extractor
            feature_dim = hsa_config.get('feature_dim', 768)
            img_size = hsa_config.get('img_size', 224)
            patch_size = hsa_config.get('patch_size', 16)
            
            self.feature_extractor = TactileFeatureExtractor(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=feature_dim,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Initialize HSA loss
            temperature = hsa_config.get('temperature', 0.07)
            use_third_person = hsa_config.get('use_third_person', False)
            tp_weight = hsa_config.get('tp_weight', 0.5)
            
            self.hsa_loss_fn = HSALossWithThirdPerson(
                temperature=temperature,
                use_third_person=use_third_person,
                tp_weight=tp_weight,
                reduction='mean'
            )
            
            # Camera and robot configuration
            self.camera_params = hsa_config.get('camera_params', None)
            self.robot_type = hsa_config.get('robot_type', 'Nova 2')
            self.sensor_offset = hsa_config.get('sensor_offset', np.array([0.0, 0.0, 0.02]))
            self.sensor_size = hsa_config.get('sensor_size', (0.04, 0.04))  # 4cm x 4cm
            
            # Camera indices in the camera list
            self.wrist_camera_idx = hsa_config.get('wrist_camera_idx', 1)  # e.g., 'left_wrist'
            self.tactile_camera_idx = hsa_config.get('tactile_camera_idx', 0)  # First tactile sensor
            
            print(f"HSA Loss enabled: weight={self.hsa_weight}, temperature={temperature}")
        else:
            self.feature_extractor = None
            self.hsa_loss_fn = None
            print("HSA Loss disabled")
    
    def extract_tactile_visual_features(self,
                                        wrist_image: torch.Tensor,
                                        tactile_image: torch.Tensor,
                                        qpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract tactile and wrist visual features using the feature extractor.
        
        Args:
            wrist_image: Wrist camera image, shape (B, C, H, W)
            tactile_image: Tactile sensor image, shape (B, C, H, W)
            qpos: Joint angles, shape (B, state_dim) - first 6 are joint angles
        
        Returns:
            h_tau: Tactile features (B, D)
            h_w: Wrist visual features (B, D)
        """
        B = wrist_image.shape[0]
        device = wrist_image.device
        
        h_tau_list = []
        h_w_list = []
        
        for i in range(B):
            # Get joint angles (first 6 values from qpos)
            joint_angles = qpos[i, :6].cpu().numpy()
            
            # Convert images to numpy (H, W, C) format
            wrist_img_np = wrist_image[i].permute(1, 2, 0).cpu().numpy()
            wrist_img_np = (wrist_img_np * 255).astype(np.uint8)
            
            tactile_img_np = tactile_image[i].permute(1, 2, 0).cpu().numpy()
            tactile_img_np = (tactile_img_np * 255).astype(np.uint8)
            
            # Get image sizes
            wrist_h, wrist_w = wrist_img_np.shape[:2]
            
            # Compute sensor pose using forward kinematics
            sensor_pose = ForwardKinematics.compute_tactile_sensor_pose(
                joint_angles=joint_angles,
                robot_type=self.robot_type,
                sensor_offset=self.sensor_offset
            )
            
            # Get camera parameters (use default if not provided)
            if self.camera_params is not None:
                K_w = self.camera_params['K_wrist']
                E_w = self.camera_params['E_wrist']
            else:
                # Use default camera params based on image size
                from dobot_control.tactile_feature_extraction import generate_fake_camera_params
                K_w, E_w = generate_fake_camera_params((wrist_w, wrist_h))
            
            # Compute bounding box in wrist view
            bbox_w = CameraProjection.compute_sensor_bounding_box(
                sensor_pose=sensor_pose,
                sensor_size=self.sensor_size,
                K=K_w,
                E=E_w,
                img_size=(wrist_w, wrist_h)
            )
            
            # Extract features
            features = self.feature_extractor.extract_features(
                wrist_image=wrist_img_np,
                tactile_image=tactile_img_np,
                bbox_wrist=bbox_w,
                bbox_tp=None  # No third-person view for now
            )
            
            h_tau_list.append(features['h_tau'])
            h_w_list.append(features['h_w'])
        
        # Stack into batch
        h_tau = torch.stack(h_tau_list).to(device)
        h_w = torch.stack(h_w_list).to(device)
        
        return h_tau, h_w
    
    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None, 
                 tactile_image=None, compute_hsa=None):
        """
        Forward pass with optional HSA loss computation.
        
        Args:
            qpos: Joint positions/states, shape (B, state_dim)
            image: Camera images (list or tensor)
            actions: Action sequences (for training), shape (B, seq_len, action_dim)
            is_pad: Padding mask, shape (B, seq_len)
            vq_sample: VQ sample (if using VQ-VAE)
            tactile_image: Tactile sensor images, shape (B, C, H, W)
            compute_hsa: Whether to compute HSA loss (defaults to self.enable_hsa during training)
        
        Returns:
            If training: loss_dict with keys: 'l1', 'kl', 'loss', optionally 'hsa_wrist', 'hsa_total'
            If inference: predicted actions
        """
        # Call base ACT policy
        base_output = super().__call__(qpos, image, actions, is_pad, vq_sample)
        
        # If not training or HSA not enabled, return base output
        if actions is None:  # Inference mode
            return base_output
        
        # Determine whether to compute HSA
        if compute_hsa is None:
            compute_hsa = self.enable_hsa and self.training
        
        if not compute_hsa or tactile_image is None:
            return base_output
        
        # Extract wrist camera image from image input
        # Handle both list and tensor inputs
        if isinstance(image, list):
            # List format: [rgb_tensor, tactile_tensor]
            # rgb_tensor shape: (B, num_rgb, C, H, W)
            wrist_image = image[0][:, self.wrist_camera_idx]  # (B, C, H, W)
        else:
            # Tensor format: (B, num_cameras, C, H, W)
            wrist_image = image[:, self.wrist_camera_idx]  # (B, C, H, W)
        
        # Select the tactile camera
        if isinstance(tactile_image, list):
            tactile_img = tactile_image[self.tactile_camera_idx]
        else:
            tactile_img = tactile_image[:, self.tactile_camera_idx]
        
        # Extract features and compute HSA loss
        try:
            h_tau, h_w = self.extract_tactile_visual_features(
                wrist_image=wrist_image,
                tactile_image=tactile_img,
                qpos=qpos
            )
            
            # Compute HSA loss
            hsa_loss_dict = self.hsa_loss_fn(h_tau=h_tau, h_w=h_w)
            
            # Add HSA loss to total loss
            base_output['hsa_wrist'] = hsa_loss_dict['hsa_wrist']
            base_output['hsa_total'] = hsa_loss_dict['hsa_total']
            base_output['loss'] = base_output['loss'] + self.hsa_weight * hsa_loss_dict['hsa_total']
            
        except Exception as e:
            print(f"Warning: Failed to compute HSA loss: {e}")
            # Continue training without HSA loss on error
        
        return base_output
    
    def train(self, mode=True):
        """
        Set the module in training mode.
        Also controls the feature extractor backbone if HSA is enabled.
        """
        super().train(mode)
        if self.enable_hsa and self.feature_extractor is not None:
            if mode:
                self.feature_extractor.backbone.train()
            else:
                self.feature_extractor.backbone.eval()
        return self
    
    def eval(self):
        """
        Set the module in evaluation mode.
        """
        return self.train(False)
    
    def configure_optimizers(self):
        """
        Configure optimizer. If HSA is enabled, include feature extractor parameters.
        """
        base_optimizer = super().configure_optimizers()
        
        if self.enable_hsa and self.feature_extractor is not None:
            # Get parameters from feature extractor
            all_params = list(base_optimizer.param_groups)
            
            # Add feature extractor parameters (with slightly smaller LR)
            feature_params = list(self.feature_extractor.backbone.parameters())
            if len(feature_params) > 0:
                all_params.append({
                    'params': feature_params,
                    'lr': base_optimizer.param_groups[0]['lr'] * 0.5  # 0.5x LR for feature extractor (was 0.1x)
                })
                print(f"Added {len(feature_params)} feature extractor params to optimizer with LR={base_optimizer.param_groups[0]['lr'] * 0.5:.2e}")
            
            # Create new optimizer with all parameters
            optimizer = torch.optim.AdamW(
                all_params,
                lr=base_optimizer.param_groups[0]['lr']
            )
            return optimizer
        
        return base_optimizer


def create_default_hsa_config(
    enable_hsa: bool = True,
    hsa_weight: float = 1.0,
    temperature: float = 0.07,
    img_size: int = 224,
    wrist_camera_name: str = 'left_wrist',
    camera_names: list = None,
    robot_type: str = 'Nova 2'
) -> Dict:
    """
    Create a default HSA configuration dictionary.
    
    Args:
        enable_hsa: Whether to enable HSA loss
        hsa_weight: Weight for HSA loss
        temperature: Temperature for contrastive loss
        img_size: Image size for feature extraction
        wrist_camera_name: Name of the wrist camera
        camera_names: List of all camera names
        robot_type: Robot type for forward kinematics
    
    Returns:
        HSA configuration dictionary
    """
    if camera_names is None:
        camera_names = ['top', 'left_wrist', 'right_wrist']
    
    # Find wrist camera index
    wrist_camera_idx = camera_names.index(wrist_camera_name) if wrist_camera_name in camera_names else 1
    
    return {
        'enable_hsa': enable_hsa,
        'hsa_weight': hsa_weight,
        'temperature': temperature,
        'use_third_person': False,
        'tp_weight': 0.5,
        'feature_dim': 768,
        'img_size': img_size,
        'patch_size': 16,
        'camera_params': None,  # Will use default
        'robot_type': robot_type,
        'sensor_offset': np.array([0.0, 0.0, 0.02]),
        'sensor_size': (0.04, 0.04),
        'wrist_camera_idx': wrist_camera_idx,
        'tactile_camera_idx': 0,
    }

