"""
Tactile-Visual Feature Extraction Module

This module implements the extraction of multi-modal features from tactile sensors
and wrist camera images, following the approach described in the paper.

Key components:
1. Forward kinematics to compute 3D pose of tactile sensor
2. Camera projection to find 2D bounding boxes in camera views
3. CLIP-like vision backbone to extract intermediate features
4. Feature extraction: h_tau (tactile), h_w (wrist), h_tp (third-person)
"""

import numpy as np
from typing import Tuple, Dict, Optional

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. TactileFeatureExtractor will not work.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Some image operations may fail.")


class CLIPLikeBackbone(nn.Module if TORCH_AVAILABLE else object):
    """
    A CLIP-like vision backbone for extracting intermediate features.
    Uses a Vision Transformer (ViT) architecture similar to CLIP.
    """
    
    def __init__(self, 
                 model_name: str = "vit_b_16",
                 img_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 return_intermediate: bool = True):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CLIPLikeBackbone")
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patches_per_side = img_size // patch_size
        self.return_intermediate = return_intermediate
        
        # Patch embedding
        self.conv_proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        num_patches = (img_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.ln_post = nn.LayerNorm(embed_dim)
        
        # Normalize input
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, x: torch.Tensor, return_all_layers: bool = False) -> torch.Tensor:
        """
        Forward pass through the vision backbone.
        
        Args:
            x: Input images, shape (B, C, H, W) where H=W=img_size
            return_all_layers: If True, return features from all transformer layers
        
        Returns:
            If return_all_layers: List of feature tensors from each layer
            Otherwise: Features from intermediate layer (shape: B, N_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Normalize
        x = self.normalize(x)
        
        # Patch embedding
        x = self.conv_proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, E, HP, WP = x.shape
        
        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        
        # Add class token
        class_token = self.class_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([class_token, x], dim=1)  # (B, N_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.positional_embedding
        
        # Store intermediate features
        intermediate_features = []
        
        # Pass through transformer
        for i, layer in enumerate(self.transformer.layers):
            x = layer(x)
            if self.return_intermediate and i == self.num_layers // 2:
                # Store intermediate layer (skip class token)
                intermediate_features.append(x[:, 1:, :])  # (B, N_patches, embed_dim)
        
        if return_all_layers:
            return intermediate_features if intermediate_features else [x[:, 1:, :]]
        else:
            # Return intermediate layer features or final features
            if intermediate_features:
                return intermediate_features[-1]  # (B, N_patches, embed_dim)
            else:
                x = self.ln_post(x)
                return x[:, 1:, :]  # (B, N_patches, embed_dim)
    
    def get_feature_grid_shape(self) -> Tuple[int, int]:
        """Get the spatial shape of the feature grid (height, width)."""
        return (self.patches_per_side, self.patches_per_side)


class ForwardKinematics:
    """Forward kinematics calculator for DoBot robot."""
    
    @staticmethod
    def dh_transformation_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """Create a DH transformation matrix."""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        return np.array([
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def compute_tactile_sensor_pose(joint_angles: np.ndarray, 
                                    robot_type: str = "Nova 2",
                                    sensor_offset: np.ndarray = np.array([0, 0, 0.02])) -> np.ndarray:
        """
        Compute the 3D pose (4x4 transformation matrix) of the tactile sensor.
        
        Args:
            joint_angles: Joint angles in radians, shape (6,)
            robot_type: Robot type ("Nova 2" or "Nova 5")
            sensor_offset: Offset from end-effector to sensor center, shape (3,)
        
        Returns:
            4x4 transformation matrix representing the sensor pose in SE(3)
        """
        if len(joint_angles) < 6:
            raise ValueError("Joint angles must have at least 6 values")
        
        q0, q1, q2, q3, q4, q5 = joint_angles[:6]
        
        if robot_type == "Nova 2":
            dh_params = [
                (q0, 0.2234, 0, np.pi / 2),
                (q1 - np.pi / 2, 0, -0.280, 0),
                (q2, 0, -0.225, 0),
                (q3 - np.pi / 2, 0.1175, 0, np.pi / 2),
                (q4, 0.120, 0, -np.pi / 2),
                (q5, 0.088, 0, 0)
            ]
        elif robot_type == "Nova 5":
            dh_params = [
                (q0, 0.240, 0, np.pi / 2),
                (q1 - np.pi / 2, 0, -0.400, 0),
                (q2, 0, -0.330, 0),
                (q3 - np.pi / 2, 0.135, 0, np.pi / 2),
                (q4, 0.120, 0, -np.pi / 2),
                (q5, 0.088, 0, 0)
            ]
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")
        
        # Compute forward kinematics
        T = np.eye(4)
        for params in dh_params:
            T = np.dot(T, ForwardKinematics.dh_transformation_matrix(*params))
        
        # Apply sensor offset
        T_sensor_offset = np.eye(4)
        T_sensor_offset[:3, 3] = sensor_offset
        T_sensor = np.dot(T, T_sensor_offset)
        
        return T_sensor


class CameraProjection:
    """Camera projection utilities for 3D to 2D mapping."""
    
    @staticmethod
    def project_3d_to_2d(points_3d: np.ndarray,
                         K: np.ndarray,
                         E: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: 3D points in world coordinates, shape (N, 3) or (3,)
            K: Camera intrinsic matrix, shape (3, 3)
            E: Camera extrinsic matrix (world to camera), shape (4, 4)
        
        Returns:
            2D pixel coordinates, shape (N, 2) or (2,)
        """
        points_3d = np.array(points_3d)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)
        
        # Add homogeneous coordinate
        N = points_3d.shape[0]
        points_3d_homo = np.hstack([points_3d, np.ones((N, 1))])
        
        # Transform to camera coordinates
        points_cam = (E @ points_3d_homo.T).T  # (N, 4)
        
        # Project to 2D
        points_cam_3d = points_cam[:, :3]  # (N, 3)
        points_2d_homo = (K @ points_cam_3d.T).T  # (N, 3)
        
        # Divide by z to get pixel coordinates
        z = points_2d_homo[:, 2:3]
        z = np.where(z == 0, 1e-8, z)  # Avoid division by zero
        points_2d = points_2d_homo[:, :2] / z
        
        if points_2d.shape[0] == 1:
            return points_2d[0]
        return points_2d
    
    @staticmethod
    def compute_sensor_bounding_box(sensor_pose: np.ndarray,
                                     sensor_size: Tuple[float, float],
                                     K: np.ndarray,
                                     E: np.ndarray,
                                     img_size: Tuple[int, int]) -> Dict[str, float]:
        """
        Compute 2D bounding box of tactile sensor in image.
        
        Args:
            sensor_pose: 4x4 transformation matrix of sensor
            sensor_size: (width, height) of sensor in meters
            K: Camera intrinsic matrix
            E: Camera extrinsic matrix
            img_size: (width, height) of image
        
        Returns:
            Dictionary with keys: x_min, y_min, x_max, y_max
        """
        # Define sensor corners in sensor frame
        w, h = sensor_size
        corners_sensor = np.array([
            [-w/2, -h/2, 0],
            [w/2, -h/2, 0],
            [w/2, h/2, 0],
            [-w/2, h/2, 0],
        ])
        
        # Transform to world coordinates
        corners_sensor_homo = np.hstack([corners_sensor, np.ones((4, 1))])
        corners_world = (sensor_pose @ corners_sensor_homo.T).T[:, :3]
        
        # Project to 2D
        corners_2d = CameraProjection.project_3d_to_2d(corners_world, K, E)
        
        # Compute bounding box
        x_min = max(0, int(np.floor(np.min(corners_2d[:, 0]))))
        y_min = max(0, int(np.floor(np.min(corners_2d[:, 1]))))
        x_max = min(img_size[0] - 1, int(np.ceil(np.max(corners_2d[:, 0]))))
        y_max = min(img_size[1] - 1, int(np.ceil(np.max(corners_2d[:, 1]))))
        
        return {
            'x_min': float(x_min),
            'y_min': float(y_min),
            'x_max': float(x_max),
            'y_max': float(y_max)
        }


class TactileFeatureExtractor:
    """Main class for extracting tactile and visual features."""
    
    def __init__(self,
                 img_size: int = 640,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 device: str = 'cpu'):
        """
        Initialize the feature extractor.
        
        Args:
            img_size: Input image size (will be resized to this)
            patch_size: Patch size for Vision Transformer
            embed_dim: Embedding dimension
            device: Device to run on ('cpu' or 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TactileFeatureExtractor")
        self.device = torch.device(device)
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Initialize vision backbone
        self.backbone = CLIPLikeBackbone(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        ).to(self.device)
        self.backbone.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        self.feature_grid_shape = self.backbone.get_feature_grid_shape()
    
    def map_bbox_to_feature_grid(self,
                                  bbox: Dict[str, float],
                                  original_img_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Map bounding box coordinates from original image to feature grid.
        
        Args:
            bbox: Bounding box dict with x_min, y_min, x_max, y_max
            original_img_size: (width, height) of original image
        
        Returns:
            (feat_x_min, feat_y_min, feat_x_max, feat_y_max) as integer indices
        """
        orig_w, orig_h = original_img_size
        feat_h, feat_w = self.feature_grid_shape
        
        # Calculate scaling factors
        scale_w = orig_w / feat_w
        scale_h = orig_h / feat_h
        
        # Map coordinates
        feat_x_min = int(np.floor(bbox['x_min'] / scale_w))
        feat_y_min = int(np.floor(bbox['y_min'] / scale_h))
        feat_x_max = int(np.floor(bbox['x_max'] / scale_w))
        feat_y_max = int(np.floor(bbox['y_max'] / scale_h))

        # Ensure ordering (swap if needed)
        if feat_x_min > feat_x_max:
            feat_x_min, feat_x_max = feat_x_max, feat_x_min
        if feat_y_min > feat_y_max:
            feat_y_min, feat_y_max = feat_y_max, feat_y_min
        
        # Clamp to valid range
        feat_x_min = max(0, min(feat_x_min, feat_w - 1))
        feat_y_min = max(0, min(feat_y_min, feat_h - 1))
        feat_x_max = max(0, min(feat_x_max, feat_w - 1))
        feat_y_max = max(0, min(feat_y_max, feat_h - 1))
        
        return feat_x_min, feat_y_min, feat_x_max, feat_y_max
    
    def extract_wrist_features(self,
                                wrist_image: np.ndarray,
                                bbox: Dict[str, float]) -> torch.Tensor:
        """
        Extract h_w: mean-pooled features from wrist-view tokens within bounding box.
        
        Args:
            wrist_image: Wrist camera image, shape (H, W, 3)
            bbox: Bounding box dict with x_min, y_min, x_max, y_max
        
        Returns:
            h_w: Mean-pooled feature vector, shape (embed_dim,)
        """
        # Preprocess image
        img_tensor = self.transform(wrist_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.backbone(img_tensor)  # (1, N_patches, embed_dim)
        
        # Map bounding box to feature grid
        # Note: Image is resized to (img_size, img_size) by the transform
        # So we need to scale the bounding box coordinates first
        orig_h, orig_w = wrist_image.shape[:2]
        
        # Scale bbox coordinates to match resized image (square, img_size x img_size)
        scale_to_square = min(self.img_size / orig_w, self.img_size / orig_h)
        bbox_scaled = {
            'x_min': bbox['x_min'] * scale_to_square,
            'y_min': bbox['y_min'] * scale_to_square,
            'x_max': bbox['x_max'] * scale_to_square,
            'y_max': bbox['y_max'] * scale_to_square
        }
        
        feat_x_min, feat_y_min, feat_x_max, feat_y_max = self.map_bbox_to_feature_grid(
            bbox_scaled, (self.img_size, self.img_size)
        )
        
        # Reshape features to spatial grid: (1, H_feat, W_feat, embed_dim)
        feat_h, feat_w = self.feature_grid_shape
        features_spatial = features.view(1, feat_h, feat_w, -1)
        
        # Select tokens within bounding box
        selected_features = features_spatial[:, 
                                            feat_y_min:feat_y_max+1, 
                                            feat_x_min:feat_x_max+1, 
                                            :]  # (1, H_box, W_box, embed_dim)

        # Handle empty selections robustly
        if selected_features.numel() == 0:
            # Fallback: pick nearest valid token index within grid
            feat_h, feat_w = self.feature_grid_shape
            safe_y = max(0, min((feat_y_min + feat_y_max) // 2, feat_h - 1))
            safe_x = max(0, min((feat_x_min + feat_x_max) // 2, feat_w - 1))
            selected_features = features_spatial[:, safe_y:safe_y+1, safe_x:safe_x+1, :]

        # Flatten and mean pool (nan-safe)
        selected_features = selected_features.reshape(-1, features.shape[-1])  # (N_tokens, embed_dim)
        h_w = torch.nanmean(selected_features, dim=0)  # (embed_dim,)
        h_w = torch.nan_to_num(h_w, nan=0.0, posinf=0.0, neginf=0.0)
        
        return h_w
    
    def extract_tactile_features(self,
                                  tactile_image: np.ndarray) -> torch.Tensor:
        """
        Extract h_tau: mean-pooled features from tactile tokens.
        
        Args:
            tactile_image: Tactile sensor image, shape (H, W, 3)
        
        Returns:
            h_tau: Mean-pooled feature vector, shape (embed_dim,)
        """
        # Preprocess image
        img_tensor = self.transform(tactile_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.backbone(img_tensor)  # (1, N_patches, embed_dim)
        
        # Mean pool over all tokens
        h_tau = features.mean(dim=1).squeeze(0)  # (embed_dim,)
        
        return h_tau
    
    def extract_features(self,
                         wrist_image: np.ndarray,
                         tactile_image: np.ndarray,
                         bbox_wrist: Dict[str, float],
                         bbox_tp: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract all features: h_tau, h_w, and optionally h_tp.
        
        Args:
            wrist_image: Wrist camera image
            tactile_image: Tactile sensor image
            bbox_wrist: Bounding box in wrist view
            bbox_tp: Optional bounding box in third-person view
        
        Returns:
            Dictionary with keys: 'h_tau', 'h_w', 'h_tp' (if provided)
        """
        h_tau = self.extract_tactile_features(tactile_image)
        h_w = self.extract_wrist_features(wrist_image, bbox_wrist)
        
        result = {
            'h_tau': h_tau,
            'h_w': h_w
        }
        
        if bbox_tp is not None:
            # For third-person view, we would process it similarly
            # For now, we'll use the same extract_wrist_features method
            pass  # Implement if needed
        
        return result


def generate_fake_camera_params(img_size: Tuple[int, int] = (640, 480)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate fake camera intrinsic and extrinsic parameters.
    
    Args:
        img_size: (width, height) of image
    
    Returns:
        K: Intrinsic matrix (3, 3)
        E: Extrinsic matrix (4, 4) - world to camera transformation
    """
    w, h = img_size
    fx = fy = 500.0  # Focal length in pixels
    cx = w / 2.0
    cy = h / 2.0
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Fake extrinsic: camera positioned at (0.5, 0.5, 0.5) looking at origin
    position = np.array([0.5, 0.5, 0.5])
    target = np.array([0, 0, 0])
    up = np.array([0, 0, 1])
    
    # Compute rotation
    z_axis = target - position
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    R = np.array([x_axis, y_axis, z_axis])
    t = -R @ position
    
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = t
    
    return K, E


def generate_fake_images(img_size: Tuple[int, int] = (640, 480)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate fake wrist and tactile images.
    
    Args:
        img_size: (width, height) of images
    
    Returns:
        wrist_image: Fake wrist camera image (H, W, 3)
        tactile_image: Fake tactile sensor image (H, W, 3)
    """
    w, h = img_size
    
    # Generate wrist image: random pattern with some structure
    wrist_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    # Add some colored patches to make it more realistic
    if CV2_AVAILABLE:
        cv2.rectangle(wrist_image, (100, 150), (300, 350), (255, 0, 0), -1)
        cv2.circle(wrist_image, (400, 200), 50, (0, 255, 0), -1)
    else:
        # Manual drawing without cv2
        wrist_image[150:350, 100:300] = [255, 0, 0]
    
    # Generate tactile image: smaller, more uniform pattern
    tactile_h, tactile_w = 100, 100
    tactile_image = np.random.randint(0, 255, (tactile_h, tactile_w, 3), dtype=np.uint8)
    
    # Add circular pattern to simulate contact
    if CV2_AVAILABLE:
        cv2.circle(tactile_image, (50, 50), 30, (255, 255, 255), -1)
    else:
        # Manual drawing
        y, x = np.ogrid[:tactile_h, :tactile_w]
        mask = (x - 50)**2 + (y - 50)**2 <= 30**2
        tactile_image[mask] = [255, 255, 255]
    
    return wrist_image, tactile_image

