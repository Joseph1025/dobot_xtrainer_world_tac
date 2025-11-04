# HSA Loss Integration Guide

## Overview

This guide explains how to use **HSA (Hard Sample Aware) Loss** for tactile-visual feature alignment in ACT policy training.

The HSA loss aligns features extracted from tactile sensors with features from wrist camera images using contrastive learning:

$$
\mathcal{L}_{\text{HSA-W}} = -\log \frac{\exp(h_\tau \cdot h_w / \kappa)}{\exp(h_\tau \cdot h_w / \kappa) + \sum_{i=1}^{N_k} \exp(h_\tau \cdot h_{w,i}^{\text{neg}} / \kappa)}
$$

where:
- $h_\tau$: tactile features extracted from tactile sensor images
- $h_w$: wrist visual features from wrist camera images
- $h_{w,i}^{\text{neg}}$: negative samples (from other samples in the batch)
- $\kappa$: temperature parameter (typically 0.07)

## Architecture

### Key Components

1. **Tactile Feature Extractor** (`dobot_control/tactile_feature_extraction.py`)
   - CLIP-like Vision Transformer backbone
   - Extracts features from tactile sensor images
   - Computes 3D sensor pose using forward kinematics
   - Projects sensor location to 2D bounding boxes in camera views

2. **HSA Loss Module** (`dobot_control/hsa_loss.py`)
   - Implements contrastive loss for feature alignment
   - Supports in-batch negatives and hard negatives
   - Optional third-person camera integration

3. **Enhanced ACT Policy** (`ModelTrain/module/policy_with_hsa.py`)
   - Extends standard ACT policy with HSA loss computation
   - Integrates tactile feature extraction into training loop
   - Combines ACT loss (L1 + KL) with HSA loss

4. **Enhanced Training Module** (`ModelTrain/module/train_module_with_hsa.py`)
   - Modified training loop supporting HSA loss
   - Handles tactile image data alongside RGB images
   - Updates loss computation and logging

## Quick Start

### 1. Basic Training with HSA Loss

```bash
python ModelTrain/model_train_with_hsa.py \
    --task_name dobot_pick_random_1013 \
    --ckpt_dir ./ckpt/dobot_pick_hsa \
    --enable_hsa \
    --batch_size 16 \
    --num_steps 30000
```

### 2. With Custom HSA Parameters

```bash
python ModelTrain/model_train_with_hsa.py \
    --task_name dobot_pick_random_1013 \
    --ckpt_dir ./ckpt/dobot_pick_hsa \
    --enable_hsa \
    --hsa_weight 2.0 \
    --hsa_temperature 0.1 \
    --hsa_img_size 224 \
    --hsa_feature_dim 768 \
    --robot_type "Nova 2" \
    --wrist_camera left_wrist \
    --batch_size 16 \
    --num_steps 30000 \
    --lr 2e-5 \
    --kl_weight 10
```

## Configuration Parameters

### Standard ACT Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--task_name` | str | required | Task name from `constants.py` |
| `--ckpt_dir` | str | required | Checkpoint directory |
| `--batch_size` | int | 16 | Training batch size |
| `--num_steps` | int | 30000 | Total training steps |
| `--lr` | float | 2e-5 | Learning rate |
| `--kl_weight` | int | 10 | KL divergence weight |
| `--chunk_size` | int | 45 | Action sequence length |
| `--hidden_dim` | int | 512 | Transformer hidden dimension |

### HSA Loss Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable_hsa` | flag | False | Enable HSA loss |
| `--hsa_weight` | float | 1.0 | Weight for HSA loss term |
| `--hsa_temperature` | float | 0.07 | Temperature for contrastive loss |
| `--hsa_img_size` | int | 224 | Image size for feature extraction |
| `--hsa_feature_dim` | int | 768 | Feature dimension (ViT) |
| `--robot_type` | str | "Nova 2" | Robot type: "Nova 2" or "Nova 5" |
| `--wrist_camera` | str | "left_wrist" | Name of wrist camera |

## Dataset Requirements

Your dataset must include:

1. **RGB Camera Images**: Standard wrist and third-person cameras
   - Location: `/observations/images/{camera_name}` in HDF5 files
   
2. **Tactile Sensor Images**: Images from tactile sensors
   - Location: `/observations/{tactile_name}` in HDF5 files
   
3. **Joint Angles**: Robot joint positions (qpos)
   - Location: `/observations/qpos` in HDF5 files
   - First 6 values should be joint angles in radians

4. **Actions**: Robot actions
   - Location: `/action` in HDF5 files

### Task Configuration

Update `ModelTrain/constants.py` to include tactile cameras:

```python
TASK_CONFIGS = {
    'your_task_name': {
        'dataset_dir': '/path/to/dataset',
        'episode_len': 1000,
        'train_ratio': 0.98,
        'camera_names': ['top', 'left_wrist', 'right_wrist'],
        'tactile_camera_names': ['tactile_left', 'tactile_right']  # Add this
    }
}
```

## Loss Formulation

The total training loss combines three terms:

```
Total Loss = L1_loss + KL_weight × KL_loss + HSA_weight × HSA_loss
```

Where:
- **L1_loss**: Action prediction error (standard ACT)
- **KL_loss**: KL divergence for VAE regularization (standard ACT)
- **HSA_loss**: Tactile-visual feature alignment (new)

### Loss Values Interpretation

| Loss Range | Interpretation |
|------------|----------------|
| 4.0 - 6.0 | Random features (initialization) |
| 2.0 - 3.0 | Partial alignment (mid-training) |
| 0.5 - 1.5 | Good alignment (well-trained) |
| < 0.5 | Excellent alignment (optimal) |

## Examples

### Running Examples

```bash
# Test HSA loss module
python -m dobot_control.hsa_loss

# Test tactile feature extraction
python examples/example_tactile_feature_extraction.py

# Run all HSA training examples
python examples/example_hsa_training.py
```

### Python API Usage

```python
from ModelTrain.module.policy_with_hsa import ACTPolicyWithHSA, create_default_hsa_config

# Create HSA configuration
hsa_config = create_default_hsa_config(
    enable_hsa=True,
    hsa_weight=1.0,
    temperature=0.07,
    img_size=224,
    wrist_camera_name='left_wrist',
    camera_names=['top', 'left_wrist', 'right_wrist'],
    robot_type='Nova 2'
)

# Create ACT configuration
act_config = {
    'lr': 1e-5,
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['top', 'left_wrist', 'right_wrist'],
    'tactile_camera_names': ['tactile_left'],
    'action_dim': 16,
    'no_encoder': False,
}

# Create policy with HSA
policy = ACTPolicyWithHSA(act_config, hsa_config)
```

## How It Works

### Training Pipeline

1. **Data Loading**
   ```
   Dataset → RGB images + Tactile images + Joint angles + Actions
   ```

2. **Feature Extraction**
   ```
   Joint angles → Forward Kinematics → Sensor 3D Pose
   Sensor Pose + Camera Params → 2D Bounding Box
   Wrist Image + Bounding Box → h_w (wrist features)
   Tactile Image → h_tau (tactile features)
   ```

3. **Loss Computation**
   ```
   ACT forward → L1 loss + KL loss
   h_tau, h_w → HSA loss
   Total loss = L1 + KL_weight × KL + HSA_weight × HSA
   ```

4. **Optimization**
   ```
   Total loss → Backprop → Update ACT model + Feature extractor
   ```

### Forward Kinematics

The system uses Denavit-Hartenberg (DH) parameters to compute the 3D pose of the tactile sensor:

```python
sensor_pose = ForwardKinematics.compute_tactile_sensor_pose(
    joint_angles=qpos[:6],  # First 6 joint angles
    robot_type="Nova 2",     # or "Nova 5"
    sensor_offset=[0, 0, 0.02]  # Offset from end-effector
)
```

### Camera Projection

Projects 3D sensor pose to 2D bounding box in camera view:

```python
bbox = CameraProjection.compute_sensor_bounding_box(
    sensor_pose=sensor_pose,
    sensor_size=(0.04, 0.04),  # 4cm × 4cm
    K=camera_intrinsic,
    E=camera_extrinsic,
    img_size=(640, 480)
)
```

## Troubleshooting

### Common Issues

1. **"Cannot find tactile camera in dataset"**
   - Ensure `tactile_camera_names` is set in task config
   - Verify tactile images exist in HDF5 files at `/observations/{tactile_name}`

2. **"HSA loss is NaN"**
   - Check that joint angles are valid (not all zeros)
   - Verify camera parameters are reasonable
   - Try reducing `hsa_weight` or increasing `hsa_temperature`

3. **"Feature extractor out of memory"**
   - Reduce `hsa_img_size` (e.g., from 224 to 112)
   - Reduce `batch_size`
   - Use gradient checkpointing (implement if needed)

4. **"HSA loss not decreasing"**
   - Ensure tactile and wrist cameras have good views of the sensor
   - Check that sensor offset is correct
   - Try increasing `hsa_weight`
   - Verify forward kinematics is computing correct poses

### Debug Mode

Add debug prints in forward pass:

```python
# In policy_with_hsa.py, extract_tactile_visual_features()
print(f"Joint angles: {joint_angles}")
print(f"Sensor pose: {sensor_pose[:3, 3]}")
print(f"Bounding box: {bbox_w}")
print(f"h_tau: mean={h_tau.mean():.3f}, std={h_tau.std():.3f}")
print(f"h_w: mean={h_w.mean():.3f}, std={h_w.std():.3f}")
```

## Performance Tips

1. **Start with standard ACT training** (no HSA) to ensure basic setup works
2. **Add HSA gradually** with low weight (0.1-0.5) and increase
3. **Monitor all losses** to ensure HSA doesn't overwhelm ACT losses
4. **Use larger batches** (16-32) for better negative sampling
5. **Tune temperature** based on loss values:
   - Too high (>0.2): Loss too easy, weak alignment
   - Too low (<0.03): Loss too hard, unstable training
   - Sweet spot: 0.05-0.1

## Citation

If you use HSA loss in your research, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Conference/Journal},
  year={2025}
}
```

## File Structure

```
dobot_xtrainer_world_tac/
├── dobot_control/
│   ├── tactile_feature_extraction.py  # Feature extraction
│   └── hsa_loss.py                     # HSA loss implementation
├── ModelTrain/
│   ├── module/
│   │   ├── policy_with_hsa.py          # Enhanced ACT policy
│   │   └── train_module_with_hsa.py    # Enhanced training loop
│   └── model_train_with_hsa.py         # Training script
├── examples/
│   ├── example_tactile_feature_extraction.py
│   └── example_hsa_training.py         # HSA training examples
└── HSA_LOSS_GUIDE.md                   # This file
```

## Support

For issues or questions:
1. Check the examples in `examples/example_hsa_training.py`
2. Review the troubleshooting section above
3. Run debug mode to inspect intermediate values
4. Open an issue on GitHub with:
   - Your configuration
   - Error messages
   - Debug output

## License

Same as the main project. See LICENSE file.

