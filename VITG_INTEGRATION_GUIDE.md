# V-JEPA2 ViTG Integration Guide

This guide explains how to use the V-JEPA2 ViTG encoder for processing tactile images in the ACT policy.

## Overview

The integration replaces ResNet18 visual backbones with a frozen V-JEPA2 ViTG encoder that processes tactile images and produces 1280-dimensional embeddings. The ViTG encoder remains frozen during training, while the projection layers and ACT policy are trainable.

## Architecture Changes

### Key Components

1. **ViTG Encoder Wrapper** (`ModelTrain/module/vitg_encoder.py`)
   - Loads V-JEPA2 ViTG checkpoint
   - Freezes all parameters (requires_grad=False)
   - Outputs 1280-dim CLS token embeddings

2. **Modified DETR Architecture** (`ModelTrain/detr/models/detr_vae.py`)
   - Supports both ResNet and ViTG backbones
   - Projects ViTG embeddings (1280→hidden_dim)
   - Handles global vs spatial features appropriately

3. **Dataset Preprocessing** (`ModelTrain/module/utils.py`)
   - Automatically resizes tactile images to 224×224 for ViTG
   - Maintains compatibility with existing HDF5 format

## Usage

### Training with ViTG

To train an ACT policy with ViTG for tactile processing:

```bash
python ModelTrain/model_train.py \
    --task_name dobot_peg_fixed_tactile \
    --ckpt_dir ./ckpt/tactile_vitg_experiment \
    --use_vitg \
    --vitg_ckpt_path /path/to/vjepa2_vitg.pt \
    --batch_size 16 \
    --num_steps 30000 \
    --lr 2e-5 \
    --chunk_size 45 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --kl_weight 10
```

### Key Arguments

- `--use_vitg`: Enable ViTG encoder (boolean flag)
- `--vitg_ckpt_path`: Path to your V-JEPA2 ViTG checkpoint file (.pt)
- `--task_name`: Task configuration (e.g., `dobot_peg_fixed_tactile`)

### Training without ViTG (Standard ResNet)

Simply omit the `--use_vitg` flag to use the standard ResNet18 backbones:

```bash
python ModelTrain/model_train.py \
    --task_name dobot_peg_fixed_tactile \
    --ckpt_dir ./ckpt/tactile_resnet_experiment \
    --batch_size 16 \
    --num_steps 30000
```

## Inference

The inference code automatically detects whether a model was trained with ViTG based on the saved policy configuration. No changes needed to inference scripts.

```python
from ModelTrain.module.model_module import Imitate_Model

# Load model (automatically handles ViTG if used during training)
model = Imitate_Model(
    ckpt_dir='./ckpt/tactile_vitg_experiment',
    ckpt_name='policy_last.ckpt'
)
model.loadModel()

# Run inference
action = model.predict(observation, timestep)
```

## Dataset Requirements

### HDF5 Structure

Tactile images should be stored in the same format as camera images:

```
/observations/images/left_wrist    # Tactile sensor on left wrist
/observations/images/right_wrist   # Tactile sensor on right wrist
/observations/images/top           # Optional: overhead camera or tactile
/observations/qpos                 # Joint positions
/action                            # Actions
```

### Image Specifications

- **Format**: HDF5 arrays with shape (H, W, C)
- **Channels**: RGB (3 channels)
- **Resolution**: Any size (automatically resized to 224×224)
- **Data type**: uint8 (0-255) or float32 (0.0-1.0)

## Task Configuration

Add or modify task configurations in `ModelTrain/constants.py`:

```python
TASK_CONFIGS = {
    'dobot_peg_fixed_tactile': {
        'dataset_dir': DATA_DIR + '/dobot_peg_fixed_tactile/train_data',
        'episode_len': 700,
        'train_ratio': 0.9,
        'camera_names': ['left_wrist', 'right_wrist']  # Tactile sensor names
    },
}
```

## Technical Details

### ViTG Encoder

- **Architecture**: Vision Transformer Giant (ViT-G)
- **Embedding Dimension**: 1280
- **Input Size**: 224×224 (automatically resized)
- **Output**: CLS token embedding per tactile sensor
- **Training**: Frozen (no gradients)

### Projection Layer

The ViTG embeddings are projected to the transformer's hidden dimension:

```
ViTG Output (1280-dim) → Linear Projection → Transformer Hidden Dim (512-dim default)
```

This projection layer is **trainable** and learns to adapt ViTG features for the specific manipulation task.

### Position Embeddings

Since ViTG produces global embeddings (not spatial features), learned position embeddings are used to distinguish between different tactile sensors.

### Optimizer Configuration

When using ViTG:
- ViTG encoder parameters: **Excluded** from optimizer (frozen)
- Projection layer: Standard learning rate
- Transformer & policy: Standard learning rate

When using ResNet:
- ResNet backbones: Lower learning rate (1e-5)
- Rest of model: Standard learning rate (2e-5)

## Troubleshooting

### Issue: Checkpoint Loading Error

**Problem**: ViTG checkpoint format incompatible

**Solution**: The checkpoint must contain the full model, not just state_dict. Check the checkpoint structure:

```python
import torch
ckpt = torch.load('vjepa2_vitg.pt')
print(type(ckpt))
print(ckpt.keys() if isinstance(ckpt, dict) else "Full model")
```

If only state_dict is available, you'll need to implement the V-JEPA2 model architecture in `vitg_encoder.py`.

### Issue: Out of Memory

**Problem**: ViTG model is large (>1B parameters)

**Solutions**:
- Reduce batch size
- Use gradient checkpointing
- Ensure ViTG is properly frozen (no gradients stored)

### Issue: Images Wrong Size

**Problem**: Tactile images not 224×224

**Solution**: The code automatically resizes images. Verify transformations are applied:
- Training: Check dataset prints "Dataset configured for ViTG"
- Inference: Check policy_config contains `use_vitg: True`

### Issue: Poor Performance

**Potential Causes**:
1. ViTG checkpoint not trained on tactile data (domain mismatch)
2. Projection layer needs more capacity
3. Learning rate too high/low for projection layer

**Solutions**:
- Fine-tune projection layer learning rate
- Add more projection layers
- Experiment with different hidden_dim values

## Performance Tips

1. **Batch Size**: Start with smaller batch sizes (8-16) as ViTG forward passes are memory-intensive
2. **Learning Rate**: The default 2e-5 should work well for the projection layer
3. **Training Steps**: May need longer training than ResNet baseline (ViTG features are pretrained but projection needs to adapt)
4. **Multiple Sensors**: Each tactile sensor gets its own ViTG encoder instance, increasing memory usage

## Comparison: ViTG vs ResNet

| Aspect | ViTG | ResNet18 |
|--------|------|----------|
| Parameters (trainable) | ~5M (projection only) | ~11M (full backbone) |
| Parameters (total) | ~1B (frozen) + ~5M | ~11M |
| Memory (forward) | High | Low |
| Memory (backward) | Low (frozen) | Medium |
| Input Size | 224×224 | 480×640 (original) |
| Feature Type | Global embedding | Spatial features |
| Pretraining | V-JEPA2 (tactile) | ImageNet (RGB) |

## Citation

If you use this integration, consider citing:

- ACT (Action Chunking Transformer)
- V-JEPA (Visual Joint Embedding Predictive Architecture)
- Your ViTG checkpoint source

## Example: Complete Training Script

```bash
#!/bin/bash

# Configuration
TASK_NAME="dobot_peg_fixed_tactile"
CKPT_DIR="./ckpt/vitg_tactile_$(date +%Y%m%d_%H%M%S)"
VITG_CKPT="/path/to/vjepa2_vitg.pt"

# Train with ViTG
python ModelTrain/model_train.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir ${CKPT_DIR} \
    --use_vitg \
    --vitg_ckpt_path ${VITG_CKPT} \
    --batch_size 16 \
    --num_steps 30000 \
    --lr 2e-5 \
    --kl_weight 10 \
    --chunk_size 45 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --seed 42 \
    --save_every 5000 \
    --validate_every 500

echo "Training complete! Checkpoint saved to: ${CKPT_DIR}"
```

## Next Steps

1. Test with your V-JEPA2 ViTG checkpoint
2. Verify checkpoint loads correctly
3. Run a small training experiment (1000 steps)
4. Compare with ResNet baseline
5. Tune hyperparameters for your specific task

