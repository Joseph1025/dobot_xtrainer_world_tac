# Quick Start: HSA Loss Training

## TL;DR

Train ACT policy with tactile-visual feature alignment in 3 steps:

```bash
# 1. Update task config (add tactile cameras)
# Edit ModelTrain/constants.py

# 2. Run training with HSA
python ModelTrain/model_train_with_hsa.py \
    --task_name your_task \
    --ckpt_dir ./ckpt/experiment_hsa \
    --enable_hsa \
    --batch_size 16 \
    --num_steps 30000

# 3. Monitor losses
# HSA loss should decrease from ~4.0 to ~1.0
```

## What is HSA Loss?

HSA (Hard Sample Aware) Loss aligns tactile sensor features with wrist camera features using contrastive learning:

```
L_HSA = -log(exp(h_tau·h_w/κ) / (exp(h_tau·h_w/κ) + Σ exp(h_tau·h_neg/κ)))
```

**Benefits:**
- Better tactile-visual correspondence
- Improved manipulation with tactile feedback
- More robust multi-modal representations

## Prerequisites

### 1. Dataset Requirements

Your HDF5 dataset must contain:

```
/observations/images/top              # RGB camera
/observations/images/left_wrist       # RGB camera
/observations/images/right_wrist      # RGB camera
/observations/tactile_left            # Tactile sensor ← NEW
/observations/qpos                    # Joint angles (first 6 values)
/action                               # Robot actions
```

### 2. Task Configuration

Edit `ModelTrain/constants.py`:

```python
TASK_CONFIGS = {
    'your_task': {
        'dataset_dir': '/path/to/dataset',
        'episode_len': 1000,
        'train_ratio': 0.98,
        'camera_names': ['top', 'left_wrist', 'right_wrist'],
        'tactile_camera_names': ['tactile_left'],  # ← ADD THIS LINE
    }
}
```

## Training Commands

### Basic (Recommended for First Try)

```bash
python ModelTrain/model_train_with_hsa.py \
    --task_name your_task \
    --ckpt_dir ./ckpt/exp_hsa_basic \
    --enable_hsa \
    --batch_size 16 \
    --num_steps 30000
```

### With Custom HSA Parameters

```bash
python ModelTrain/model_train_with_hsa.py \
    --task_name your_task \
    --ckpt_dir ./ckpt/exp_hsa_custom \
    --enable_hsa \
    --hsa_weight 1.5 \
    --hsa_temperature 0.08 \
    --robot_type "Nova 2" \
    --wrist_camera left_wrist \
    --batch_size 16 \
    --num_steps 30000
```

### All Parameters

```bash
python ModelTrain/model_train_with_hsa.py \
    --task_name your_task \
    --ckpt_dir ./ckpt/exp_hsa_full \
    --enable_hsa \
    --hsa_weight 1.0 \
    --hsa_temperature 0.07 \
    --hsa_img_size 224 \
    --hsa_feature_dim 768 \
    --robot_type "Nova 2" \
    --wrist_camera left_wrist \
    --batch_size 16 \
    --num_steps 30000 \
    --lr 2e-5 \
    --kl_weight 10 \
    --chunk_size 45 \
    --hidden_dim 512
```

## Expected Training Behavior

### Loss Values Over Time

```
Step    | L1 Loss | KL Loss | HSA Loss | Total Loss
--------|---------|---------|----------|------------
0       | 2.500   | 8.000   | 4.500    | 10.500
5000    | 1.200   | 3.000   | 3.000    | 7.000
10000   | 0.500   | 1.000   | 1.500    | 3.000
20000   | 0.200   | 0.500   | 0.800    | 1.500
30000   | 0.100   | 0.300   | 0.500    | 0.900
```

### What to Watch

✅ **Good:**
- HSA loss decreases steadily
- L1 loss converges as usual
- Total loss decreases smoothly

⚠️ **Warning:**
- HSA loss stays at 4.0+ after 10k steps → increase `hsa_weight`
- L1 loss unstable → decrease `hsa_weight`
- HSA loss is NaN → increase `temperature` or check data

## Key Parameters Explained

| Parameter | What It Does | Typical Range | Default |
|-----------|--------------|---------------|---------|
| `--enable_hsa` | Turn on HSA loss | flag | False |
| `--hsa_weight` | How much to weight HSA vs ACT | 0.3 - 2.0 | 1.0 |
| `--hsa_temperature` | Contrastive loss hardness | 0.05 - 0.15 | 0.07 |
| `--hsa_img_size` | Feature extraction resolution | 112 - 224 | 224 |
| `--robot_type` | For forward kinematics | "Nova 2", "Nova 5" | "Nova 2" |
| `--wrist_camera` | Which camera views sensor | camera name | "left_wrist" |

## Testing Before Training

### 1. Test HSA Loss Module

```bash
python -m dobot_control.hsa_loss
```

Expected output:
```
[Test 1] Basic HSA loss with in-batch negatives
  Loss: 1.8813
...
All tests passed!
```

### 2. Test Feature Extraction

```bash
python examples/example_tactile_feature_extraction.py
```

Expected output:
```
[Step 4] Extracting features using CLIP-like backbone...
  - h_tau shape: torch.Size([768])
  - h_w shape: torch.Size([768])
```

### 3. Run All Examples

```bash
python examples/example_hsa_training.py
```

## Troubleshooting

### "Cannot find tactile camera in dataset"

**Fix:** Add `tactile_camera_names` to task config in `constants.py`

### "HSA loss is NaN"

**Fix:** 
```bash
--hsa_temperature 0.1 --hsa_weight 0.5
```

### "Out of memory"

**Fix:**
```bash
--batch_size 8 --hsa_img_size 112
```

### "HSA loss not decreasing"

**Fix:**
1. Check tactile images are valid (not blank)
2. Verify joint angles in qpos (not all zeros)
3. Increase `--hsa_weight 2.0`

## Comparing Results

### Without HSA (Baseline)

```bash
# Standard ACT training
python ModelTrain/model_train.py \
    --task_name your_task \
    --ckpt_dir ./ckpt/baseline_no_hsa \
    --batch_size 16 \
    --num_steps 30000
```

### With HSA (Improved)

```bash
# HSA-enhanced training
python ModelTrain/model_train_with_hsa.py \
    --task_name your_task \
    --ckpt_dir ./ckpt/with_hsa \
    --enable_hsa \
    --batch_size 16 \
    --num_steps 30000
```

**Compare:**
- Final validation loss
- Success rate on real robot
- Robustness to variations
- Tactile feedback utilization

## File Overview

```
Key Files:
├── dobot_control/
│   ├── hsa_loss.py                    # HSA loss implementation
│   └── tactile_feature_extraction.py  # Feature extraction
├── ModelTrain/
│   ├── module/
│   │   ├── policy_with_hsa.py         # Enhanced ACT policy
│   │   └── train_module_with_hsa.py   # Training loop
│   └── model_train_with_hsa.py        # Main training script ← RUN THIS
└── examples/
    └── example_hsa_training.py        # Usage examples

Documentation:
├── QUICKSTART_HSA.md                  # This file (start here)
├── HSA_LOSS_GUIDE.md                  # Complete guide
└── HSA_IMPLEMENTATION_SUMMARY.md      # Technical details
```

## Next Steps

1. ✅ Test the implementation:
   ```bash
   python -m dobot_control.hsa_loss
   ```

2. ✅ Update your task config:
   ```python
   # Edit ModelTrain/constants.py
   'tactile_camera_names': ['tactile_left']
   ```

3. ✅ Run training:
   ```bash
   python ModelTrain/model_train_with_hsa.py \
       --task_name your_task \
       --enable_hsa \
       --batch_size 16
   ```

4. ✅ Monitor training:
   - Check HSA loss decreases
   - Watch for convergence
   - Compare with baseline

5. ✅ Evaluate on robot:
   - Test manipulation tasks
   - Compare success rates
   - Assess tactile utilization

## Getting Help

1. **Check examples:** `python examples/example_hsa_training.py`
2. **Read guide:** `HSA_LOSS_GUIDE.md`
3. **Enable debug:** Add `print()` statements in `policy_with_hsa.py`
4. **Test components:** Run unit tests individually

## Summary

**What HSA Loss Does:**
- Aligns tactile sensor features with wrist camera features
- Uses contrastive learning (like CLIP)
- Improves multi-modal representations

**How to Use:**
1. Add tactile cameras to task config
2. Run `model_train_with_hsa.py` with `--enable_hsa`
3. Monitor HSA loss (should decrease to ~1.0)

**When It Helps:**
- Tasks requiring tactile feedback
- Manipulation with uncertain contact
- Learning robust visual-tactile correspondence

**Start Here:** Basic command above, then tune `hsa_weight` based on results.

Good luck! 🚀

