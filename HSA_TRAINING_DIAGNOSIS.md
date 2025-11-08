# HSA Training Diagnosis

## Problem: HSA Loss Not Converging

Looking at `train_hsa_loss.png`, the HSA loss is:
- **Stuck at ~2.6-2.9** (no downward trend)
- **High variance** (±0.3 fluctuation)
- **No learning** over 25,000 steps

For reference, with temperature=0.1 and batch_size=16:
- **Random features**: log(16) ≈ 2.77
- **Your loss**: 2.6-2.9 → **Features are essentially random!**

## Diagnostic Checks Added

### 1. **CLIP Parameter Verification**
Added to `policy_jepa_adapter_with_hsa.py` lines 381-395:
```python
trainable_params = [p for p in feature_params if p.requires_grad]
frozen_params = [p for p in feature_params if not p.requires_grad]
```

**Check logs for**:
- "Added X CLIP params to optimizer"
- If X < 1000: **Problem!** - Not all parameters registered
- Expected: 80M+ params for ViT-Large

### 2. **Gradient Flow Check**
Added to `policy_jepa_adapter_with_hsa.py` lines 344-351:
```python
print(f"  h_tau requires_grad: {h_tau.requires_grad}")
print(f"  h_w requires_grad: {h_w.requires_grad}")
print(f"  HSA loss requires_grad: {hsa_loss_dict['hsa_total'].requires_grad}")
```

**Should all be True!** If False, gradients won't flow.

### 3. **Module Breakdown**
Added lines 152-158 to show parameter count per CLIP component:
```
CLIP Module Breakdown:
  conv_proj: X params
  transformer: X params
  ln_post: X params
```

Should show millions of parameters.

## Possible Root Causes

### Issue 1: Only Getting 150 Parameters ❌
From earlier log: **"Added 150 feature extractor params"**

This is WRONG! Should be ~86M params for ViT-Base or ~300M for ViT-Large.

**Hypothesis**: `list(self.feature_extractor.backbone.parameters())` only getting top-level params, not recursive.

**Fix**: Use `self.feature_extractor.backbone.parameters()` directly (already done).

### Issue 2: Image Normalization Bug ✅ FIXED
**Was**: Converting normalized images [−2.5, 2.5] to uint8 → corrupted
**Now**: Saving unnormalized [0, 1] images before base policy normalizes

### Issue 3: Learning Rate Too Low ✅ FIXED
**Was**: CLIP LR = 5e-6 (0.5× base)
**Now**: CLIP LR = 1e-5 (1.0× base)

### Issue 4: Temperature Too Low ✅ FIXED
**Was**: Temperature = 0.07 (hard negatives, noisy)
**Now**: Temperature = 0.1 (softer, more stable)

### Issue 5: Parameters Not Trainable? ❓
**Check**: Are CLIP parameters frozen somewhere?

Look for:
- `.eval()` mode during training
- `.requires_grad = False` assignments
- `torch.no_grad()` contexts

### Issue 6: Gradient Detachment? ❓
**Check**: Are features being detached before HSA loss?

Already verified: No `.detach()` calls found.

## Run Diagnostics

### Step 1: Run Pipeline Test
```bash
cd /home/zexi/Dev/dobot_xtrainer_world_tac
python test_hsa_pipeline.py
```

**Expected output**:
```
✓ Features have requires_grad=True
✓ All X CLIP params received gradients
✓ Gradients are flowing (norm: > 0)
✓ Parameters updated after optimizer step
🎉 HSA pipeline is working correctly!
```

### Step 2: Check Training Logs
Restart training and look for:

```
HSA Loss Configuration
  ...
  Third-Person Camera: Enabled (using hardcoded calibration)

CLIP Backbone Parameters
  Total Parameters: 86,567,424  ← Should be millions
  Trainable Parameters: 86,567,424  ← Should match total
  Frozen Parameters: 0  ← Should be 0
  
  CLIP Module Breakdown:  ← Check all modules listed
    conv_proj: X params
    transformer: Y params (should be largest)
    ln_post: Z params

Added X CLIP params to optimizer with LR=1.00e-05  ← X should be 100-300
```

**AND on first batch**:
```
[HSA Gradient Check]
  h_tau requires_grad: True  ← Must be True
  h_w requires_grad: True    ← Must be True
  HSA loss requires_grad: True  ← Must be True
  Total loss requires_grad: True  ← Must be True
```

### Step 3: Monitor HSA Loss Trajectory

With all fixes, expected behavior:
```
Steps 0-100:    2.8 → 2.5 (initial learning)
Steps 100-1000: 2.5 → 1.8 (fast convergence)
Steps 1000-5000: 1.8 → 1.0 (refinement)
Steps 5000+:    1.0 → 0.6 (fine-tuning)
```

If still flat → deeper issue with gradient computation or feature extraction.

## Next Steps

1. ✅ Run `test_hsa_pipeline.py` to verify basic functionality
2. ✅ Restart training and check diagnostic outputs
3. ✅ Watch first 1000 steps - should see decreasing trend
4. If still flat after fixes:
   - Add gradient norm logging per step
   - Print feature statistics (mean, std) to check if changing
   - Verify dataset has diversity (not all same images)

## Emergency Fallback: Simpler Loss

If nothing works, try simplest possible alignment loss:
```python
# Simple L2 feature alignment loss
def simple_alignment_loss(h_tau, h_w):
    return F.mse_loss(h_tau, h_w)
```

This should **definitely** converge. If it doesn't, issue is with gradient flow, not the HSA loss itself.

