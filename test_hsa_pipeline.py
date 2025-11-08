#!/usr/bin/env python3
"""
Quick test to verify HSA training pipeline is working correctly.
"""

import torch
import numpy as np
from dobot_control.tactile_feature_extraction import TactileFeatureExtractor
from dobot_control.hsa_loss import HSALossWithThirdPerson

def test_hsa_pipeline():
    """Test that HSA loss can backpropagate through CLIP."""
    
    print("="*60)
    print("HSA Pipeline Test")
    print("="*60)
    
    # Initialize components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    feature_extractor = TactileFeatureExtractor(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        device=device
    )
    
    hsa_loss_fn = HSALossWithThirdPerson(
        temperature=0.1,
        use_third_person=False,
        reduction='mean'
    )
    
    # Check CLIP parameters
    clip_params = list(feature_extractor.backbone.parameters())
    trainable_params = [p for p in clip_params if p.requires_grad]
    total_param_count = sum(p.numel() for p in clip_params)
    
    print(f"\nCLIP Parameters:")
    print(f"  Parameter Tensors: {len(clip_params)}")
    print(f"  Trainable Tensors: {len(trainable_params)}")
    print(f"  Total Parameter Count: {total_param_count:,} ({total_param_count/1e6:.1f}M)")
    print(f"  Average params per tensor: {total_param_count/len(clip_params):,.0f}")
    
    # Create fake images (batch of 4)
    batch_size = 4
    wrist_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    tactile_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Extract features
    print(f"\n[Forward Pass]")
    feature_extractor.backbone.train()  # Set CLIP to training mode
    
    features_list = []
    for i in range(batch_size):
        features = feature_extractor.extract_features(
            wrist_image=wrist_img,
            tactile_image=tactile_img
        )
        features_list.append(features)
    
    # Stack features
    h_tau = torch.stack([f['h_tau'] for f in features_list]).to(device)
    h_w = torch.stack([f['h_w'] for f in features_list]).to(device)
    
    print(f"  h_tau shape: {h_tau.shape}, requires_grad: {h_tau.requires_grad}")
    print(f"  h_w shape: {h_w.shape}, requires_grad: {h_w.requires_grad}")
    
    # Compute loss
    hsa_loss_dict = hsa_loss_fn(h_tau=h_tau, h_w=h_w)
    loss = hsa_loss_dict['hsa_total']
    
    print(f"\n[Loss Computation]")
    print(f"  HSA loss: {loss.item():.4f}")
    print(f"  Loss requires_grad: {loss.requires_grad}")
    
    # Backward pass
    print(f"\n[Backward Pass]")
    loss.backward()
    
    # Check gradients
    params_with_grad = 0
    params_without_grad = 0
    total_grad_norm = 0.0
    
    for p in trainable_params:
        if p.grad is not None:
            params_with_grad += 1
            total_grad_norm += p.grad.norm().item()
        else:
            params_without_grad += 1
    
    print(f"  Parameters with gradients: {params_with_grad}/{len(trainable_params)}")
    print(f"  Parameters without gradients: {params_without_grad}")
    print(f"  Total gradient norm: {total_grad_norm:.6f}")
    
    # Test optimizer step
    print(f"\n[Optimizer Test]")
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
    
    # Get initial param values
    initial_param = trainable_params[0].clone()
    
    optimizer.step()
    
    # Check if parameters changed
    param_changed = not torch.equal(initial_param, trainable_params[0])
    print(f"  Parameter updated: {param_changed}")
    print(f"  Parameter change norm: {(trainable_params[0] - initial_param).norm().item():.8f}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"{'='*60}")
    
    if h_tau.requires_grad and h_w.requires_grad and loss.requires_grad:
        print("✓ Features have requires_grad=True")
    else:
        print("✗ Features missing requires_grad!")
    
    grad_coverage = params_with_grad / len(trainable_params) * 100
    if grad_coverage >= 50:  # At least half should receive gradients
        print(f"✓ {params_with_grad}/{len(trainable_params)} CLIP params received gradients ({grad_coverage:.1f}%)")
    else:
        print(f"✗ Only {params_with_grad}/{len(trainable_params)} params received gradients ({grad_coverage:.1f}%)")
    
    if total_grad_norm > 0.01:  # Non-trivial gradient flow
        print(f"✓ Gradients are flowing (norm: {total_grad_norm:.6f})")
    else:
        print(f"✗ Gradients too small (norm: {total_grad_norm:.6f})")
    
    if param_changed:
        print(f"✓ Parameters updated after optimizer step")
    else:
        print(f"✗ Parameters did NOT update!")
    
    print(f"{'='*60}\n")
    
    # Pipeline is working if: features have gradients, gradients flow, and params update
    pipeline_ok = (h_tau.requires_grad and h_w.requires_grad and 
                   total_grad_norm > 0.01 and param_changed and 
                   grad_coverage >= 50)
    
    if pipeline_ok:
        print("🎉 HSA pipeline is working correctly!")
        print("   → Ready for training. HSA loss should decrease from ~2.7 to <1.0")
        return True
    else:
        print("⚠️  HSA pipeline has issues - check errors above")
        return False


if __name__ == '__main__':
    success = test_hsa_pipeline()
    exit(0 if success else 1)

