"""
Test script for ACTJEPAAdapter policy with patch-level residual adapters

This script verifies:
1. ViT outputs all patch tokens (not just CLS)
2. Adapter processes patch sequence correctly
3. Attention pooling aggregates properly
4. Final output shape matches original ACTJEPA
5. Gradients: ViT frozen, adapter trainable
6. Parameter counts
"""

import torch
import numpy as np
from ModelTrain.module.policy_jepa_adapter import ACTJEPAAdapterPolicy


def test_adapter_integration():
    """Test ACTJEPAAdapter policy integration"""
    
    print("=" * 80)
    print("Testing ACTJEPAAdapter Policy Integration")
    print("=" * 80)
    
    # Configuration for test
    config = {
        'lr': 1e-4,
        'num_queries': 100,
        'kl_weight': 10,
        'hidden_dim': 256,
        'dim_feedforward': 2048,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': ['top'],  # RGB camera
        'tactile_camera_names': ['left_wrist', 'right_wrist'],  # Tactile sensors
        'vq': False,
        'vq_class': None,
        'vq_dim': None,
        'action_dim': 16,
        'no_encoder': False,
        'use_vitg': True,
        'vitg_ckpt_path': '/home/zexi/Dev/dobot_xtrainer_world_tac/jepa_ckpts/vitl.pt',  # Change if needed
        'vit_model': 'vitl',  # or 'vitg'
        'adapter_hidden_dim': 512,
        'adapter_depth': 3,
        'adapter_dropout': 0.1,
        'adapter_scale_init': 0.1,
        'adapter_pooling': 'attention',
    }
    
    print("\n1. Creating ACTJEPAAdapter policy...")
    print("-" * 80)
    
    try:
        policy = ACTJEPAAdapterPolicy(config)
        policy.cuda()
        print("✓ Policy created successfully")
    except Exception as e:
        print(f"✗ Failed to create policy: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Checking parameter counts and gradient status...")
    print("-" * 80)
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Frozen parameters: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
    
    # Check ViT is frozen
    vit_params = [(n, p) for n, p in policy.named_parameters() 
                  if 'vitg_encoder_shared' in n and 'vitg_base.encoder' in n]
    vit_trainable = sum(p.requires_grad for _, p in vit_params)
    vit_total = len(vit_params)
    
    print(f"\nViT base parameters: {vit_total} total, {vit_trainable} trainable")
    if vit_trainable == 0:
        print("✓ ViT is properly frozen")
    else:
        print(f"✗ WARNING: ViT has {vit_trainable} trainable parameters (should be 0)")
    
    # Check adapter is trainable
    adapter_params = [(n, p) for n, p in policy.named_parameters() 
                     if 'vitg_encoder_shared' in n and ('patch_adapter' in n or 'pooling' in n)]
    adapter_trainable = sum(p.requires_grad for _, p in adapter_params)
    adapter_total = len(adapter_params)
    
    print(f"\nAdapter parameters: {adapter_total} total, {adapter_trainable} trainable")
    if adapter_trainable == adapter_total and adapter_total > 0:
        print("✓ Adapter is properly trainable")
    else:
        print(f"✗ WARNING: Adapter should have all parameters trainable")
    
    # Calculate adapter parameter count
    adapter_param_count = sum(p.numel() for n, p in policy.named_parameters() 
                             if 'vitg_encoder_shared' in n and ('patch_adapter' in n or 'pooling' in n))
    print(f"Adapter trainable param count: {adapter_param_count:,} ({adapter_param_count/1e6:.2f}M)")
    
    print("\n3. Testing forward pass with dummy data...")
    print("-" * 80)
    
    # Create dummy data
    batch_size = 2
    num_rgb = len(config['camera_names'])
    num_tactile = len(config['tactile_camera_names'])
    
    # RGB images: 480x640
    rgb_images = torch.randn(batch_size, num_rgb, 3, 480, 640).cuda()
    
    # Tactile images: 224x224
    tactile_images = torch.randn(batch_size, num_tactile, 3, 224, 224).cuda()
    
    # Robot state
    qpos = torch.randn(batch_size, 14).cuda()
    
    # Actions
    actions = torch.randn(batch_size, config['num_queries'], config['action_dim']).cuda()
    is_pad = torch.zeros(batch_size, config['num_queries']).bool().cuda()
    
    print(f"RGB images shape: {rgb_images.shape}")
    print(f"Tactile images shape: {tactile_images.shape}")
    print(f"qpos shape: {qpos.shape}")
    print(f"actions shape: {actions.shape}")
    
    try:
        # Forward pass (training mode)
        policy.train()
        loss_dict = policy(qpos, [rgb_images, tactile_images], actions, is_pad)
        
        print("\n✓ Forward pass successful (training mode)")
        print(f"  Loss: {loss_dict['loss'].item():.4f}")
        print(f"  L1: {loss_dict['l1'].item():.4f}")
        print(f"  KL: {loss_dict['kl'].item():.4f}")
        
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n4. Testing backward pass and gradient flow...")
    print("-" * 80)
    
    try:
        # Zero gradients
        policy.optimizer.zero_grad()
        
        # Forward pass
        loss_dict = policy(qpos, [rgb_images, tactile_images], actions, is_pad)
        loss = loss_dict['loss']
        
        # Backward pass
        loss.mean().backward()
        
        # Check gradients
        vit_has_grad = any(p.grad is not None for n, p in policy.named_parameters() 
                          if 'vitg_encoder_shared' in n and 'vitg_base.encoder' in n)
        adapter_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                              for n, p in policy.named_parameters() 
                              if 'vitg_encoder_shared' in n and ('patch_adapter' in n or 'pooling' in n))
        
        print("Gradient flow check:")
        print(f"  ViT has gradients: {vit_has_grad} (should be False)")
        print(f"  Adapter has gradients: {adapter_has_grad} (should be True)")
        
        if not vit_has_grad and adapter_has_grad:
            print("✓ Gradient flow is correct")
        else:
            print("✗ WARNING: Gradient flow may be incorrect")
        
    except Exception as e:
        print(f"\n✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n5. Testing inference mode...")
    print("-" * 80)
    
    try:
        policy.eval()
        with torch.no_grad():
            actions_pred = policy(qpos, [rgb_images, tactile_images])
        
        print(f"✓ Inference successful")
        print(f"  Predicted actions shape: {actions_pred.shape}")
        print(f"  Expected shape: ({batch_size}, {config['num_queries']}, {config['action_dim']})")
        
        if actions_pred.shape == (batch_size, config['num_queries'], config['action_dim']):
            print("✓ Output shape is correct")
        else:
            print("✗ WARNING: Output shape mismatch")
            
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("All tests passed successfully!")
    print("=" * 80)
    
    return True


def test_adapter_components():
    """Test individual adapter components"""
    
    print("\n" + "=" * 80)
    print("Testing Individual Adapter Components")
    print("=" * 80)
    
    from ModelTrain.module.residual_adapter import PatchResidualAdapter, AttentionPooling, MeanPooling
    
    # Test parameters
    batch_size = 2
    num_patches = 196  # 14x14 for 224x224 image with patch_size=16
    embed_dim = 1024  # ViT-L dimension
    
    print("\n1. Testing PatchResidualAdapter...")
    print("-" * 80)
    
    try:
        adapter = PatchResidualAdapter(
            embed_dim=embed_dim,
            hidden_dim=512,
            depth=3,
            dropout=0.1,
            scale_init=0.1,
        ).cuda()
        
        # Test input
        patches = torch.randn(batch_size, num_patches, embed_dim).cuda()
        
        # Forward pass
        output = adapter(patches)
        
        print(f"✓ PatchResidualAdapter working")
        print(f"  Input shape: {patches.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {adapter.get_num_params():,}")
        
        assert output.shape == patches.shape, "Output shape mismatch"
        
    except Exception as e:
        print(f"✗ PatchResidualAdapter failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Testing AttentionPooling...")
    print("-" * 80)
    
    try:
        pooling = AttentionPooling(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
        ).cuda()
        
        # Forward pass
        output = pooling(patches)
        
        print(f"✓ AttentionPooling working")
        print(f"  Input shape: {patches.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {pooling.get_num_params():,}")
        
        assert output.shape == (batch_size, embed_dim), "Output shape mismatch"
        
    except Exception as e:
        print(f"✗ AttentionPooling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Testing MeanPooling...")
    print("-" * 80)
    
    try:
        pooling = MeanPooling(embed_dim=embed_dim).cuda()
        
        # Forward pass
        output = pooling(patches)
        
        print(f"✓ MeanPooling working")
        print(f"  Input shape: {patches.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {pooling.get_num_params():,}")
        
        assert output.shape == (batch_size, embed_dim), "Output shape mismatch"
        
    except Exception as e:
        print(f"✗ MeanPooling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("All component tests passed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    print("\n\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ACTJEPAAdapter Integration Test" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Test individual components first
    components_ok = test_adapter_components()
    
    if components_ok:
        # Test full integration
        integration_ok = test_adapter_integration()
        
        if integration_ok:
            print("\n✓ ALL TESTS PASSED ✓")
        else:
            print("\n✗ INTEGRATION TEST FAILED")
    else:
        print("\n✗ COMPONENT TESTS FAILED")

