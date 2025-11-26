"""
Quick verification script - tests core functionality without loading large models.
"""

import torch
import sys

def test_basic_functionality():
    """Test basic SplitLoRA functionality."""
    
    print("="*70)
    print("SplitLoRA Quick Verification")
    print("="*70)
    
    try:
        # Test 1: Import modules
        print("\n1. Testing imports...")
        from splitlora import SplitLoRALayer, SplitLoRALinear, count_parameters
        print("   ✓ Core modules imported")
        
        # Test 2: SVD initialization
        print("\n2. Testing SVD initialization...")
        pretrained = torch.randn(256, 256)
        layer_svd = SplitLoRALayer(
            in_features=256,
            out_features=256,
            rank=8,
            init_strategy='svd',
            pretrained_weight=pretrained
        )
        assert layer_svd.lora_A_frozen.shape == (8, 256)
        assert layer_svd.lora_B_frozen.shape == (256, 8)
        assert not layer_svd.lora_A_frozen.requires_grad
        assert layer_svd.lora_A_trainable.requires_grad
        print("   ✓ SVD initialization correct")
        
        # Test 3: Kaiming initialization
        print("\n3. Testing Kaiming initialization...")
        layer_kaiming = SplitLoRALayer(
            in_features=256,
            out_features=256,
            rank=8,
            init_strategy='kaiming'
        )
        assert torch.allclose(layer_kaiming.lora_B_frozen, torch.zeros_like(layer_kaiming.lora_B_frozen))
        print("   ✓ Kaiming initialization correct")
        
        # Test 4: Forward pass
        print("\n4. Testing forward pass...")
        x = torch.randn(2, 10, 256)
        output = layer_svd(x)
        assert output.shape == (2, 10, 256)
        print("   ✓ Forward pass works")
        
        # Test 5: Parameter counting
        print("\n5. Testing parameter counting...")
        trainable = sum(p.numel() for p in layer_svd.parameters() if p.requires_grad)
        expected_trainable = 2 * 8 * 8  # 2 * rank * rank
        assert trainable == expected_trainable, f"Expected {expected_trainable}, got {trainable}"
        print(f"   ✓ Parameter count correct: {trainable:,} trainable")
        
        # Test 6: Gradient flow
        print("\n6. Testing gradient flow...")
        layer_test = SplitLoRALayer(256, 256, 8, 'svd', pretrained)
        with torch.no_grad():
            layer_test.lora_A_trainable.fill_(0.1)
            layer_test.lora_B_trainable.fill_(0.1)
        
        x = torch.randn(2, 10, 256, requires_grad=True)
        output = layer_test(x)
        loss = output.sum()
        loss.backward()
        
        assert layer_test.lora_A_trainable.grad is not None
        assert layer_test.lora_B_trainable.grad is not None
        assert layer_test.lora_A_frozen.grad is None
        assert layer_test.lora_B_frozen.grad is None
        print("   ✓ Gradients flow only to trainable parameters")
        
        # Test 7: SplitLoRALinear
        print("\n7. Testing SplitLoRALinear...")
        import torch.nn as nn
        linear = nn.Linear(256, 256)
        splitlora_linear = SplitLoRALinear(linear, rank=8, init_strategy='svd')
        
        # Check base is frozen
        assert not next(splitlora_linear.linear.parameters()).requires_grad
        
        # Forward pass
        x = torch.randn(2, 10, 256)
        output = splitlora_linear(x)
        assert output.shape == (2, 10, 256)
        print("   ✓ SplitLoRALinear works correctly")
        
        # Test 8: Weight merging
        print("\n8. Testing weight merging...")
        with torch.no_grad():
            splitlora_linear.splitlora.lora_A_trainable.fill_(0.1)
            splitlora_linear.splitlora.lora_B_trainable.fill_(0.1)
        
        output_before = splitlora_linear(x)
        merged = splitlora_linear.merge_weights()
        output_after = merged(x)
        
        diff = (output_before - output_after).abs().max().item()
        assert diff < 1e-4, f"Merge error too large: {diff}"
        print(f"   ✓ Weight merging accurate (error: {diff:.6f})")
        
        print("\n" + "="*70)
        print("✅ All core tests passed!")
        print("="*70)
        print("\nSplitLoRA is ready to use!")
        print("\nNext steps:")
        print("  1. Run: python examples.py (for usage examples)")
        print("  2. Run: python train.py --epochs 5 (quick training test)")
        print("  3. Run: bash quick_start.sh (full validation)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
