"""
Unit tests for SplitLoRA implementation.

Tests:
1. SplitLoRA layer initialization (SVD and Kaiming)
2. Forward pass correctness
3. Parameter counting
4. Gradient flow
5. Weight merging
6. ViT integration
"""

import torch
import torch.nn as nn
import pytest
from splitlora import (
    SplitLoRALayer, SplitLoRALinear, count_parameters, 
    mark_only_splitlora_as_trainable
)
from vit_splitlora import create_vit_with_splitlora, inject_splitlora_layer_selective
from baselines import LoRALinear, create_vit_with_lora


class TestSplitLoRALayer:
    """Test SplitLoRALayer class."""
    
    def test_svd_initialization(self):
        """Test SVD-based initialization."""
        in_dim, out_dim, rank = 256, 256, 8
        
        # Create a random pretrained weight
        pretrained = torch.randn(out_dim, in_dim)
        
        # Create layer
        layer = SplitLoRALayer(
            in_features=in_dim,
            out_features=out_dim,
            rank=rank,
            init_strategy='svd',
            pretrained_weight=pretrained,
        )
        
        # Check shapes
        assert layer.lora_A_frozen.shape == (rank, in_dim)
        assert layer.lora_B_frozen.shape == (out_dim, rank)
        assert layer.lora_A_trainable.shape == (rank, rank)
        assert layer.lora_B_trainable.shape == (rank, rank)
        
        # Check frozen parameters
        assert not layer.lora_A_frozen.requires_grad
        assert not layer.lora_B_frozen.requires_grad
        assert layer.lora_A_trainable.requires_grad
        assert layer.lora_B_trainable.requires_grad
        
        # Check that BA approximates truncated SVD
        # Reconstruct: B @ A should be close to truncated SVD of pretrained
        with torch.no_grad():
            reconstructed = layer.lora_B_frozen @ layer.lora_A_frozen
            
            # Compute truncated SVD reference
            U, S, Vt = torch.linalg.svd(pretrained, full_matrices=False)
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vt_r = Vt[:rank, :]
            reference = U_r @ torch.diag(S_r) @ Vt_r
            
            # Should be close
            diff = (reconstructed - reference).abs().max()
            assert diff < 1e-5, f"SVD reconstruction error too large: {diff}"
        
        print("✓ SVD initialization test passed")
    
    def test_kaiming_initialization(self):
        """Test Kaiming initialization."""
        in_dim, out_dim, rank = 256, 256, 8
        
        layer = SplitLoRALayer(
            in_features=in_dim,
            out_features=out_dim,
            rank=rank,
            init_strategy='kaiming',
        )
        
        # Check shapes
        assert layer.lora_A_frozen.shape == (rank, in_dim)
        assert layer.lora_B_frozen.shape == (out_dim, rank)
        
        # B should be zeros
        assert torch.allclose(layer.lora_B_frozen, torch.zeros_like(layer.lora_B_frozen))
        
        # A should be non-zero
        assert not torch.allclose(layer.lora_A_frozen, torch.zeros_like(layer.lora_A_frozen))
        
        print("✓ Kaiming initialization test passed")
    
    def test_forward_pass(self):
        """Test forward pass computation."""
        batch_size, seq_len = 4, 10
        in_dim, out_dim, rank = 256, 256, 8
        
        pretrained = torch.randn(out_dim, in_dim)
        
        layer = SplitLoRALayer(
            in_features=in_dim,
            out_features=out_dim,
            rank=rank,
            init_strategy='svd',
            pretrained_weight=pretrained,
        )
        
        # Create input
        x = torch.randn(batch_size, seq_len, in_dim)
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, out_dim)
        
        # Check that output is not zero (since trainable params are initialized)
        # Initially B' is zeros, so output should be zero
        assert torch.allclose(output, torch.zeros_like(output)), \
            "Output should be zero initially (B' initialized to zeros)"
        
        # Set B' to non-zero and test again
        with torch.no_grad():
            layer.lora_B_trainable.fill_(0.1)
        
        output = layer(x)
        assert not torch.allclose(output, torch.zeros_like(output)), \
            "Output should be non-zero after setting B'"
        
        print("✓ Forward pass test passed")
    
    def test_parameter_count(self):
        """Test parameter counting."""
        in_dim, out_dim, rank = 256, 256, 8
        
        pretrained = torch.randn(out_dim, in_dim)
        
        layer = SplitLoRALayer(
            in_features=in_dim,
            out_features=out_dim,
            rank=rank,
            init_strategy='svd',
            pretrained_weight=pretrained,
        )
        
        # Count trainable parameters: A' (rank × rank) + B' (rank × rank)
        expected_trainable = 2 * rank * rank
        actual_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        
        assert actual_trainable == expected_trainable, \
            f"Expected {expected_trainable} trainable params, got {actual_trainable}"
        
        # Total parameters
        expected_total = 2 * rank * in_dim + 2 * rank * out_dim + 2 * rank * rank
        actual_total = sum(p.numel() for p in layer.parameters())
        
        # Note: frozen params are rank×in + out×rank, trainable are rank×rank each
        expected_total = rank * in_dim + out_dim * rank + 2 * rank * rank
        
        assert actual_total == expected_total, \
            f"Expected {expected_total} total params, got {actual_total}"
        
        print("✓ Parameter count test passed")
    
    def test_gradient_flow(self):
        """Test that gradients flow only to trainable parameters."""
        in_dim, out_dim, rank = 256, 256, 8
        
        pretrained = torch.randn(out_dim, in_dim)
        
        layer = SplitLoRALayer(
            in_features=in_dim,
            out_features=out_dim,
            rank=rank,
            init_strategy='svd',
            pretrained_weight=pretrained,
        )
        
        # Set trainable params to non-zero
        with torch.no_grad():
            layer.lora_A_trainable.fill_(0.1)
            layer.lora_B_trainable.fill_(0.1)
        
        # Forward pass
        x = torch.randn(2, 10, in_dim)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert layer.lora_A_trainable.grad is not None, "A' should have gradients"
        assert layer.lora_B_trainable.grad is not None, "B' should have gradients"
        assert layer.lora_A_frozen.grad is None, "A should not have gradients"
        assert layer.lora_B_frozen.grad is None, "B should not have gradients"
        
        print("✓ Gradient flow test passed")


class TestSplitLoRALinear:
    """Test SplitLoRALinear class."""
    
    def test_linear_integration(self):
        """Test integration with nn.Linear."""
        in_dim, out_dim, rank = 256, 256, 8
        
        # Create linear layer
        linear = nn.Linear(in_dim, out_dim)
        
        # Wrap with SplitLoRA
        splitlora_linear = SplitLoRALinear(
            linear, rank=rank, init_strategy='svd'
        )
        
        # Check that base linear is frozen
        for param in splitlora_linear.linear.parameters():
            assert not param.requires_grad
        
        # Forward pass
        x = torch.randn(2, 10, in_dim)
        output = splitlora_linear(x)
        
        assert output.shape == (2, 10, out_dim)
        
        print("✓ Linear integration test passed")
    
    def test_weight_merging(self):
        """Test weight merging functionality."""
        in_dim, out_dim, rank = 256, 256, 8
        
        linear = nn.Linear(in_dim, out_dim)
        splitlora_linear = SplitLoRALinear(linear, rank=rank, init_strategy='svd')
        
        # Set trainable params to non-zero
        with torch.no_grad():
            splitlora_linear.splitlora.lora_A_trainable.fill_(0.1)
            splitlora_linear.splitlora.lora_B_trainable.fill_(0.1)
        
        # Forward pass before merging
        x = torch.randn(2, 10, in_dim)
        output_before = splitlora_linear(x)
        
        # Merge weights
        merged_linear = splitlora_linear.merge_weights()
        output_after = merged_linear(x)
        
        # Outputs should be very close
        diff = (output_before - output_after).abs().max()
        assert diff < 1e-4, f"Merge error too large: {diff}"
        
        print("✓ Weight merging test passed")


class TestViTIntegration:
    """Test ViT integration."""
    
    def test_vit_creation(self):
        """Test creating ViT with SplitLoRA."""
        model = create_vit_with_splitlora(
            model_name='vit_base_patch16_224',
            pretrained=False,  # Faster for testing
            num_classes=10,
            rank=8,
            init_strategy='svd',
        )
        
        # Check that model has SplitLoRA layers
        has_splitlora = False
        for name, module in model.named_modules():
            if 'splitlora' in name:
                has_splitlora = True
                break
        
        assert has_splitlora, "Model should contain SplitLoRA layers"
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)
        
        print("✓ ViT creation test passed")
    
    def test_layer_selective_injection(self):
        """Test selective layer injection."""
        import timm
        
        # Attention only
        model_attn = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
        model_attn = inject_splitlora_layer_selective(
            model_attn, rank=8, attention_only=True
        )
        
        trainable_attn = count_parameters(model_attn, trainable_only=True)
        
        # MLP only
        model_mlp = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
        model_mlp = inject_splitlora_layer_selective(
            model_mlp, rank=8, mlp_only=True
        )
        
        trainable_mlp = count_parameters(model_mlp, trainable_only=True)
        
        # Both configurations should have trainable parameters
        assert trainable_attn > 0, "Attention-only should have trainable parameters"
        assert trainable_mlp > 0, "MLP-only should have trainable parameters"
        
        # For ViT-B/16 with same rank, MLP and attention have same structure
        # so they can have same param count - just check they're reasonable
        print(f"   Attention-only params: {trainable_attn:,}")
        print(f"   MLP-only params: {trainable_mlp:,}")
        
        print("✓ Layer-selective injection test passed")


class TestBaselines:
    """Test baseline implementations."""
    
    def test_lora_baseline(self):
        """Test standard LoRA implementation."""
        model = create_vit_with_lora(
            model_name='vit_base_patch16_224',
            pretrained=False,
            num_classes=10,
            rank=8,
        )
        
        # Check that model has LoRA layers
        has_lora = False
        for name, module in model.named_modules():
            if 'lora' in name and 'split' not in name:
                has_lora = True
                break
        
        assert has_lora, "Model should contain LoRA layers"
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)
        
        print("✓ LoRA baseline test passed")
    
    def test_parameter_efficiency(self):
        """Compare parameter counts: SplitLoRA vs LoRA."""
        rank = 8
        
        # SplitLoRA
        model_split = create_vit_with_splitlora(
            model_name='vit_base_patch16_224',
            pretrained=False,
            num_classes=10,
            rank=rank,
        )
        params_split = count_parameters(model_split, trainable_only=True)
        
        # LoRA
        model_lora = create_vit_with_lora(
            model_name='vit_base_patch16_224',
            pretrained=False,
            num_classes=10,
            rank=rank,
        )
        params_lora = count_parameters(model_lora, trainable_only=True)
        
        print(f"   SplitLoRA trainable params: {params_split:,}")
        print(f"   LoRA trainable params: {params_lora:,}")
        
        # SplitLoRA should have fewer trainable parameters (2r^2 vs 2rd per layer)
        # since stage 1 is frozen
        assert params_split < params_lora, \
            "SplitLoRA should have fewer trainable parameters than LoRA"
        
        print("✓ Parameter efficiency test passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*70)
    print("Running SplitLoRA Unit Tests")
    print("="*70 + "\n")
    
    # Test SplitLoRALayer
    print("Testing SplitLoRALayer...")
    layer_tests = TestSplitLoRALayer()
    layer_tests.test_svd_initialization()
    layer_tests.test_kaiming_initialization()
    layer_tests.test_forward_pass()
    layer_tests.test_parameter_count()
    layer_tests.test_gradient_flow()
    
    # Test SplitLoRALinear
    print("\nTesting SplitLoRALinear...")
    linear_tests = TestSplitLoRALinear()
    linear_tests.test_linear_integration()
    linear_tests.test_weight_merging()
    
    # Test ViT integration
    print("\nTesting ViT Integration...")
    vit_tests = TestViTIntegration()
    vit_tests.test_vit_creation()
    vit_tests.test_layer_selective_injection()
    
    # Test baselines
    print("\nTesting Baselines...")
    baseline_tests = TestBaselines()
    baseline_tests.test_lora_baseline()
    baseline_tests.test_parameter_efficiency()
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
