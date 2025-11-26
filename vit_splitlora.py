"""
Vision Transformer with SplitLoRA integration.

This module provides functions to inject SplitLoRA adapters into:
- Q, K, V projections in attention layers
- Attention output projection
- MLP linear layers (fc1 and fc2)
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, List
import timm
from splitlora import SplitLoRALinear, count_parameters, mark_only_splitlora_as_trainable


def inject_splitlora_into_vit(
    model: nn.Module,
    rank: int = 8,
    init_strategy: Literal['svd', 'kaiming'] = 'svd',
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    freeze_stage1: bool = True,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Inject SplitLoRA adapters into a Vision Transformer model.
    
    Args:
        model: Pretrained ViT model (e.g., from timm)
        rank: Low-rank dimension for adapters
        init_strategy: 'svd' or 'kaiming' initialization
        alpha: Scaling factor (default: rank)
        dropout: Dropout probability
        freeze_stage1: Whether to freeze SVD-initialized adapters
        target_modules: List of target modules. Options:
            - 'qkv': Q, K, V projections
            - 'attn_output': Attention output projection
            - 'mlp_fc1': First MLP layer
            - 'mlp_fc2': Second MLP layer
            If None, applies to all modules.
    
    Returns:
        Modified model with SplitLoRA adapters
    """
    if target_modules is None:
        target_modules = ['qkv', 'attn_output', 'mlp_fc1', 'mlp_fc2']
    
    # Track injection statistics
    stats = {
        'qkv': 0,
        'attn_output': 0,
        'mlp_fc1': 0,
        'mlp_fc2': 0,
    }
    
    # Inject into transformer blocks
    for name, module in model.named_modules():
        # Handle attention QKV projection
        if 'qkv' in target_modules and name.endswith('attn.qkv'):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            # Replace with SplitLoRA version
            original_qkv = module
            splitlora_qkv = SplitLoRALinear(
                original_qkv, rank, init_strategy, alpha, dropout, freeze_stage1
            )
            setattr(parent, 'qkv', splitlora_qkv)
            stats['qkv'] += 1
        
        # Handle attention output projection
        if 'attn_output' in target_modules and name.endswith('attn.proj'):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            original_proj = module
            splitlora_proj = SplitLoRALinear(
                original_proj, rank, init_strategy, alpha, dropout, freeze_stage1
            )
            setattr(parent, 'proj', splitlora_proj)
            stats['attn_output'] += 1
        
        # Handle MLP first layer (fc1)
        if 'mlp_fc1' in target_modules and name.endswith('mlp.fc1'):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            original_fc1 = module
            splitlora_fc1 = SplitLoRALinear(
                original_fc1, rank, init_strategy, alpha, dropout, freeze_stage1
            )
            setattr(parent, 'fc1', splitlora_fc1)
            stats['mlp_fc1'] += 1
        
        # Handle MLP second layer (fc2)
        if 'mlp_fc2' in target_modules and name.endswith('mlp.fc2'):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            original_fc2 = module
            splitlora_fc2 = SplitLoRALinear(
                original_fc2, rank, init_strategy, alpha, dropout, freeze_stage1
            )
            setattr(parent, 'fc2', splitlora_fc2)
            stats['mlp_fc2'] += 1
    
    # Print injection summary
    print("SplitLoRA injection summary:")
    for module_type, count in stats.items():
        if module_type in target_modules:
            print(f"  {module_type}: {count} layers modified")
    
    # Ensure only SplitLoRA parameters are trainable
    mark_only_splitlora_as_trainable(model)
    
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    print(f"\nParameter summary:")
    print(f"  Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"  Total: {total:,}")
    
    return model


def create_vit_with_splitlora(
    model_name: str = 'vit_base_patch16_224',
    pretrained: bool = True,
    num_classes: int = 10,
    rank: int = 8,
    init_strategy: Literal['svd', 'kaiming'] = 'svd',
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    freeze_stage1: bool = True,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Create a Vision Transformer model with SplitLoRA adapters.
    
    Args:
        model_name: Model architecture name (from timm)
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes
        rank: Low-rank dimension for adapters
        init_strategy: 'svd' or 'kaiming' initialization
        alpha: Scaling factor
        dropout: Dropout probability
        freeze_stage1: Whether to freeze SVD-initialized adapters
        target_modules: Which modules to adapt
    
    Returns:
        ViT model with SplitLoRA
    """
    # Load pretrained ViT
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    # Inject SplitLoRA
    model = inject_splitlora_into_vit(
        model, rank, init_strategy, alpha, dropout, freeze_stage1, target_modules
    )
    
    return model


def inject_splitlora_layer_selective(
    model: nn.Module,
    rank: int = 8,
    init_strategy: Literal['svd', 'kaiming'] = 'svd',
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    freeze_stage1: bool = True,
    attention_only: bool = False,
    mlp_only: bool = False,
    layer_indices: Optional[List[int]] = None,
) -> nn.Module:
    """
    Selective layer-wise SplitLoRA injection for ablation studies.
    
    Args:
        model: Pretrained ViT model
        rank: Low-rank dimension
        init_strategy: Initialization strategy
        alpha: Scaling factor
        dropout: Dropout probability
        freeze_stage1: Whether to freeze stage 1
        attention_only: Only inject into attention layers
        mlp_only: Only inject into MLP layers
        layer_indices: Specific transformer block indices to modify
    
    Returns:
        Modified model
    """
    if attention_only and mlp_only:
        raise ValueError("Cannot specify both attention_only and mlp_only")
    
    # Determine target modules based on flags
    if attention_only:
        target_modules = ['qkv', 'attn_output']
    elif mlp_only:
        target_modules = ['mlp_fc1', 'mlp_fc2']
    else:
        target_modules = ['qkv', 'attn_output', 'mlp_fc1', 'mlp_fc2']
    
    stats = {mod: 0 for mod in target_modules}
    
    # Iterate through blocks
    blocks = model.blocks if hasattr(model, 'blocks') else []
    
    for block_idx, block in enumerate(blocks):
        # Skip if layer_indices specified and current block not in list
        if layer_indices is not None and block_idx not in layer_indices:
            continue
        
        # Inject into attention
        if 'qkv' in target_modules and hasattr(block.attn, 'qkv'):
            original = block.attn.qkv
            block.attn.qkv = SplitLoRALinear(
                original, rank, init_strategy, alpha, dropout, freeze_stage1
            )
            stats['qkv'] += 1
        
        if 'attn_output' in target_modules and hasattr(block.attn, 'proj'):
            original = block.attn.proj
            block.attn.proj = SplitLoRALinear(
                original, rank, init_strategy, alpha, dropout, freeze_stage1
            )
            stats['attn_output'] += 1
        
        # Inject into MLP
        if 'mlp_fc1' in target_modules and hasattr(block.mlp, 'fc1'):
            original = block.mlp.fc1
            block.mlp.fc1 = SplitLoRALinear(
                original, rank, init_strategy, alpha, dropout, freeze_stage1
            )
            stats['mlp_fc1'] += 1
        
        if 'mlp_fc2' in target_modules and hasattr(block.mlp, 'fc2'):
            original = block.mlp.fc2
            block.mlp.fc2 = SplitLoRALinear(
                original, rank, init_strategy, alpha, dropout, freeze_stage1
            )
            stats['mlp_fc2'] += 1
    
    print("Layer-selective SplitLoRA injection:")
    for mod, count in stats.items():
        print(f"  {mod}: {count} layers")
    
    mark_only_splitlora_as_trainable(model)
    
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    print(f"\nParameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}% trainable)")
    
    return model


if __name__ == "__main__":
    print("Testing ViT + SplitLoRA integration...")
    
    # Test 1: Full injection
    print("\n1. Full SplitLoRA injection:")
    model = create_vit_with_splitlora(
        model_name='vit_base_patch16_224',
        pretrained=True,
        num_classes=10,
        rank=8,
        init_strategy='svd',
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    print(f"   Output shape: {output.shape}")
    
    # Test 2: Attention-only injection
    print("\n2. Attention-only injection:")
    model_attn = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
    model_attn = inject_splitlora_layer_selective(
        model_attn, rank=8, attention_only=True
    )
    
    # Test 3: MLP-only injection
    print("\n3. MLP-only injection:")
    model_mlp = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
    model_mlp = inject_splitlora_layer_selective(
        model_mlp, rank=8, mlp_only=True
    )
    
    # Test 4: Specific layers
    print("\n4. Specific layers (0, 5, 11):")
    model_selective = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
    model_selective = inject_splitlora_layer_selective(
        model_selective, rank=8, layer_indices=[0, 5, 11]
    )
    
    print("\n✓ All integration tests passed!")
