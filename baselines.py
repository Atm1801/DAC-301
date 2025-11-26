"""
Baseline implementations: Standard LoRA and Full Fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import timm


class LoRALayer(nn.Module):
    """
    Standard LoRA implementation for comparison.
    Forward: Wx → Wx + BAx
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank
        
        # A: rank × in_features (initialized with Kaiming)
        # B: out_features × rank (initialized with zeros)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: BAx"""
        # x: (..., in_features) → (..., rank)
        h = F.linear(x, self.lora_A)
        h = self.dropout(h)
        # (..., rank) → (..., out_features)
        output = F.linear(h, self.lora_B)
        return output * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with standard LoRA adaptation."""
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        # Freeze base layer
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Add LoRA
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: base + LoRA"""
        return self.linear(x) + self.lora(x)


def inject_lora_into_vit(
    model: nn.Module,
    rank: int = 8,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Inject standard LoRA into ViT model.
    Same target modules as SplitLoRA for fair comparison.
    """
    if target_modules is None:
        target_modules = ['qkv', 'attn_output', 'mlp_fc1', 'mlp_fc2']
    
    stats = {mod: 0 for mod in target_modules}
    
    for name, module in model.named_modules():
        # QKV projection
        if 'qkv' in target_modules and name.endswith('attn.qkv'):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            original = module
            lora_module = LoRALinear(original, rank, alpha, dropout)
            setattr(parent, 'qkv', lora_module)
            stats['qkv'] += 1
        
        # Attention output
        if 'attn_output' in target_modules and name.endswith('attn.proj'):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            original = module
            lora_module = LoRALinear(original, rank, alpha, dropout)
            setattr(parent, 'proj', lora_module)
            stats['attn_output'] += 1
        
        # MLP fc1
        if 'mlp_fc1' in target_modules and name.endswith('mlp.fc1'):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            original = module
            lora_module = LoRALinear(original, rank, alpha, dropout)
            setattr(parent, 'fc1', lora_module)
            stats['mlp_fc1'] += 1
        
        # MLP fc2
        if 'mlp_fc2' in target_modules and name.endswith('mlp.fc2'):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            
            original = module
            lora_module = LoRALinear(original, rank, alpha, dropout)
            setattr(parent, 'fc2', lora_module)
            stats['mlp_fc2'] += 1
    
    print("LoRA injection summary:")
    for mod, count in stats.items():
        print(f"  {mod}: {count} layers")
    
    # Mark only LoRA as trainable
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}% trainable)")
    
    return model


def create_vit_with_lora(
    model_name: str = 'vit_base_patch16_224',
    pretrained: bool = True,
    num_classes: int = 10,
    rank: int = 8,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """Create ViT with standard LoRA."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model = inject_lora_into_vit(model, rank, alpha, dropout, target_modules)
    return model


def create_vit_full_finetune(
    model_name: str = 'vit_base_patch16_224',
    pretrained: bool = True,
    num_classes: int = 10,
) -> nn.Module:
    """Create ViT for full fine-tuning (all parameters trainable)."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    # All parameters trainable
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Full fine-tuning - Trainable parameters: {trainable:,}")
    
    return model


if __name__ == "__main__":
    print("Testing baseline implementations...")
    
    # Test LoRA
    print("\n1. Standard LoRA:")
    model_lora = create_vit_with_lora(
        model_name='vit_base_patch16_224',
        pretrained=True,
        num_classes=10,
        rank=8,
    )
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model_lora(x)
    print(f"   Output shape: {output.shape}")
    
    # Test full fine-tuning
    print("\n2. Full fine-tuning:")
    model_full = create_vit_full_finetune(
        model_name='vit_base_patch16_224',
        pretrained=True,
        num_classes=10,
    )
    
    with torch.no_grad():
        output = model_full(x)
    print(f"   Output shape: {output.shape}")
    
    print("\n✓ Baseline tests passed!")
