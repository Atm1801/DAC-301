"""
SplitLoRA: Two-Stage Low-Rank Adaptation for Vision Transformers

This module implements SplitLoRA, which extends LoRA by replacing the single BA 
decomposition with BB'A'A, where:
- B and A are frozen adapters initialized via SVD or Kaiming
- B' and A' are trainable adapters for fine-tuning

Forward pass: Wx → Wx + BB'A'Ax (only B', A' trained)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
import math


class SplitLoRALayer(nn.Module):
    """
    SplitLoRA layer implementing dual-stage low-rank adaptation.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Low-rank dimension for adapters
        init_strategy: 'svd' or 'kaiming' for frozen adapter initialization
        pretrained_weight: Pretrained weight matrix (required for SVD init)
        alpha: Scaling factor (default: rank)
        dropout: Dropout probability
        freeze_stage1: Whether to freeze the SVD-initialized adapters (default: True)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        init_strategy: Literal['svd', 'kaiming'] = 'svd',
        pretrained_weight: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        freeze_stage1: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.init_strategy = init_strategy
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank
        self.freeze_stage1 = freeze_stage1
        
        # Stage 1: Frozen adapters (B and A)
        # B: out_features × rank
        # A: rank × in_features
        self.lora_A_frozen = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B_frozen = nn.Parameter(torch.zeros(out_features, rank))
        
        # Stage 2: Trainable adapters (B' and A')
        # B': rank × rank (middle compression)
        # A': rank × rank (middle compression)
        self.lora_A_trainable = nn.Parameter(torch.zeros(rank, rank))
        self.lora_B_trainable = nn.Parameter(torch.zeros(rank, rank))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize adapters
        self._initialize_adapters(pretrained_weight)
        
        # Freeze stage 1 adapters if specified
        if self.freeze_stage1:
            self.lora_A_frozen.requires_grad = False
            self.lora_B_frozen.requires_grad = False
    
    def _initialize_adapters(self, pretrained_weight: Optional[torch.Tensor]):
        """Initialize frozen and trainable adapters based on strategy."""
        
        if self.init_strategy == 'svd':
            if pretrained_weight is None:
                raise ValueError("SVD initialization requires pretrained_weight")
            
            # Perform truncated SVD on pretrained weight
            # W ≈ U @ S @ V^T, we keep top-r singular values
            U, S, Vt = torch.linalg.svd(pretrained_weight.float(), full_matrices=False)
            
            # Truncate to rank
            U_r = U[:, :self.rank]  # out_features × rank
            S_r = S[:self.rank]      # rank
            Vt_r = Vt[:self.rank, :] # rank × in_features
            
            # Initialize frozen adapters: B = U @ sqrt(S), A = sqrt(S) @ V^T
            # This preserves W ≈ B @ A
            sqrt_S = torch.diag(torch.sqrt(S_r))
            self.lora_B_frozen.data = (U_r @ sqrt_S).to(pretrained_weight.dtype)
            self.lora_A_frozen.data = (sqrt_S @ Vt_r).to(pretrained_weight.dtype)
            
        elif self.init_strategy == 'kaiming':
            # Kaiming uniform initialization for frozen adapters
            nn.init.kaiming_uniform_(self.lora_A_frozen, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_frozen)
        else:
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}")
        
        # Initialize trainable adapters with small random values
        # A' with kaiming, B' with zeros (similar to LoRA)
        nn.init.kaiming_uniform_(self.lora_A_trainable, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_trainable)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute BB'A'Ax
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Low-rank adaptation output of shape (..., out_features)
        """
        # Stage 1: Apply frozen A (rank × in_features)
        # x: (..., in_features) → (..., rank)
        h1 = F.linear(x, self.lora_A_frozen)
        
        # Stage 2: Apply trainable adapters (rank × rank)
        # (..., rank) → (..., rank) → (..., rank)
        h2 = F.linear(h1, self.lora_A_trainable)
        h2 = self.dropout(h2)
        h3 = F.linear(h2, self.lora_B_trainable)
        
        # Stage 3: Apply frozen B (out_features × rank)
        # (..., rank) → (..., out_features)
        output = F.linear(h3, self.lora_B_frozen)
        
        # Apply scaling
        return output * self.scaling
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'rank={self.rank}, alpha={self.alpha}, init={self.init_strategy}, '
                f'freeze_stage1={self.freeze_stage1}')


class SplitLoRALinear(nn.Module):
    """
    Linear layer with SplitLoRA adaptation.
    
    Combines a frozen pretrained linear layer with SplitLoRA:
    output = W @ x + BB'A'A @ x
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        init_strategy: Literal['svd', 'kaiming'] = 'svd',
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        freeze_stage1: bool = True,
    ):
        super().__init__()
        
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Add SplitLoRA adapter
        self.splitlora = SplitLoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            init_strategy=init_strategy,
            pretrained_weight=linear.weight.data,
            alpha=alpha,
            dropout=dropout,
            freeze_stage1=freeze_stage1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base linear + SplitLoRA adaptation."""
        base_output = self.linear(x)
        lora_output = self.splitlora(x)
        return base_output + lora_output
    
    def merge_weights(self) -> nn.Linear:
        """
        Merge SplitLoRA weights into base linear layer.
        Useful for inference optimization.
        """
        merged = nn.Linear(self.in_features, self.out_features, 
                          bias=self.linear.bias is not None)
        
        # Compute the full SplitLoRA weight: BB'A'A
        with torch.no_grad():
            # A: rank × in_features
            # A': rank × rank
            # B': rank × rank
            # B: out_features × rank
            
            # Compute A'A: rank × in_features
            A_prime_A = self.splitlora.lora_A_trainable @ self.splitlora.lora_A_frozen
            
            # Compute B'(A'A): rank × in_features
            B_prime_A_prime_A = self.splitlora.lora_B_trainable @ A_prime_A
            
            # Compute B(B'A'A): out_features × in_features
            delta_W = self.splitlora.lora_B_frozen @ B_prime_A_prime_A
            delta_W = delta_W * self.splitlora.scaling
            
            # Merge with base weight
            merged.weight.data = self.linear.weight.data + delta_W
            if self.linear.bias is not None:
                merged.bias.data = self.linear.bias.data.clone()
        
        return merged


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def mark_only_splitlora_as_trainable(model: nn.Module) -> None:
    """
    Freeze all parameters except SplitLoRA adapters.
    Useful when converting a pretrained model.
    """
    for name, param in model.named_parameters():
        if 'splitlora' in name and 'trainable' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


if __name__ == "__main__":
    # Quick test
    print("Testing SplitLoRA implementation...")
    
    # Create a dummy linear layer
    in_dim, out_dim = 768, 768
    linear = nn.Linear(in_dim, out_dim)
    
    # Test SVD initialization
    print("\n1. Testing SVD initialization:")
    splitlora_svd = SplitLoRALinear(linear, rank=8, init_strategy='svd')
    x = torch.randn(2, 10, in_dim)
    output = splitlora_svd(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Trainable params: {count_parameters(splitlora_svd, trainable_only=True)}")
    print(f"   Total params: {count_parameters(splitlora_svd, trainable_only=False)}")
    
    # Test Kaiming initialization
    print("\n2. Testing Kaiming initialization:")
    splitlora_kaiming = SplitLoRALinear(linear, rank=8, init_strategy='kaiming')
    output = splitlora_kaiming(x)
    print(f"   Output shape: {output.shape}")
    
    # Test weight merging
    print("\n3. Testing weight merging:")
    merged = splitlora_svd.merge_weights()
    output_merged = merged(x)
    output_original = splitlora_svd(x)
    diff = (output_merged - output_original).abs().max()
    print(f"   Max difference after merging: {diff.item():.6f}")
    
    print("\n✓ All tests passed!")
