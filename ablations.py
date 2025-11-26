"""
Ablation study scripts for SplitLoRA experiments.

Ablations:
1. Layer-wise injection: attention vs MLP vs both
2. Rank variation: r ∈ {4, 8, 16, 32}
3. Initialization strategy: SVD vs Kaiming
4. Frozen vs trainable stage 1
"""

import torch
import argparse
import json
from pathlib import Path
from typing import Dict, List
import itertools

from train import train_splitlora, get_data_loaders
from vit_splitlora import create_vit_with_splitlora, inject_splitlora_layer_selective
from baselines import create_vit_with_lora, create_vit_full_finetune
from splitlora import count_parameters
import pandas as pd


def ablation_layer_wise(
    dataset: str = 'cifar10',
    num_classes: int = 10,
    rank: int = 8,
    epochs: int = 30,
    batch_size: int = 128,
    save_dir: str = './ablations/layer_wise',
):
    """
    Ablation 1: Compare attention-only, MLP-only, and both.
    """
    print("\n" + "="*70)
    print("ABLATION 1: Layer-wise Injection")
    print("="*70 + "\n")
    
    configs = [
        {'name': 'attention_only', 'target_modules': ['qkv', 'attn_output']},
        {'name': 'mlp_only', 'target_modules': ['mlp_fc1', 'mlp_fc2']},
        {'name': 'both', 'target_modules': ['qkv', 'attn_output', 'mlp_fc1', 'mlp_fc2']},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n>>> Running: {config['name']}")
        
        model, history = train_splitlora(
            model_name='vit_base_patch16_224',
            dataset=dataset,
            num_classes=num_classes,
            rank=rank,
            init_strategy='svd',
            target_modules=config['target_modules'],
            epochs=epochs,
            batch_size=batch_size,
            save_dir=save_dir,
            warmup_epochs=1,
        )
        
        # Record results
        results.append({
            'config': config['name'],
            'target_modules': ','.join(config['target_modules']),
            'best_val_acc': max(history['val_acc']),
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f"{save_dir}/layer_wise_results.csv", index=False)
    print("\n" + "="*70)
    print("Layer-wise Ablation Results:")
    print(df.to_string(index=False))
    print("="*70)


def ablation_rank_variation(
    dataset: str = 'cifar10',
    num_classes: int = 10,
    ranks: List[int] = [16, 32],
    epochs: int = 30,
    batch_size: int = 128,
    save_dir: str = './ablations/rank_variation',
):
    """
    Ablation 2: Compare different adapter ranks.
    """
    print("\n" + "="*70)
    print("ABLATION 2: Rank Variation")
    print("="*70 + "\n")
    
    results = []
    
    for rank in ranks:
        print(f"\n>>> Running: rank={rank}")
        
        model, history = train_splitlora(
            model_name='vit_base_patch16_224',
            dataset=dataset,
            num_classes=num_classes,
            rank=rank,
            init_strategy='svd',
            epochs=epochs,
            batch_size=batch_size,
            save_dir=save_dir,
            warmup_epochs=1,
        )
        
        # Count parameters
        trainable_params = count_parameters(model, trainable_only=True)
        total_params = count_parameters(model, trainable_only=False)
        
        results.append({
            'rank': rank,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'param_ratio': trainable_params / total_params,
            'best_val_acc': max(history['val_acc']),
            'final_val_acc': history['val_acc'][-1],
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f"{save_dir}/rank_variation_results.csv", index=False)
    print("\n" + "="*70)
    print("Rank Variation Results:")
    print(df.to_string(index=False))
    print("="*70)


def ablation_initialization(
    dataset: str = 'cifar10',
    num_classes: int = 10,
    rank: int = 8,
    epochs: int = 30,
    batch_size: int = 128,
    save_dir: str = './ablations/initialization',
):
    """
    Ablation 3: Compare SVD vs Kaiming initialization.
    """
    print("\n" + "="*70)
    print("ABLATION 3: Initialization Strategy")
    print("="*70 + "\n")
    
    strategies = ['kaiming']
    results = []
    
    for strategy in strategies:
        print(f"\n>>> Running: {strategy} initialization")
        
        model, history = train_splitlora(
            model_name='vit_base_patch16_224',
            dataset=dataset,
            num_classes=num_classes,
            rank=rank,
            init_strategy=strategy,
            epochs=epochs,
            batch_size=batch_size,
            save_dir=save_dir,
            warmup_epochs=1,
        )
        
        results.append({
            'init_strategy': strategy,
            'best_val_acc': max(history['val_acc']),
            'final_val_acc': history['val_acc'][-1],
            'convergence_epoch': history['val_acc'].index(max(history['val_acc'])) + 1,
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f"{save_dir}/initialization_results.csv", index=False)
    print("\n" + "="*70)
    print("Initialization Strategy Results:")
    print(df.to_string(index=False))
    print("="*70)


def ablation_frozen_vs_trainable(
    dataset: str = 'cifar10',
    num_classes: int = 10,
    rank: int = 8,
    epochs: int = 30,
    batch_size: int = 128,
    save_dir: str = './ablations/frozen_vs_trainable',
):
    """
    Ablation 4: Compare frozen stage 1 vs trainable stage 1.
    """
    print("\n" + "="*70)
    print("ABLATION 4: Frozen vs Trainable Stage 1")
    print("="*70 + "\n")
    
    configs = [
        # {'name': 'frozen_stage1', 'freeze_stage1': True},
        {'name': 'trainable_stage1', 'freeze_stage1': False},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n>>> Running: {config['name']}")
        
        model, history = train_splitlora(
            model_name='vit_base_patch16_224',
            dataset=dataset,
            num_classes=num_classes,
            rank=rank,
            init_strategy='svd',
            freeze_stage1=config['freeze_stage1'],
            epochs=epochs,
            batch_size=batch_size,
            save_dir=save_dir,
            warmup_epochs=1,
        )
        
        trainable_params = count_parameters(model, trainable_only=True)
        
        results.append({
            'config': config['name'],
            'freeze_stage1': config['freeze_stage1'],
            'trainable_params': trainable_params,
            'best_val_acc': max(history['val_acc']),
            'final_val_acc': history['val_acc'][-1],
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f"{save_dir}/frozen_vs_trainable_results.csv", index=False)
    print("\n" + "="*70)
    print("Frozen vs Trainable Results:")
    print(df.to_string(index=False))
    print("="*70)


def comprehensive_comparison(
    dataset: str = 'cifar10',
    num_classes: int = 10,
    rank: int = 8,
    epochs: int = 5,
    batch_size: int = 128,
    save_dir: str = './ablations/comprehensive',
):
    """
    Compare SplitLoRA (SVD), SplitLoRA (Kaiming), LoRA, and Full Fine-tuning.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON")
    print("="*70 + "\n")
    
    results = []
    
    # # 1. SplitLoRA with SVD
    print("\n>>> Running: SplitLoRA (SVD)")
    model_splitlora_svd, hist_svd = train_splitlora(
        model_name='vit_base_patch16_224',
        dataset=dataset,
        num_classes=num_classes,
        rank=rank,
        init_strategy='svd',
        epochs=epochs,
        batch_size=batch_size,
        save_dir=save_dir,
        warmup_epochs=1,
    )
    
    results.append({
        'method': 'SplitLoRA-SVD',
        'trainable_params': count_parameters(model_splitlora_svd, trainable_only=True),
        'best_val_acc': max(hist_svd['val_acc']),
        'final_val_acc': hist_svd['val_acc'][-1],
    })
    
    # 2. SplitLoRA with Kaiming
    print("\n>>> Running: SplitLoRA (Kaiming)")
    model_splitlora_kaiming, hist_kaiming = train_splitlora(
        model_name='vit_base_patch16_224',
        dataset=dataset,
        num_classes=num_classes,
        rank=rank,
        init_strategy='kaiming',
        epochs=epochs,
        batch_size=batch_size,
        save_dir=save_dir,
        warmup_epochs=1,
    )
    
    results.append({
        'method': 'SplitLoRA-Kaiming',
        'trainable_params': count_parameters(model_splitlora_kaiming, trainable_only=True),
        'best_val_acc': max(hist_kaiming['val_acc']),
        'final_val_acc': hist_kaiming['val_acc'][-1],
    })
    
    # 3. Standard LoRA (baseline)
    print("\n>>> Running: Standard LoRA")
    from train import train_epoch, validate
    import torch.nn as nn
    import torch.optim as optim
    
    model_lora = create_vit_with_lora(
        model_name='vit_base_patch16_224',
        pretrained=True,
        num_classes=num_classes,
        rank=rank,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_lora = model_lora.to(device)
    
    train_loader, val_loader = get_data_loaders(dataset, batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_lora.parameters(), lr=1e-3, weight_decay=1e-4)
    
    hist_lora = {'val_acc': []}
    for epoch in range(1, epochs + 1):
        train_epoch(model_lora, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model_lora, val_loader, criterion, device)
        hist_lora['val_acc'].append(val_metrics['accuracy'])
    
    results.append({
        'method': 'LoRA',
        'trainable_params': count_parameters(model_lora, trainable_only=True),
        'best_val_acc': max(hist_lora['val_acc']),
        'final_val_acc': hist_lora['val_acc'][-1],
    })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f"{save_dir}/comprehensive_comparison.csv", index=False)
    print("\n" + "="*70)
    print("Comprehensive Comparison Results:")
    print(df.to_string(index=False))
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ablation studies')
    
    parser.add_argument('--ablation', type=str, required=True,
                       choices=['layer_wise', 'rank', 'init', 'frozen', 'comprehensive', 'all'],
                       help='Which ablation to run')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'imagenet100'])
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--save-dir', type=str, default='./ablations')
    
    args = parser.parse_args()
    
    if args.ablation == 'layer_wise' or args.ablation == 'all':
        ablation_layer_wise(
            dataset=args.dataset,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=f"{args.save_dir}/layer_wise"
        )
    
    if args.ablation == 'rank' or args.ablation == 'all':
        ablation_rank_variation(
            dataset=args.dataset,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=f"{args.save_dir}/rank_variation"
        )
    
    if args.ablation == 'init' or args.ablation == 'all':
        ablation_initialization(
            dataset=args.dataset,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=f"{args.save_dir}/initialization"
        )
    
    if args.ablation == 'frozen' or args.ablation == 'all':
        ablation_frozen_vs_trainable(
            dataset=args.dataset,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=f"{args.save_dir}/frozen_vs_trainable"
        )
    
    if args.ablation == 'comprehensive':
        comprehensive_comparison(
            dataset=args.dataset,
            num_classes=args.num_classes,
            epochs=5,
            batch_size=args.batch_size,
            save_dir=f"{args.save_dir}/comprehensive"
        )
