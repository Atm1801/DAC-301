"""
Evaluation and visualization tools for SplitLoRA experiments.

- Plot training curves
- Parameter-accuracy trade-off analysis
- Runtime cost analysis
- Confusion matrices
- Layer-wise sensitivity analysis
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
from sklearn.metrics import confusion_matrix, classification_report

from vit_splitlora import create_vit_with_splitlora
from baselines import create_vit_with_lora, create_vit_full_finetune
from train import get_data_loaders, validate
from splitlora import count_parameters


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_training_curves(history_path: str, save_path: Optional[str] = None):
    """Plot training and validation curves."""
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_rank_comparison(results_csv: str, save_path: Optional[str] = None):
    """Plot parameter-accuracy trade-off for different ranks."""
    
    df = pd.read_csv(results_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy vs Rank
    axes[0].plot(df['rank'], df['best_val_acc'], marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Rank (r)')
    axes[0].set_ylabel('Best Validation Accuracy (%)')
    axes[0].set_title('Accuracy vs Rank')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(df['rank'])
    
    # Accuracy vs Parameters
    axes[1].plot(df['trainable_params'], df['best_val_acc'], marker='o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Trainable Parameters')
    axes[1].set_ylabel('Best Validation Accuracy (%)')
    axes[1].set_title('Accuracy vs Trainable Parameters')
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_method_comparison(results_csv: str, save_path: Optional[str] = None):
    """Compare different methods (SplitLoRA, LoRA, Full FT)."""
    
    df = pd.read_csv(results_csv)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['best_val_acc'], width, label='Best Val Acc', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['final_val_acc'], width, label='Final Val Acc', alpha=0.8)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['method'], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add parameter counts as text
    for i, row in df.iterrows():
        params_text = f"{row['trainable_params']:,.0f}" if pd.notna(row['trainable_params']) else "N/A"
        ax.text(i, 5, params_text, ha='center', va='bottom', fontsize=9, rotation=90)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def measure_inference_time(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    num_iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Measure model inference time."""
    
    device = next(model.parameters()).device
    model.eval()
    
    # Create dummy input
    x = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    return {
        'mean_ms': times.mean(),
        'std_ms': times.std(),
        'median_ms': np.median(times),
        'min_ms': times.min(),
        'max_ms': times.max(),
    }


def runtime_analysis(
    model_name: str = 'vit_base_patch16_224',
    num_classes: int = 10,
    ranks: List[int] = [4, 8, 16, 32],
    save_path: Optional[str] = None,
):
    """Compare inference times for different configurations."""
    
    print("Running runtime analysis...")
    results = []
    
    # Base model (frozen)
    print("  Measuring: Base ViT (frozen)")
    import timm
    model_base = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model_base = model_base.cuda() if torch.cuda.is_available() else model_base
    for param in model_base.parameters():
        param.requires_grad = False
    
    timing = measure_inference_time(model_base)
    results.append({
        'method': 'Base ViT',
        'rank': 0,
        'mean_ms': timing['mean_ms'],
        'std_ms': timing['std_ms'],
    })
    
    # SplitLoRA with different ranks
    for rank in ranks:
        print(f"  Measuring: SplitLoRA (rank={rank})")
        model = create_vit_with_splitlora(
            model_name=model_name,
            pretrained=True,
            num_classes=num_classes,
            rank=rank,
            init_strategy='svd',
        )
        model = model.cuda() if torch.cuda.is_available() else model
        
        timing = measure_inference_time(model)
        results.append({
            'method': f'SplitLoRA-{rank}',
            'rank': rank,
            'mean_ms': timing['mean_ms'],
            'std_ms': timing['std_ms'],
        })
    
    # Standard LoRA
    for rank in ranks:
        print(f"  Measuring: LoRA (rank={rank})")
        model = create_vit_with_lora(
            model_name=model_name,
            pretrained=True,
            num_classes=num_classes,
            rank=rank,
        )
        model = model.cuda() if torch.cuda.is_available() else model
        
        timing = measure_inference_time(model)
        results.append({
            'method': f'LoRA-{rank}',
            'rank': rank,
            'mean_ms': timing['mean_ms'],
            'std_ms': timing['std_ms'],
        })
    
    df = pd.DataFrame(results)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    splitlora_data = df[df['method'].str.contains('SplitLoRA')]
    lora_data = df[df['method'].str.contains('LoRA') & ~df['method'].str.contains('Split')]
    base_data = df[df['method'] == 'Base ViT']
    
    if not splitlora_data.empty:
        ax.errorbar(splitlora_data['rank'], splitlora_data['mean_ms'], 
                   yerr=splitlora_data['std_ms'], marker='o', label='SplitLoRA', linewidth=2, markersize=8)
    
    if not lora_data.empty:
        ax.errorbar(lora_data['rank'], lora_data['mean_ms'],
                   yerr=lora_data['std_ms'], marker='s', label='LoRA', linewidth=2, markersize=8)
    
    if not base_data.empty:
        ax.axhline(y=base_data['mean_ms'].values[0], color='gray', linestyle='--', 
                  label='Base ViT', linewidth=2)
    
    ax.set_xlabel('Rank (r)')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Runtime Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        
        # Save data
        df.to_csv(save_path.replace('.png', '.csv'), index=False)
    else:
        plt.show()
    
    plt.close()
    
    return df


def plot_confusion_matrix(model: nn.Module, val_loader, class_names: List[str],
                         save_path: Optional[str] = None):
    """Plot confusion matrix for model predictions."""
    
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualization and analysis tools')
    parser.add_argument('--task', type=str, required=True,
                       choices=['training_curves', 'rank_comparison', 'method_comparison', 
                               'runtime_analysis'],
                       help='Which analysis to run')
    parser.add_argument('--input', type=str, help='Input file path (CSV or JSON)')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--num-classes', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.task == 'training_curves':
        plot_training_curves(args.input, args.output)
    elif args.task == 'rank_comparison':
        plot_rank_comparison(args.input, args.output)
    elif args.task == 'method_comparison':
        plot_method_comparison(args.input, args.output)
    elif args.task == 'runtime_analysis':
        runtime_analysis(args.model, args.num_classes, save_path=args.output)
