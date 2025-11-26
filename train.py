"""
Training infrastructure for SplitLoRA experiments.

Supports:
- CIFAR-10, CIFAR-100, ImageNet-100
- SplitLoRA, LoRA, and full fine-tuning baselines
- Distributed training
- Checkpointing and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import timm
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from typing import Dict, Optional, List
import time

from vit_splitlora import create_vit_with_splitlora, inject_splitlora_layer_selective
from splitlora import count_parameters


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 4):
    """Create CIFAR-10 data loaders with standard augmentation."""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


def get_cifar100_loaders(batch_size: int = 128, num_workers: int = 4):
    """Create CIFAR-100 data loaders with standard augmentation."""
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = CIFAR100(root='./data', train=False, download=True, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


def get_imagenet100_loaders(data_path: str, batch_size: int = 128, num_workers: int = 4):
    """
    Create ImageNet-100 data loaders.
    
    Expected directory structure:
    data_path/
        train/
            class_1/
            class_2/
            ...
        val/
            class_1/
            class_2/
            ...
    """
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    train_dataset = ImageFolder(f"{data_path}/train", transform=train_transform)
    val_dataset = ImageFolder(f"{data_path}/val", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


def get_data_loaders(dataset: str, batch_size: int = 128, num_workers: int = 4, 
                    imagenet_path: Optional[str] = None):
    """Get data loaders for specified dataset."""
    
    if dataset == 'cifar10':
        return get_cifar10_loaders(batch_size, num_workers)
    elif dataset == 'cifar100':
        return get_cifar100_loaders(batch_size, num_workers)
    elif dataset == 'imagenet100':
        if imagenet_path is None:
            raise ValueError("imagenet_path required for ImageNet-100")
        return get_imagenet100_loaders(imagenet_path, batch_size, num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
               optimizer: optim.Optimizer, device: torch.device, epoch: int) -> Dict:
    """Train for one epoch."""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return {
        'loss': running_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
            device: torch.device) -> Dict:
    """Validate the model."""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return {
        'loss': running_loss / len(val_loader),
        'accuracy': 100. * correct / total
    }


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int,
                   best_acc: float, save_path: str):
    """Save training checkpoint."""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, save_path)


def train_splitlora(
    model_name: str = 'vit_base_patch16_224',
    dataset: str = 'cifar10',
    num_classes: int = 10,
    rank: int = 8,
    init_strategy: str = 'svd',
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    freeze_stage1: bool = True,
    target_modules: Optional[List[str]] = None,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    save_dir: str = './checkpoints',
    imagenet_path: Optional[str] = None,
):
    """
    Main training function for SplitLoRA.
    
    Args:
        model_name: ViT architecture
        dataset: 'cifar10', 'cifar100', or 'imagenet100'
        num_classes: Number of output classes
        rank: Adapter rank
        init_strategy: 'svd' or 'kaiming'
        alpha: Scaling factor
        dropout: Dropout probability
        freeze_stage1: Whether to freeze stage 1
        target_modules: Which modules to adapt
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Warmup epochs
        save_dir: Directory to save checkpoints
        imagenet_path: Path to ImageNet-100 data
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment name
    exp_name = f"{model_name}_{dataset}_splitlora_r{rank}_{init_strategy}"
    if target_modules:
        exp_name += f"_{'_'.join(target_modules)}"
    
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(dataset, batch_size, 
                                                imagenet_path=imagenet_path)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}\n")
    
    # Create model
    print("Creating model...")
    model = create_vit_with_splitlora(
        model_name=model_name,
        pretrained=True,
        num_classes=num_classes,
        rank=rank,
        init_strategy=init_strategy,
        alpha=alpha,
        dropout=dropout,
        freeze_stage1=freeze_stage1,
        target_modules=target_modules,
    )
    model = model.to(device)
    print()
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            # Avoid division by zero when epochs <= warmup_epochs
            if epochs <= warmup_epochs:
                return 1.0
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics (convert to float for JSON serialization)
        history['train_loss'].append(float(train_metrics['loss']))
        history['train_acc'].append(float(train_metrics['accuracy']))
        history['val_loss'].append(float(val_metrics['loss']))
        history['val_acc'].append(float(val_metrics['accuracy']))
        history['lr'].append(float(scheduler.get_last_lr()[0]))
        
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            save_checkpoint(model, optimizer, epoch, best_acc,
                          save_dir / f"{exp_name}_best.pth")
            print(f"✓ Best model saved (acc: {best_acc:.2f}%)")
        
        # Save latest checkpoint
        save_checkpoint(model, optimizer, epoch, best_acc,
                      save_dir / f"{exp_name}_latest.pth")
        
        # Save history
        with open(save_dir / f"{exp_name}_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    training_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {training_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"{'='*60}\n")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SplitLoRA on Vision Transformers')
    
    # Model args
    parser.add_argument('--model', type=str, default='vit_base_patch16_224',
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'imagenet100'],
                       help='Dataset to use')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes')
    
    # SplitLoRA args
    parser.add_argument('--rank', type=int, default=8,
                       help='Adapter rank')
    parser.add_argument('--init-strategy', type=str, default='svd',
                       choices=['svd', 'kaiming'],
                       help='Initialization strategy')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Scaling factor (default: rank)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout probability')
    parser.add_argument('--no-freeze-stage1', action='store_true',
                       help='Do not freeze stage 1 adapters')
    parser.add_argument('--target-modules', type=str, nargs='+', default=None,
                       choices=['qkv', 'attn_output', 'mlp_fc1', 'mlp_fc2'],
                       help='Which modules to adapt')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=1,
                       help='Warmup epochs')
    
    # Misc args
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--imagenet-path', type=str, default=None,
                       help='Path to ImageNet-100 data')
    
    args = parser.parse_args()
    
    # Run training
    train_splitlora(
        model_name=args.model,
        dataset=args.dataset,
        num_classes=args.num_classes,
        rank=args.rank,
        init_strategy=args.init_strategy,
        alpha=args.alpha,
        dropout=args.dropout,
        freeze_stage1=not args.no_freeze_stage1,
        target_modules=args.target_modules,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        save_dir=args.save_dir,
        imagenet_path=args.imagenet_path,
    )
