# python YijingCode/train.py

import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
from datetime import datetime

from YijingCode.CBmodel import Text_CB_AE, Image_CB_AE, DualStreamLoss
from YijingCode.dataset import DualStreamDataset, get_dataloader

def train_epoch(models, loss_fn, dataloader, optimizer, device, epoch, print_freq=10):
    """Train for one epoch."""
    text_model, image_model = models
    text_model.train()
    image_model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_concept = 0.0
    total_consistency = 0.0
    total_adversarial = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        text_emb = batch['text_emb'].to(device)
        image_lat = batch['image_lat'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        txt_out = text_model(text_emb)
        img_out = image_model(image_lat)
        
        # Compute loss
        losses = loss_fn(
            img_out=img_out,
            txt_out=txt_out,
            target_img=image_lat,
            target_txt=text_emb,
            pseudo_labels=labels
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += losses['total'].item()
        total_recon += losses['recon']
        total_concept += losses['concept']
        total_consistency += losses['consistency']
        total_adversarial += losses['adversarial']
        num_batches += 1
        
        # Print progress
        if (batch_idx + 1) % print_freq == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {losses['total'].item():.4f} | "
                  f"Recon: {losses['recon']:.4f} | "
                  f"Concept: {losses['concept']:.4f} | "
                  f"Consistency: {losses['consistency']:.4f}")
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_concept = total_concept / num_batches
    avg_consistency = total_consistency / num_batches
    avg_adversarial = total_adversarial / num_batches
    
    return {
        'loss': avg_loss,
        'recon': avg_recon,
        'concept': avg_concept,
        'consistency': avg_consistency,
        'adversarial': avg_adversarial
    }

def validate(models, loss_fn, dataloader, device):
    """Validate the models."""
    text_model, image_model = models
    text_model.eval()
    image_model.eval()
    
    total_loss = 0.0
    total_recon = 0.0
    total_concept = 0.0
    total_consistency = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            text_emb = batch['text_emb'].to(device)
            image_lat = batch['image_lat'].to(device)
            labels = batch['labels'].to(device)
            
            txt_out = text_model(text_emb)
            img_out = image_model(image_lat)
            
            losses = loss_fn(
                img_out=img_out,
                txt_out=txt_out,
                target_img=image_lat,
                target_txt=text_emb,
                pseudo_labels=labels
            )
            
            total_loss += losses['total'].item()
            total_recon += losses['recon']
            total_concept += losses['concept']
            total_consistency += losses['consistency']
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'concept': total_concept / num_batches,
        'consistency': total_consistency / num_batches
    }

def save_checkpoint(models, optimizer, epoch, metrics, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    text_model, image_model = models
    
    checkpoint = {
        'epoch': epoch,
        'text_model_state_dict': text_model.state_dict(),
        'image_model_state_dict': image_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"  Saved best model to {best_path}")
    
    print(f"  Saved checkpoint to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Dual-Stream Concept Bottleneck Models')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='YijingCode/TrainingData',
                        help='Directory containing training data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--seq_len', type=int, default=333,
                        help='Text sequence length (SD3.5 Medium: 333)')
    parser.add_argument('--num_patches', type=int, default=2304,
                        help='Number of image patches (SD3.5 Medium: 2304)')
    parser.add_argument('--embed_dim', type=int, default=1536,
                        help='Embedding dimension (SD3.5 Medium: 1536)')
    parser.add_argument('--concept_dim', type=int, default=2,
                        help='Number of concepts')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='Hidden dimension for text MLP (reduce to shrink params)')
    parser.add_argument('--use_adversarial', action='store_true',
                        help='Use adversarial training (soft bottleneck)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    
    # Loss weights
    parser.add_argument('--lambda_recon', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--lambda_concept', type=float, default=1.0,
                        help='Weight for concept alignment loss')
    parser.add_argument('--lambda_consist', type=float, default=5.0,
                        help='Weight for cross-modal consistency loss')
    parser.add_argument('--lambda_adv', type=float, default=0.5,
                        help='Weight for adversarial loss')
    
    # Validation / early stopping
    parser.add_argument('--val_split', type=float, default=0.05,
                        help='Fraction of data to use as validation set')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience (epochs with no val loss improvement)')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                        help='Minimum improvement in val loss to reset early stopping')
    
    # Checkpointing and logging
    parser.add_argument('--checkpoint_dir', type=str, default='YijingCode/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print training stats every N batches')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    config_path = checkpoint_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training config to {config_path}")
    
    # Initialize models
    print("\nInitializing models...")
    text_model = Text_CB_AE(
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        concept_dim=args.concept_dim,
        hidden_dim=args.hidden_dim,
        use_adversarial=args.use_adversarial
    ).to(device)
    
    image_model = Image_CB_AE(
        num_patches=args.num_patches,
        patch_dim=args.embed_dim,
        concept_dim=args.concept_dim,
        use_adversarial=args.use_adversarial
    ).to(device)
    
    # Count parameters
    text_params = sum(p.numel() for p in text_model.parameters())
    image_params = sum(p.numel() for p in image_model.parameters())
    print(f"Text model parameters: {text_params:,}")
    print(f"Image model parameters: {image_params:,}")
    print(f"Total parameters: {text_params + image_params:,}")
    
    # Initialize loss function
    loss_fn = DualStreamLoss(
        lambda_recon=args.lambda_recon,
        lambda_concept=args.lambda_concept,
        lambda_consist=args.lambda_consist,
        lambda_adv=args.lambda_adv
    )
    
    # Initialize optimizer
    all_params = list(text_model.parameters()) + list(image_model.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create dataset and train/val split
    print(f"\nLoading dataset from {args.data_dir}...")
    metadata_path = Path(args.data_dir) / "metadata.csv"
    full_dataset = DualStreamDataset(metadata_path=metadata_path)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    
    # Deterministic split
    indices = torch.randperm(n_total)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Total samples: {n_total}  |  Train: {n_train}  |  Val: {n_val}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch:   {len(val_loader)}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        text_model.load_state_dict(checkpoint['text_model_state_dict'])
        image_model.load_state_dict(checkpoint['image_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Training loop with validation and early stopping
    print(f"\n{'='*60}")
    print(f"Starting training ({'Soft Bottleneck' if args.use_adversarial else 'Hard Bottleneck'})")
    print(f"{'='*60}")
    
    epochs_without_improve = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            models=(text_model, image_model),
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq
        )
        
        # Validate
        val_metrics = validate(
            models=(text_model, image_model),
            loss_fn=loss_fn,
            dataloader=val_loader,
            device=device
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"    Train Recon: {train_metrics['recon']:.4f}")
        print(f"    Train Concept: {train_metrics['concept']:.4f}")
        print(f"    Train Consistency: {train_metrics['consistency']:.4f}")
        if args.use_adversarial:
            print(f"    Train Adversarial: {train_metrics['adversarial']:.4f}")
        
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"    Val Recon: {val_metrics['recon']:.4f}")
        print(f"    Val Concept: {val_metrics['concept']:.4f}")
        print(f"    Val Consistency: {val_metrics['consistency']:.4f}")
        
        # Early stopping check (use validation loss)
        if val_metrics['loss'] < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_metrics['loss']
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            print(f"  No improvement in val loss for {epochs_without_improve} epoch(s).")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            is_best = val_metrics['loss'] <= best_val_loss
            save_checkpoint(
                models=(text_model, image_model),
                optimizer=optimizer,
                epoch=epoch,
                metrics={'train': train_metrics, 'val': val_metrics, 'best_val_loss': best_val_loss},
                checkpoint_dir=checkpoint_dir,
                is_best=is_best
            )
        
        # Apply early stopping
        if epochs_without_improve >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {epochs_without_improve} epochs without improvement.")
            break
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

