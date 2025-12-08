# python YijingCode/train.py

import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import argparse
import json
from datetime import datetime
import tempfile

from YijingCode.CBmodel import ConceptBottleneckTransformer, CombinedLoss
from YijingCode.dataset import DualStreamDataset, get_dataloader

# Optional Weights & Biases support
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

# Optional Hugging Face Hub support
try:
    from huggingface_hub import HfApi, HfFolder
except ImportError:
    HfApi = None
    HfFolder = None

def _build_intervention_targets(labels):
    """Flip one random concept per sample to simulate training-time intervention."""
    bsz, num_c = labels.shape
    idx = torch.randint(0, num_c, (bsz,), device=labels.device)
    flipped = labels.clone()
    flipped[torch.arange(bsz), idx] = 1.0 - flipped[torch.arange(bsz), idx]
    return flipped


def train_epoch(model, loss_fn, dataloader, optimizer, device, epoch, print_freq=10):
    """Train for one epoch."""
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    total_loss = 0.0
    total_recon = 0.0
    total_recon_text = 0.0
    total_recon_image = 0.0
    total_concept = 0.0
    total_intervene = 0.0
    total_adversarial = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        text_emb = batch["text_emb"].to(device)
        image_lat = batch["image_lat"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass (original concepts)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(text_emb, image_lat)

        # Intervention path: force a random concept flip, decode, re-encode to enforce steerability
        intervened_labels = _build_intervention_targets(labels)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            recon_text_int, recon_image_int, _, _ = model(
                text_emb, image_lat, force_concept=intervened_labels
            )
            concept_logits_int = model.encode_only(recon_text_int.detach(), recon_image_int.detach())
            intervened = {"concept_logits": concept_logits_int, "target": intervened_labels}

            losses = loss_fn(
                outputs=outputs,
                targets=(text_emb, image_lat),
                pseudo_labels=labels,
                intervened=intervened,
            )

        optimizer.zero_grad()
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses["total"].item()
        total_recon += losses["recon"]
        total_recon_text += losses["recon_text"]
        total_recon_image += losses["recon_image"]
        total_concept += losses["concept"]
        total_intervene += losses["intervene"]
        total_adversarial += losses["adversarial"]
        num_batches += 1

        if (batch_idx + 1) % print_freq == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {losses['total'].item():.4f} | "
                f"Recon: {losses['recon']:.4f} (txt {losses['recon_text']:.4f} / img {losses['recon_image']:.4f}) | "
                f"Concept: {losses['concept']:.4f} | "
                f"Intervene: {losses['intervene']:.4f}"
            )

    avg_loss = total_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_recon_text = total_recon_text / num_batches
    avg_recon_image = total_recon_image / num_batches
    avg_concept = total_concept / num_batches
    avg_intervene = total_intervene / num_batches
    avg_adversarial = total_adversarial / num_batches

    return {
        "loss": avg_loss,
        "recon": avg_recon,
        "recon_text": avg_recon_text,
        "recon_image": avg_recon_image,
        "concept": avg_concept,
        "intervene": avg_intervene,
        "adversarial": avg_adversarial,
    }


def validate(model, loss_fn, dataloader, device):
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_recon_text = 0.0
    total_recon_image = 0.0
    total_concept = 0.0
    total_intervene = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            text_emb = batch["text_emb"].to(device)
            image_lat = batch["image_lat"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(text_emb, image_lat)

                intervened_labels = _build_intervention_targets(labels)
                recon_text_int, recon_image_int, _, _ = model(
                    text_emb, image_lat, force_concept=intervened_labels
                )
                concept_logits_int = model.encode_only(recon_text_int, recon_image_int)
                intervened = {"concept_logits": concept_logits_int, "target": intervened_labels}

                losses = loss_fn(
                    outputs=outputs,
                    targets=(text_emb, image_lat),
                    pseudo_labels=labels,
                    intervened=intervened,
                )

            total_loss += losses["total"].item()
            total_recon += losses["recon"]
            total_recon_text += losses["recon_text"]
            total_recon_image += losses["recon_image"]
            total_concept += losses["concept"]
            total_intervene += losses["intervene"]
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "recon": total_recon / num_batches,
        "recon_text": total_recon_text / num_batches,
        "recon_image": total_recon_image / num_batches,
        "concept": total_concept / num_batches,
        "intervene": total_intervene / num_batches,
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    checkpoint_dir,
    is_best=False,
    hf_api=None,
    hf_repo_id=None,
    hf_branch="main",
    keep_local=True,
    config_path=None,
):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    checkpoint_name = f"checkpoint_epoch_{epoch:04d}.pt"
    best_name = "best_model.pt"

    def upload_file(local_path, remote_name):
        if hf_api is None or hf_repo_id is None:
            return
        hf_api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=hf_repo_id,
            repo_type="model",
            revision=hf_branch,
        )

    # Save/upload current checkpoint
    checkpoint_path = checkpoint_dir / checkpoint_name
    if keep_local or hf_api is None:
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint to {checkpoint_path}")
    if hf_api is not None and hf_repo_id is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_ckpt = Path(tmpdir) / checkpoint_name
            torch.save(checkpoint, tmp_ckpt)
            upload_file(tmp_ckpt, checkpoint_name)
            print(f"  Uploaded checkpoint to HF: {hf_repo_id}/{checkpoint_name}")

    # Save/upload best checkpoint
    if is_best:
        best_path = checkpoint_dir / best_name
        if keep_local or hf_api is None:
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")
        if hf_api is not None and hf_repo_id is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_best = Path(tmpdir) / best_name
                torch.save(checkpoint, tmp_best)
                upload_file(tmp_best, best_name)
                print(f"  Uploaded best model to HF: {hf_repo_id}/{best_name}")

    # Upload config for reference
    if config_path is not None and config_path.exists() and hf_api is not None and hf_repo_id is not None:
        try:
            upload_file(config_path, config_path.name)
            print(f"  Uploaded config to HF: {hf_repo_id}/{config_path.name}")
        except Exception as e:
            print(f"  Warning: failed to upload config: {e}")

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
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Attention heads for transformer encoder/decoder')
    parser.add_argument('--num_encoder_layers', type=int, default=2,
                        help='Transformer encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=2,
                        help='Transformer decoder layers')
    parser.add_argument('--ff_dim', type=int, default=2048,
                        help='Feed-forward width inside transformer blocks')
    parser.add_argument('--bottleneck_tokens', type=int, default=2,
                        help='Number of learned concept tokens passed through encoder/decoder memory')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout used inside transformer blocks')
    parser.add_argument('--use_adversarial', action='store_true',
                        help='Use adversarial training (soft bottleneck)')
    parser.add_argument('--down_proj_dim', type=int, default=512,
                        help='Model dimension after input projection (reduces memory).')
    
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
    parser.add_argument('--lambda_intervene', type=float, default=1.0,
                        help='Weight for intervention cycle loss')
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
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='W&B project name (logging disabled if not set)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity/user (optional)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name (optional)')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='W&B mode')
    parser.add_argument('--hf_repo_id', type=str, default=None,
                        help='Hugging Face repo to upload checkpoints (model repo). If set, checkpoints upload instead of local save.')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='Hugging Face token (defaults to HF_TOKEN env or cached).')
    parser.add_argument('--hf_branch', type=str, default='main',
                        help='Hugging Face repo branch to upload to.')
    parser.add_argument('--hf_keep_local', action='store_true',
                        help='Keep local checkpoint copies even when uploading to HF.')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # W&B setup
    wandb_run = None
    hf_api = None

    if args.wandb_project and args.wandb_mode != 'disabled':
        if wandb is None:
            print("wandb not installed; skipping W&B logging.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
                mode=args.wandb_mode,
            )
            wandb.config.update({"device": str(device)}, allow_val_change=True)

    # HF Hub setup
    if args.hf_repo_id:
        if HfApi is None:
            print("huggingface_hub not installed; cannot upload checkpoints.")
        else:
            token = args.hf_token or os.environ.get("HF_TOKEN") or (HfFolder.get_token() if HfFolder else None)
            if token is None:
                print("No HF token provided; set --hf_token or HF_TOKEN env to enable uploads.")
            else:
                hf_api = HfApi(token=token)
                try:
                    hf_api.create_repo(args.hf_repo_id, repo_type="model", exist_ok=True, private=True)
                    print(f"HF repo ready: {args.hf_repo_id} (private)")
                except Exception as e:
                    print(f"Warning: could not ensure HF repo {args.hf_repo_id}: {e}")
                    hf_api = None
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    config_path = checkpoint_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training config to {config_path}")
    if hf_api is not None and args.hf_repo_id:
        try:
            hf_api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo='training_config.json',
                repo_id=args.hf_repo_id,
                repo_type="model",
                revision=args.hf_branch,
            )
            print(f"Uploaded training config to HF: {args.hf_repo_id}/training_config.json")
        except Exception as e:
            print(f"Warning: failed to upload training config to HF: {e}")
    
    # Initialize models
    print("\nInitializing models...")
    model = ConceptBottleneckTransformer(
        text_len=args.seq_len,
        image_len=args.num_patches,
        embed_dim=args.embed_dim,
        concept_dim=args.concept_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        ff_dim=args.ff_dim,
        bottleneck_tokens=args.bottleneck_tokens,
        dropout=args.dropout,
        use_adversarial=args.use_adversarial,
        down_proj_dim=args.down_proj_dim,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Unified CBM parameters: {total_params:,}")
    
    # Initialize loss function
    loss_fn = CombinedLoss(
        lambda_recon=args.lambda_recon,
        lambda_concept=args.lambda_concept,
        lambda_intervene=args.lambda_intervene,
        lambda_adv=args.lambda_adv,
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
        model.load_state_dict(checkpoint['model_state_dict'])
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
            model=model,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            loss_fn=loss_fn,
            dataloader=val_loader,
            device=device
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"    Train Recon: {train_metrics['recon']:.4f}")
        print(f"      Train Recon Text: {train_metrics['recon_text']:.4f}")
        print(f"      Train Recon Image: {train_metrics['recon_image']:.4f}")
        print(f"    Train Concept: {train_metrics['concept']:.4f}")
        print(f"    Train Intervene: {train_metrics['intervene']:.4f}")
        if args.use_adversarial:
            print(f"    Train Adversarial: {train_metrics['adversarial']:.4f}")
        
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"    Val Recon: {val_metrics['recon']:.4f}")
        print(f"      Val Recon Text: {val_metrics['recon_text']:.4f}")
        print(f"      Val Recon Image: {val_metrics['recon_image']:.4f}")
        print(f"    Val Concept: {val_metrics['concept']:.4f}")
        print(f"    Val Intervene: {val_metrics['intervene']:.4f}")

        # W&B logging (relative metrics include gap to previous best)
        if wandb_run is not None:
            prev_best = best_val_loss
            best_val_for_log = min(prev_best, val_metrics["loss"])
            gap_prev_best = None if not torch.isfinite(torch.tensor(prev_best)) else val_metrics["loss"] - prev_best
            improvement = 0.0
            if gap_prev_best is not None and gap_prev_best < 0:
                improvement = -gap_prev_best

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/recon": train_metrics["recon"],
                    "train/recon_text": train_metrics["recon_text"],
                    "train/recon_image": train_metrics["recon_image"],
                    "train/concept": train_metrics["concept"],
                    "train/intervene": train_metrics["intervene"],
                    "train/adversarial": train_metrics.get("adversarial", 0.0),
                    "val/loss": val_metrics["loss"],
                    "val/recon": val_metrics["recon"],
                    "val/recon_text": val_metrics["recon_text"],
                    "val/recon_image": val_metrics["recon_image"],
                    "val/concept": val_metrics["concept"],
                    "val/intervene": val_metrics["intervene"],
                    "val/best_gap": gap_prev_best if gap_prev_best is not None else float("nan"),
                    "val/improvement": improvement,
                    "best/val_loss": best_val_for_log if torch.isfinite(torch.tensor(best_val_for_log)) else float("inf"),
                },
                step=epoch + 1,
            )
        
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
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={'train': train_metrics, 'val': val_metrics, 'best_val_loss': best_val_loss},
                checkpoint_dir=checkpoint_dir,
                is_best=is_best,
                hf_api=hf_api,
                hf_repo_id=args.hf_repo_id,
                hf_branch=args.hf_branch,
                keep_local=args.hf_keep_local or not args.hf_repo_id,
                config_path=config_path,
            )
        
        # Apply early stopping
        if epochs_without_improve >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {epochs_without_improve} epochs without improvement.")
            break
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"{'='*60}")

    if wandb_run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
