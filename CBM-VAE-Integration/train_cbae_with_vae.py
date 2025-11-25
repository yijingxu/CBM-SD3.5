"""
Improved CVPR-style training script with REAL latent→image decoder.

This script organizes training around three conceptual objectives:

A. Reconstruction objectives (Lr1, Lr2):
   - Lr1: Latent reconstruction loss
   - Lr2: Image reconstruction loss (REAL - uses VAE decoder)

B. Concept alignment objective (Lc):
   - Align encoder concepts with CLIP pseudo-labels on original images

C. Intervention objectives (Li1, Li2, preservation):
   - Li1: Teacher-supervised intervention (CLIP validates REAL decoded intervened images)
   - Li2: Cyclic consistency (re-encode intervened latents)
   - Preservation: Non-intervened concepts stay stable

KEY CHANGES FROM train_cbae_structured.py:
1. Added frozen VAE decoder from SD3.5 pipeline
2. Lr2 is now a REAL image reconstruction loss using VAE
3. Li1 now depends on REAL images decoded from intervened latents (not noise approximation)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse

from diffusers import StableDiffusion3Pipeline
from models.cbae_sd35 import ConceptBottleneckAE
from clip_pseudolabeler import CLIPPseudoLabeler


class SD35LatentDataset(Dataset):
    """Dataset for SD3.5 latents with concept labels"""

    def __init__(self, data_json_path: str):
        with open(data_json_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples from {data_json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load latent
        latent_path = sample['latent_path']
        latent = np.load(latent_path)
        latent = torch.from_numpy(latent).float()

        # Load the pre-generated image
        from PIL import Image
        import torchvision.transforms as T

        image_path = sample['image_path']
        image = Image.open(image_path).convert('RGB')

        # Transform to tensor [3, H, W] in range [0, 1]
        transform = T.ToTensor()
        image_tensor = transform(image)

        # Get concept labels (ground truth, for reference only)
        smiling = 1.0 if sample['smiling'] else 0.0
        glasses = 1.0 if sample['glasses'] else 0.0
        concept_labels = torch.tensor([smiling, glasses], dtype=torch.float32)

        return {
            'latent': latent,
            'image': image_tensor,
            'concept_labels': concept_labels,
            'sample_id': sample['sample_id'],
        }


def decode_latents_to_images(latents, vae):
    """
    Decode SD3.5 latents to RGB images using the frozen VAE decoder.

    This is the KEY ADDITION that enables:
    - Real Lr2 (image reconstruction loss)
    - Real Li1 (CLIP validation of decoded intervened images)

    Args:
        latents: (batch_size, latent_dim) or (batch_size, 16, H, W) SD3.5 latents
        vae: Frozen SD3.5 VAE decoder (parameters frozen, but forward is differentiable)

    Returns:
        images: (batch_size, 3, H, W) in range [0, 1]
    """
    # Reshape if needed: handle various CB-AE decoder output formats
    if latents.dim() == 2:
        # Format: (batch_size, latent_dim) - flattened
        batch_size = latents.size(0)
        latent_dim = latents.size(1)

        # Calculate spatial dimensions: latent_dim = 16 * H * W
        # For SD3.5: 511488 = 16 * 96 * 333 (portrait aspect ratio)
        # OR: 65536 = 16 * 64 * 64 (square)
        if latent_dim == 511488:
            # Portrait: 16 channels x 96 (height) x 333 (width)
            latents = latents.view(batch_size, 16, 96, 333)
        elif latent_dim == 65536:
            # Square: 16 channels x 64 x 64
            latents = latents.view(batch_size, 16, 64, 64)
        else:
            # Try to infer square dimensions
            spatial_size = latent_dim // 16
            H = W = int(spatial_size ** 0.5)
            latents = latents.view(batch_size, 16, H, W)

    elif latents.dim() == 4 and latents.size(1) == 1:
        # Format: (batch, 1, H, W) - CB-AE decoder output in wrong shape
        # Need to reshape to (batch, 16, H', W') where total elements match
        batch_size = latents.size(0)
        total_elements = latents.size(1) * latents.size(2) * latents.size(3)

        if total_elements == 511488:
            # Reshape (batch, 1, 333, 1536) -> (batch, 16, 96, 333)
            latents = latents.reshape(batch_size, 16, 96, 333)
        elif total_elements == 65536:
            # Reshape (batch, 1, 256, 256) -> (batch, 16, 64, 64)
            latents = latents.reshape(batch_size, 16, 64, 64)
        else:
            raise ValueError(f"Unexpected latent shape: {latents.shape} with {total_elements} elements")

    elif latents.dim() == 4 and latents.size(1) != 16:
        raise ValueError(f"Expected 16 channels for SD3.5 latents, got {latents.size(1)}")

    # Convert to VAE's dtype (bfloat16) to avoid dtype mismatch
    latents = latents.to(dtype=vae.dtype)

    # VAE expects latents to be scaled
    # SD3.5 VAE scaling factor is typically 1.5305
    latents_scaled = latents / vae.config.scaling_factor

    # Decode to images (no grad context for VAE parameters, but gradients flow through latents)
    # VAE output is in range [-1, 1]
    images = vae.decode(latents_scaled, return_dict=False)[0]

    # Convert from [-1, 1] to [0, 1] for CLIP and loss computation
    # Convert back to float32 for loss computation
    images = (images.float() + 1.0) / 2.0
    images = torch.clamp(images, 0.0, 1.0)

    return images


def encode_and_decode_latents(model, latents):
    """
    Helper: encode latents to concepts and decode back.

    Args:
        model: ConceptBottleneckAE model
        latents: (batch_size, latent_dim) original latents

    Returns:
        concepts: (batch_size, total_concept_dim)
        reconstructed_latents: same shape as latents
    """
    concepts = model.encode(latents)
    reconstructed_latents = model.decode(concepts, target_shape=latents.shape)
    return concepts, reconstructed_latents


def apply_concept_intervention(concepts, concept_idx, target_value):
    """
    Apply intervention to a specific concept dimension.

    Args:
        concepts: (batch_size, total_concept_dim)
        concept_idx: index of concept to intervene on (0 for smiling, 1 for glasses)
        target_value: 0 or 1 (will be mapped to logits ±5.0)

    Returns:
        intervened_concepts: (batch_size, total_concept_dim) with one dimension modified
    """
    intervened_concepts = concepts.clone()
    if target_value == 1:
        intervened_concepts[:, concept_idx] = 5.0  # High positive logit
    else:
        intervened_concepts[:, concept_idx] = -5.0  # High negative logit
    return intervened_concepts


def compute_standard_losses(
    model,
    latents,
    original_images,
    clip_labeler,
    vae_decoder,
    args
):
    """
    Compute losses for standard training step (objectives A and B).

    This step trains the encoder and decoder to:
    - Reconstruct latents accurately (Lr1)
    - Reconstruct images accurately (Lr2) - NOW REAL IMAGE LOSS WITH VAE!
    - Align concepts with CLIP pseudo-labels (Lc)

    Args:
        model: ConceptBottleneckAE
        latents: (batch_size, latent_dim)
        original_images: (batch_size, 3, H, W)
        clip_labeler: CLIPPseudoLabeler
        vae_decoder: Frozen SD3.5 VAE decoder
        args: training arguments

    Returns:
        dict with keys:
            - loss_recon_latent (Lr1): float
            - loss_recon_image (Lr2): float
            - loss_concept_align (Lc): float
            - concepts: tensor (batch_size, total_concept_dim)
    """
    batch_size = latents.size(0)

    # Encode and decode
    concepts, reconstructed_latents = encode_and_decode_latents(model, latents)

    # Lr1: Latent reconstruction loss
    # Encourages decoder to reconstruct original latents from concepts
    loss_recon_latent = F.mse_loss(reconstructed_latents, latents)

    # Lr2: Image reconstruction loss - NOW REAL!
    # Decode both original and reconstructed latents to images using frozen VAE
    # Gradients flow: loss → reconstructed_images → reconstructed_latents → decoder
    reconstructed_images = decode_latents_to_images(
        reconstructed_latents, vae_decoder
    )

    # Resize reconstructed images to match original images size if needed
    if reconstructed_images.shape != original_images.shape:
        reconstructed_images = F.interpolate(
            reconstructed_images,
            size=original_images.shape[2:],  # (H, W)
            mode='bilinear',
            align_corners=False
        )

    # Original images are pre-generated, so we compare against them directly
    # This encourages the CB-AE to preserve visual information
    loss_recon_image = F.mse_loss(reconstructed_images, original_images)

    # Lc: Concept alignment loss with CLIP pseudo-labels
    # Get CLIP pseudo-labels from ORIGINAL images
    # IMPORTANT: We don't wrap this in torch.no_grad() because we need gradients
    # to flow back through the model parameters. CLIP itself is frozen (its parameters
    # don't update), but we need gradients w.r.t. the concepts produced by our encoder.
    clip_probs, clip_labels = clip_labeler.get_pseudo_labels(
        original_images, return_prob=True
    )

    loss_concept_align = 0.0
    for c_idx in range(2):  # Loop over: smiling (0), glasses (1)
        # Convert CLIP predictions to binary targets
        target = clip_labels[c_idx].float().unsqueeze(1)  # (batch_size, 1)
        pred = concepts[:, c_idx:c_idx+1]  # (batch_size, 1)

        # Use CLIP confidence scores as weights
        # Higher confidence = higher weight in loss
        weights = clip_probs[c_idx].detach()  # Detach weights only, not predictions

        # Binary cross entropy with logits (concepts are logits, not probabilities)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        loss_concept_align += (loss * weights.unsqueeze(1)).mean()

    loss_concept_align = loss_concept_align / 2  # Average over concepts

    return {
        'loss_recon_latent': loss_recon_latent,
        'loss_recon_image': loss_recon_image,
        'loss_concept_align': loss_concept_align,
        'concepts': concepts,
    }


def compute_intervention_losses(
    model,
    latents,
    original_images,
    original_concepts,
    clip_labeler,
    vae_decoder,
    concept_idx,
    target_value,
    args
):
    """
    Compute losses for intervention training step (objective C).

    This step trains the decoder to create latent changes that:
    - Are recognized by CLIP as matching the intervention target (Li1) - NOW WITH REAL IMAGES!
    - Can be re-encoded to recover the intervention (Li2)
    - Preserve other concepts (preservation)

    Args:
        model: ConceptBottleneckAE
        latents: (batch_size, latent_dim)
        original_images: (batch_size, 3, H, W)
        original_concepts: (batch_size, total_concept_dim) - detached
        clip_labeler: CLIPPseudoLabeler
        vae_decoder: Frozen SD3.5 VAE decoder
        concept_idx: 0 (smiling) or 1 (glasses)
        target_value: 0 or 1
        args: training arguments

    Returns:
        dict with keys:
            - loss_interv_teacher (Li1): float
            - loss_interv_cyclic (Li2): float
            - loss_preservation: float
    """
    batch_size = latents.size(0)

    # Apply intervention to concepts
    # E.g., if concept_idx=0 (smiling) and target_value=1, set smiling logit to 5.0
    intervened_concepts = apply_concept_intervention(
        original_concepts.detach(), concept_idx, target_value
    )

    # Decode intervened concepts to latents
    # This is where the decoder learns to create meaningful latent changes
    intervened_latents = model.decode(intervened_concepts, target_shape=latents.shape)

    # Decode intervened latents to REAL images using frozen VAE
    # KEY CHANGE: No more noise approximation! We use the actual VAE decoder.
    # Gradients flow: Li1 loss → intervened_images → intervened_latents → CB-AE decoder
    intervened_images = decode_latents_to_images(
        intervened_latents, vae_decoder
    )

    # Resize intervened images to match original images size if needed
    if intervened_images.shape != original_images.shape:
        intervened_images = F.interpolate(
            intervened_images,
            size=original_images.shape[2:],  # (H, W)
            mode='bilinear',
            align_corners=False
        )

    # Li1: Teacher-supervised intervention loss
    # Get CLIP predictions on REAL decoded intervened images
    # IMPORTANT: No torch.no_grad() here! We need gradients to flow back through
    # the decoder that created intervened_latents and thus intervened_images.
    clip_probs_interv, clip_labels_interv = clip_labeler.get_pseudo_labels(
        intervened_images, return_prob=True
    )

    # Target: CLIP should recognize the intervened concept value
    # E.g., if we set glasses=1, CLIP should predict glasses with high probability
    target_clip = torch.full((batch_size, 1), float(target_value), device=latents.device)
    pred_clip_prob = clip_probs_interv[concept_idx].unsqueeze(1)

    # Match dtypes (CLIP may return float16)
    target_clip = target_clip.to(pred_clip_prob.dtype)

    # CLIP intervention loss: ensure intervened images show the target concept
    # This trains the decoder to create latent changes that produce REAL visual changes
    loss_interv_teacher = F.binary_cross_entropy(pred_clip_prob, target_clip)

    # Li2: Cyclic consistency on intervened concepts
    # Re-encode the intervened latents and check if we get back the intervention
    recon_intervened_concepts = model.encode(intervened_latents)

    # Target: the intervened concept dimension should be ±5.0
    if target_value == 1:
        target_logit = torch.ones(batch_size, 1, device=latents.device) * 5.0
    else:
        target_logit = torch.ones(batch_size, 1, device=latents.device) * -5.0

    # MSE loss between re-encoded concept and target
    loss_interv_cyclic = F.mse_loss(
        recon_intervened_concepts[:, concept_idx:concept_idx+1],
        target_logit
    )

    # Preservation loss: non-intervened concepts should stay similar to originals
    # E.g., if we intervene on smiling, glasses should remain unchanged
    loss_preservation = 0.0
    for c_idx in range(2):
        if c_idx != concept_idx:
            loss_preservation += F.mse_loss(
                recon_intervened_concepts[:, c_idx],
                original_concepts[:, c_idx].detach()
            )

    return {
        'loss_interv_teacher': loss_interv_teacher,
        'loss_interv_cyclic': loss_interv_cyclic,
        'loss_preservation': loss_preservation,
    }


def train_epoch(
    model: ConceptBottleneckAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    optimizer_interv: torch.optim.Optimizer,
    clip_labeler: CLIPPseudoLabeler,
    vae_decoder,
    device: torch.device,
    epoch: int,
    args,
):
    """
    Train for one epoch with structured losses.

    Each batch performs two optimization steps:
    1. Standard step: optimizes reconstruction and concept alignment (A, B)
    2. Intervention step: optimizes intervention objectives (C)
    """
    model.train()
    vae_decoder.eval()  # Keep VAE in eval mode (frozen teacher)

    # Accumulators for logging
    total_losses = {
        'recon_latent': 0.0,
        'recon_image': 0.0,
        'concept_align': 0.0,
        'interv_teacher': 0.0,
        'interv_cyclic': 0.0,
        'preservation': 0.0,
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        latents = batch['latent'].to(device)
        original_images = batch['image'].to(device)
        batch_size = latents.size(0)

        # ============================================================
        # STANDARD TRAINING STEP (Objectives A and B)
        # ============================================================
        optimizer.zero_grad()

        # Compute standard losses: Lr1, Lr2 (REAL!), Lc
        standard_losses = compute_standard_losses(
            model, latents, original_images, clip_labeler, vae_decoder, args
        )

        # Weighted combination of standard losses
        loss_standard = (
            args.lambda_recon_latent * standard_losses['loss_recon_latent'] +
            args.lambda_recon_image * standard_losses['loss_recon_image'] +
            args.lambda_concept_align * standard_losses['loss_concept_align']
        )

        loss_standard.backward()
        optimizer.step()

        # ============================================================
        # INTERVENTION TRAINING STEP (Objective C)
        # ============================================================
        optimizer_interv.zero_grad()

        # Randomly select which concept to intervene on and the target value
        rand_concept_idx = torch.randint(0, 2, (1,)).item()  # 0=smiling, 1=glasses
        target_value = torch.randint(0, 2, (1,)).item()  # 0 or 1

        # Get original concepts (detached to prevent gradients from intervention step
        # affecting the encoder trained in the standard step)
        with torch.no_grad():
            original_concepts = model.encode(latents)

        # Compute intervention losses: Li1 (REAL!), Li2, preservation
        intervention_losses = compute_intervention_losses(
            model, latents, original_images, original_concepts,
            clip_labeler, vae_decoder, rand_concept_idx, target_value, args
        )

        # Weighted combination of intervention losses
        loss_intervention = (
            args.lambda_interv_teacher * intervention_losses['loss_interv_teacher'] +
            args.lambda_interv_cyclic * intervention_losses['loss_interv_cyclic'] +
            args.lambda_preservation * intervention_losses['loss_preservation']
        )

        loss_intervention.backward()
        optimizer_interv.step()

        # ============================================================
        # LOGGING
        # ============================================================
        total_losses['recon_latent'] += standard_losses['loss_recon_latent'].item()
        total_losses['recon_image'] += standard_losses['loss_recon_image'].item()
        total_losses['concept_align'] += standard_losses['loss_concept_align'].item()
        total_losses['interv_teacher'] += intervention_losses['loss_interv_teacher'].item()
        total_losses['interv_cyclic'] += intervention_losses['loss_interv_cyclic'].item()
        total_losses['preservation'] += intervention_losses['loss_preservation'].item()

        # Progress bar shows key losses
        pbar.set_postfix({
            'Lr1': f"{standard_losses['loss_recon_latent'].item():.2f}",
            'Lr2': f"{standard_losses['loss_recon_image'].item():.4f}",
            'Lc': f"{standard_losses['loss_concept_align'].item():.3f}",
            'Li1': f"{intervention_losses['loss_interv_teacher'].item():.3f}",
        })

    # Compute and print epoch averages
    n = len(dataloader)
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Lr1 (Latent Recon):     {total_losses['recon_latent']/n:.4f}")
    print(f"  Lr2 (Image Recon REAL): {total_losses['recon_image']/n:.6f}")
    print(f"  Lc  (Concept Align):    {total_losses['concept_align']/n:.4f}")
    print(f"  Li1 (Interv REAL):      {total_losses['interv_teacher']/n:.4f}")
    print(f"  Li2 (Interv Cyclic):    {total_losses['interv_cyclic']/n:.4f}")
    print(f"  Preservation:           {total_losses['preservation']/n:.4f}")

    return total_losses['recon_latent']/n


def main():
    parser = argparse.ArgumentParser(
        description="Train CB-AE with REAL VAE decoder (A: Reconstruction, B: Concept Alignment, C: Intervention)"
    )

    # Data
    parser.add_argument('--data_json', type=str, default='TrainingData/concept_labels.json',
                        help='Path to JSON file with training data')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')

    # Model architecture
    parser.add_argument('--latent_dim', type=int, default=511488,
                        help='Dimension of SD3.5 latents')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for encoder/decoder')
    parser.add_argument('--num_encoder_layers', type=int, default=3,
                        help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=3,
                        help='Number of decoder layers')

    # SD3.5 pipeline
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-3.5-medium',
                        help='HuggingFace model ID for SD3.5 (for VAE decoder)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate for standard optimizer')
    parser.add_argument('--lr_interv', type=float, default=0.0002,
                        help='Learning rate for intervention optimizer')

    # Loss weights - Reconstruction objectives (A)
    parser.add_argument('--lambda_recon_latent', type=float, default=1.0,
                        help='Weight for Lr1: latent reconstruction loss')
    parser.add_argument('--lambda_recon_image', type=float, default=1.0,
                        help='Weight for Lr2: image reconstruction loss (REAL with VAE)')

    # Loss weights - Concept alignment objective (B)
    parser.add_argument('--lambda_concept_align', type=float, default=1.0,
                        help='Weight for Lc: concept alignment with CLIP pseudo-labels')

    # Loss weights - Intervention objectives (C)
    parser.add_argument('--lambda_interv_teacher', type=float, default=2.0,
                        help='Weight for Li1: teacher-supervised intervention loss (REAL with VAE)')
    parser.add_argument('--lambda_interv_cyclic', type=float, default=1.0,
                        help='Weight for Li2: cyclic consistency on intervened concepts')
    parser.add_argument('--lambda_preservation', type=float, default=0.5,
                        help='Weight for preservation of non-intervened concepts')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================================================
    # Load SD3.5 VAE Decoder (Frozen Teacher) - Memory Optimized
    # ============================================================
    print("\nLoading SD3.5 VAE decoder (frozen teacher)...")
    from apps.env_utils import get_env_var
    from diffusers import AutoencoderKL
    HF_TOKEN = get_env_var("HF_TOKEN")

    # Load ONLY the VAE to save memory (not the entire pipeline)
    vae_decoder = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Freeze VAE parameters (frozen teacher - no training)
    vae_decoder.requires_grad_(False)
    vae_decoder.eval()

    print(f"VAE decoder loaded and frozen!")
    print(f"  Scaling factor: {vae_decoder.config.scaling_factor}")
    print(f"  Parameters frozen: {not any(p.requires_grad for p in vae_decoder.parameters())}")

    # Free memory - we don't need the full pipeline
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # Load CLIP Pseudo-Labeler
    # ============================================================
    print("\nLoading CLIP Pseudo-Labeler...")
    clip_labeler = CLIPPseudoLabeler(device=device)
    print("CLIP loaded! (parameters frozen, gradients flow through inputs)")

    # ============================================================
    # Load dataset
    # ============================================================
    print("\nLoading dataset...")
    dataset = SD35LatentDataset(args.data_json)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # ============================================================
    # Create model
    # ============================================================
    print("\nCreating model...")
    concept_config = {
        'concept_names': ['smiling', 'glasses'],
        'concept_dims': [1, 1],
        'concept_types': ['binary', 'binary'],
    }

    model = ConceptBottleneckAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        concept_config=concept_config,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # ============================================================
    # Create optimizers
    # ============================================================
    # Two separate optimizers allow different learning rates for different objectives
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.99))
    optimizer_interv = torch.optim.Adam(model.parameters(), lr=args.lr_interv, betas=(0.5, 0.99))

    print(f"Standard optimizer LR: {args.lr}")
    print(f"Intervention optimizer LR: {args.lr_interv}")

    # ============================================================
    # Print loss configuration
    # ============================================================
    print("\n" + "="*80)
    print("Loss Configuration:")
    print("="*80)
    print("A. Reconstruction Objectives:")
    print(f"   Lr1 (latent recon):   λ = {args.lambda_recon_latent}")
    print(f"   Lr2 (image recon):    λ = {args.lambda_recon_image} ✨ REAL WITH VAE!")
    print("\nB. Concept Alignment Objective:")
    print(f"   Lc  (CLIP align):     λ = {args.lambda_concept_align}")
    print("\nC. Intervention Objectives:")
    print(f"   Li1 (teacher):        λ = {args.lambda_interv_teacher} ✨ REAL WITH VAE!")
    print(f"   Li2 (cyclic):         λ = {args.lambda_interv_cyclic}")
    print(f"   Preservation:         λ = {args.lambda_preservation}")
    print("="*80 + "\n")

    # ============================================================
    # Training loop
    # ============================================================
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*60)

        train_loss = train_epoch(
            model, train_loader, optimizer, optimizer_interv,
            clip_labeler, vae_decoder, device, epoch, args
        )

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = Path(args.checkpoint_dir) / f"cbae_with_vae_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_interv_state_dict': optimizer_interv.state_dict(),
                'train_loss': train_loss,
                'concept_config': concept_config,
                'hidden_dim': args.hidden_dim,
                'num_encoder_layers': args.num_encoder_layers,
                'num_decoder_layers': args.num_decoder_layers,
                'args': vars(args),  # Save all hyperparameters
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    main()
