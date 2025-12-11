"""
Training script for CB-AE with DDPM models.

This script implements the DDPM-specific training procedure:
1. Load clean images x_0 from dataset
2. Sample early timesteps t <= max_timestep
3. Add noise to get x_t
4. Extract noisy latent w_t from UNet encoder
5. Apply CB-AE to get reconstructed latent
6. Compute losses:
   - Latent reconstruction loss
   - Concept alignment loss (using pseudo-labels from x_0)
   - DDPM noise prediction loss
   - Intervention losses
"""

import os
import sys
sys.path.append('.')
from utils.utils import get_dataset, create_image_grid, get_concept_index
import argparse
import numpy as np
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from models.cbae_ddpm import cbAE_DDPM_Trainable, DDPMNoiseSchedulerHelper
from models import clip_pseudolabeler
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from diffusers import DDIMScheduler


def get_pseudo_concept_loss(
    model,
    predicted_concepts,
    pseudolabel_concepts,
    pseudolabel_probs,
    device,
    pl_prob_thresh=0.9,
    dataset='celebahq',
    ignore_index=250,
    use_pl_thresh=True
):
    """
    Compute concept alignment loss using pseudo-labels.

    Args:
        model: Model with concept configuration
        predicted_concepts: Predicted concept logits [B, total_concept_dim]
        pseudolabel_concepts: List of pseudo-label tensors per concept
        pseudolabel_probs: List of probability tensors per concept
        pl_prob_thresh: Probability threshold for filtering
        use_pl_thresh: Whether to use threshold filtering
    """
    concept_loss = 0
    batch_size = predicted_concepts.shape[0]

    if dataset in ['celebahq', 'celeba64', 'cub', 'cub64']:
        # For CUB, don't use threshold (more varied probabilities)
        if dataset in ['cub', 'cub64']:
            use_pl_thresh = False

        if use_pl_thresh:
            for cdx in range(len(pseudolabel_concepts)):
                # Assign ignore_index for low-confidence predictions
                pseudolabel_concepts[cdx][pseudolabel_probs[cdx] < pl_prob_thresh] = ignore_index

        concepts = [curr_conc.long() for curr_conc in pseudolabel_concepts]
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')

    loss_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    concept_loss_lst = []
    num_valid_concepts = 0

    for c in range(model.n_concepts):
        start, end = get_concept_index(model, c)
        c_predicted_concepts = predicted_concepts[:, start:end]
        c_real_concepts = concepts[c]
        c_concept_loss = loss_ce(c_predicted_concepts, c_real_concepts)

        # Only accumulate if loss is valid (not nan)
        if not torch.isnan(c_concept_loss):
            concept_loss += c_concept_loss
            num_valid_concepts += 1

        concept_loss_lst.append(c_concept_loss)

    # If all concepts have nan loss, return 0 (will be handled by caller)
    if num_valid_concepts == 0:
        concept_loss = torch.tensor(0.0, device=device)

    return concept_loss, concept_loss_lst


def train_epoch(
    model,
    dataloader,
    noise_scheduler,
    pseudo_labeler,
    optimizer,
    optimizer_interv,
    config,
    device,
    epoch,
    writer,
    set_of_classes,
    args
):
    """Train for one epoch."""
    model.train()

    # Freeze DDPM generator
    for param in model.gen.parameters():
        param.requires_grad = False

    reconstr_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=250)

    pl_prob_thresh = config["train_config"]["pl_prob_thresh"]
    max_timestep = model.max_timestep
    batch_size = config["dataset"]["batch_size"]

    total_loss_accum = 0.0
    total_recon_loss_accum = 0.0
    total_concept_loss_accum = 0.0
    total_noise_loss_accum = 0.0
    total_interv_loss_accum = 0.0

    # Use steps_per_epoch from config, but don't exceed actual batches
    steps_per_epoch = config["train_config"].get("steps_per_epoch", len(dataloader))
    steps_per_epoch = min(steps_per_epoch, len(dataloader))

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', total=steps_per_epoch)
    num_batches = 0

    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= steps_per_epoch:
            break

        num_batches += 1
        # Get clean images x_0 and real labels (if available)
        real_labels = None
        if isinstance(batch, (tuple, list)):
            clean_images = batch[0].to(device)
            if len(batch) > 1:
                # Check if we have real labels
                # batch[1] can be:
                #   - List of lists: [[1, 0, 1, ...], [0, 1, 0, ...], ...]
                #   - Tensor: shape [batch_size, num_concepts]
                # We need format: [tensor([1, 0, ...]), tensor([0, 1, ...]), ...]

                if isinstance(batch[1], torch.Tensor):
                    # Tensor format: [batch_size, num_concepts] -> list of tensors
                    batch_labels_tensor = batch[1].to(device)
                    num_concepts = batch_labels_tensor.shape[1]
                    real_labels = []
                    for c_idx in range(num_concepts):
                        real_labels.append(batch_labels_tensor[:, c_idx].long())
                    if batch_idx == 0 and epoch == 0:
                        print(f"[INFO] Using real labels (tensor format): {num_concepts} concepts")

                elif isinstance(batch[1][0], list):
                    # List of lists format
                    batch_labels = batch[1]  # List of lists
                    num_concepts = len(batch_labels[0])
                    real_labels = []
                    for c_idx in range(num_concepts):
                        concept_c_labels = torch.tensor([sample[c_idx] for sample in batch_labels],
                                                       dtype=torch.long, device=device)
                        real_labels.append(concept_c_labels)
                    if batch_idx == 0 and epoch == 0:
                        print(f"[INFO] Using real labels (list format): {num_concepts} concepts")
        else:
            clean_images = batch.to(device)

        batch_size_actual = clean_images.shape[0]

        # -----------------
        # Train CB-AE: Reconstruction + Concept Alignment + Noise Prediction
        # -----------------
        optimizer.zero_grad()

        # Sample random timesteps from early range [0, max_timestep]
        timesteps = torch.randint(
            0, max_timestep + 1, (batch_size_actual,), device=device
        ).long()

        # Add noise to get x_t (forward diffusion process)
        noise = torch.randn_like(clean_images)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Get noisy latent w_t from UNet encoder (g1)
        latent_t, t_emb, unet_residual = model.get_noisy_latent(noisy_images, timesteps)

        # Pass through CB-AE
        concepts = model.cbae.encode(latent_t)
        recon_latent_t = model.cbae.decode(concepts)

        # Predict noise from reconstructed latent (g2)
        noise_pred = model.predict_noise_from_latent(recon_latent_t, t_emb, unet_residual)

        # Get labels: use real labels if available, otherwise use pseudo-labels
        if real_labels is not None:
            # Use real labels from dataset
            concept_labels = real_labels
            concept_probs = None  # No probabilities for real labels
            use_pl_thresh_flag = False  # Don't use threshold filtering for real labels
            if batch_idx == 0 and epoch == 0:
                print(f"[INFO] Training with REAL LABELS (no threshold filtering)")
        else:
            # Get pseudo-labels from CLEAN images (not generated images!)
            with torch.no_grad():
                pseudo_prob, pseudo_labels = pseudo_labeler.get_pseudo_labels(
                    clean_images, return_prob=True
                )
                concept_labels = [pl.detach() for pl in pseudo_labels]
                concept_probs = [pm.detach() for pm in pseudo_prob]
                use_pl_thresh_flag = True
            if batch_idx == 0 and epoch == 0:
                print(f"[INFO] Training with PSEUDO-LABELS from {pseudo_labeler.__class__.__name__}")

        # Loss 1: Latent reconstruction loss
        recon_loss = reconstr_loss(recon_latent_t, latent_t)

        # Loss 2: Concept alignment loss (using real labels or pseudo-labels from x_0)
        concept_loss, _ = get_pseudo_concept_loss(
            model, concepts, concept_labels, concept_probs,
            pl_prob_thresh=pl_prob_thresh, device=device, dataset=args.dataset,
            use_pl_thresh=use_pl_thresh_flag
        )

        # Loss 3: DDPM noise prediction loss
        noise_loss = reconstr_loss(noise_pred, noise)

        # Total loss for reconstruction phase
        if torch.isnan(concept_loss):
            loss = recon_loss + noise_loss
        else:
            loss = recon_loss + concept_loss + noise_loss

        loss.backward()
        optimizer.step()

        # -----------------
        # Train Intervention Capability
        # -----------------
        optimizer_interv.zero_grad()

        # Sample new timesteps for intervention
        timesteps_interv = torch.randint(
            0, max_timestep + 1, (batch_size_actual,), device=device
        ).long()

        # Add noise
        noise_interv = torch.randn_like(clean_images)
        noisy_images_interv = noise_scheduler.add_noise(clean_images, noise_interv, timesteps_interv)

        # Get latent
        latent_t_interv, t_emb_interv, unet_residual_interv = model.get_noisy_latent(
            noisy_images_interv, timesteps_interv
        )

        # Randomly choose concept to intervene
        rand_concept = torch.randint(0, len(set_of_classes), (1,)).item()
        concept_value = torch.randint(0, len(set_of_classes[rand_concept]), (1,)).item()

        with torch.no_grad():
            concepts_interv = model.cbae.encode(latent_t_interv)

            # Swap max value to intervened concept (same as GAN training)
            start_idx, end_idx = get_concept_index(model, rand_concept)
            intervened_concepts = concepts_interv.clone()
            curr_c_concepts = intervened_concepts[:, start_idx:end_idx]

            # Swap max value and intervened concept value
            old_vals = curr_c_concepts[:, concept_value].clone()
            max_val, max_ind = torch.max(curr_c_concepts, dim=1)
            curr_c_concepts[:, concept_value] = max_val
            for swap_idx, (curr_ind, curr_old_val) in enumerate(zip(max_ind, old_vals)):
                curr_c_concepts[swap_idx, curr_ind] = curr_old_val

            intervened_concepts[:, start_idx:end_idx] = curr_c_concepts
            intervened_concepts = intervened_concepts.detach()

            # Set GT label for intervention
            # Don't use concept_labels directly as they may contain ignore_index (250)
            # Instead, get fresh pseudo-labels for intervention
            with torch.no_grad():
                _, intervened_pseudo_labels = pseudo_labeler.get_pseudo_labels(
                    clean_images, return_prob=True
                )
                intervened_label = [pl.detach().clone() for pl in intervened_pseudo_labels]
                # Set the intervened concept to the target value
                intervened_label[rand_concept] = (
                    torch.ones((batch_size_actual,), device=device) * concept_value
                ).long()
                intervened_label = [temp_pl.detach() for temp_pl in intervened_label]

        # Decode intervened concepts
        intervened_latent = model.cbae.decode(intervened_concepts)

        # Predict noise from intervened latent
        noise_pred_interv = model.predict_noise_from_latent(
            intervened_latent, t_emb_interv, unet_residual_interv
        )

        # Generate intervened image by denoising (one-step approximation for speed)
        # x_0 = (x_t - sqrt(1-alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
        # For simplicity in training, we use DDPM's predict_start_from_noise if available
        # Otherwise, we skip L_i1 for now and only use L_i2

        # L_i1: External consistency - pseudo-labeler prediction on intervened image
        # Generate approximation of clean image from intervened latent
        with torch.no_grad():
            # For training speed, we use a one-step approximation:
            # x_0 ≈ (x_t - sqrt(1-alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
            alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps_interv].view(-1, 1, 1, 1).to(device)
            beta_prod_t = 1 - alpha_prod_t

            # Predict x_0 from x_t and noise_pred_interv
            pred_original_sample = (noisy_images_interv - beta_prod_t.sqrt() * noise_pred_interv) / alpha_prod_t.sqrt()
            pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

            # Convert from [-1, 1] to [0, 1] for pseudo-labeler
            intervened_gen_imgs = pred_original_sample.mul(0.5).add_(0.5)

        # Get pseudo-label predictions on intervened image
        pred_logits = pseudo_labeler.get_soft_pseudo_labels(intervened_gen_imgs)

        # Debug L_i1
        if batch_idx == 0 and epoch == 0:
            print(f"\n[DEBUG] L_i1 External Consistency:")
            print(f"  intervened_gen_imgs: min={intervened_gen_imgs.min().item():.3f}, max={intervened_gen_imgs.max().item():.3f}")
            print(f"  has nan: {torch.isnan(intervened_gen_imgs).any()}")
            print(f"  pred_logits shapes: {[pl.shape for pl in pred_logits]}")
            print(f"  pred_logits[0] sample: {pred_logits[0][0]}")
            print(f"  intervened_label[0]: {intervened_label[0]}")

        # Compute L_i1: external consistency loss
        intervened_pseudo_label_loss = torch.tensor(0.0, device=device)
        for cdx, (curr_logits, actual_pl) in enumerate(zip(pred_logits, intervened_label)):
            curr_loss = ce_loss(curr_logits, actual_pl)
            if batch_idx == 0 and epoch == 0:
                print(f"  Concept {cdx} CE loss: {curr_loss.item()}")
            intervened_pseudo_label_loss += curr_loss

        # L_i2: Internal consistency - re-encode intervened latent
        recon_intervened_concepts = model.cbae.encode(intervened_latent)

        # Intervention concept loss (L_i2: Internal consistency)
        intervened_concept_loss, intervened_concept_loss_lst = get_pseudo_concept_loss(
            model, recon_intervened_concepts, intervened_label, None,
            use_pl_thresh=False, device=device, dataset=args.dataset
        )

        # Total intervention loss = L_i1 + L_i2
        total_intervened_loss = intervened_pseudo_label_loss + intervened_concept_loss

        # Debug: Check why loss might be nan
        if batch_idx == 0 and epoch == 0:
            print(f"\n[DEBUG] Intervention Loss Components:")
            print(f"  L_i1 (external): {intervened_pseudo_label_loss.item()}")
            print(f"  L_i2 (internal): {intervened_concept_loss.item()}")
            print(f"  Total: {total_intervened_loss.item()}")
            print(f"  Has nan: {torch.isnan(total_intervened_loss)}")

        # Check if loss is valid
        if not torch.isnan(total_intervened_loss) and total_intervened_loss.item() > 0:
            total_intervened_loss.backward()
            optimizer_interv.step()
        else:
            # Set to 0 for logging
            total_intervened_loss = torch.tensor(0.0, device=device)

        # Accumulate losses
        total_loss_accum += loss.item()
        total_recon_loss_accum += recon_loss.item()
        total_concept_loss_accum += concept_loss.item() if not torch.isnan(concept_loss) else 0.0
        total_noise_loss_accum += noise_loss.item()
        total_interv_loss_accum += total_intervened_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'conc': f'{concept_loss.item() if not torch.isnan(concept_loss) else 0.0:.4f}',
            'noise': f'{noise_loss.item():.4f}',
            'interv': f'{total_intervened_loss.item():.4f}'
        })

        # Log to tensorboard
        if batch_idx % config["train_config"]["log_interval"] == 0:
            global_step = epoch * steps_per_epoch + batch_idx
            if config["train_config"]["plot_loss"]:
                writer.add_scalar('loss/total_loss', loss.item(), global_step)
                writer.add_scalar('loss/recon_loss', recon_loss.item(), global_step)
                writer.add_scalar('loss/concept_loss', concept_loss.item() if not torch.isnan(concept_loss) else 0.0, global_step)
                writer.add_scalar('loss/noise_loss', noise_loss.item(), global_step)
                writer.add_scalar('loss/interv_loss', total_intervened_loss.item(), global_step)

    # Return average losses
    return {
        'total_loss': total_loss_accum / num_batches if num_batches > 0 else 0.0,
        'recon_loss': total_recon_loss_accum / num_batches if num_batches > 0 else 0.0,
        'concept_loss': total_concept_loss_accum / num_batches if num_batches > 0 else 0.0,
        'noise_loss': total_noise_loss_accum / num_batches if num_batches > 0 else 0.0,
        'interv_loss': total_interv_loss_accum / num_batches if num_batches > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset", default="celebahq", help="benchmark dataset")
    parser.add_argument("-e", "--expt-name", default="cbae_ddpm", help="experiment name")
    parser.add_argument("-t", "--tensorboard-name", default="clipzs", help="tensorboard suffix")
    parser.add_argument("-p", "--pseudo-label", type=str, default='clipzs',
                        help='pseudo-label source: clipzs, supervised, tipzs')
    parser.add_argument("--load-pretrained", action='store_true', default=False,
                        help='load pretrained CB-AE checkpoint')
    parser.add_argument("--pretrained-load-name", type=str, default='',
                        help='filename to load from models/checkpoints/')
    parser.add_argument("--config-file", type=str, default=None,
                        help='path to config file (overrides default)')
    parser.add_argument("--gpu", type=int, default=None,
                        help='GPU ID to use (e.g., 0, 1, 2, 3)')
    args = parser.parse_args()

    # Use custom config file if provided, otherwise use default
    if args.config_file is None:
        args.config_file = f"./config/{args.expt_name}/{args.dataset}.yaml"

    with open(args.config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {args.config_file}")

    # Setup device
    # Force use of specific GPU if requested
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"Forcing use of GPU {args.gpu}")

    use_cuda = config["train_config"]["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Setup directories
    writer = SummaryWriter(f'results/{args.dataset}_{args.expt_name}_{args.tensorboard_name}')
    os.makedirs("models/checkpoints/", exist_ok=True)
    os.makedirs("images/", exist_ok=True)

    if config["train_config"]["save_model"]:
        save_model_name = f"{config['dataset']['name']}_{args.expt_name}_{args.tensorboard_name}"

    # Build model
    model = cbAE_DDPM_Trainable(config)
    model.to(device)

    if args.load_pretrained:
        print(f'Loading pretrained CB-AE from models/checkpoints/{args.pretrained_load_name}')
        model.cbae.load_state_dict(torch.load(f'models/checkpoints/{args.pretrained_load_name}'))

    # Setup dataset and concepts
    # Dynamic concept loading: prioritize config over hardcoded values
    use_real_labels = config["dataset"].get("use_real_labels", False)

    # Check if concepts are defined in config (works for both real labels and pseudo-labels)
    if "concepts" in config["model"] and "concept_names" in config["model"]["concepts"]:
        # Build set_of_classes from config dynamically
        concept_names = config["model"]["concepts"]["concept_names"]
        print(f"Using concept names from config: {concept_names}")

        # Create binary concept classes based on concept names
        set_of_classes = []
        for concept_name in concept_names:
            # Format: ['NOT concept', 'concept']
            formatted_name = concept_name.replace('_', ' ').title()
            set_of_classes.append([f'NOT {formatted_name}', formatted_name])

        clsf_model_type = 'rn18'  # Default for CelebA-based datasets
        print(f"Dynamically loaded {len(set_of_classes)} concepts from config")

    elif args.dataset == 'celebahq':
        set_of_classes = [
            ['NOT Attractive', 'Attractive'],
            ['NO Lipstick', 'Wearing Lipstick'],
            ['Mouth Closed', 'Mouth Slightly Open'],
            ['NOT Smiling', 'Smiling'],
            ['Low Cheekbones', 'High Cheekbones'],
            ['NO Makeup', 'Heavy Makeup'],
            ['Female', 'Male'],
            ['Straight Eyebrows', 'Arched Eyebrows']
        ]
        clsf_model_type = 'rn18'
    elif args.dataset in ['cub', 'cub64']:
        set_of_classes = [
            ['Large size', 'Small size 5 to 9 inches'],
            ['NOT perching like shape', 'Perching like shape'],
            ['NOT solid breast pattern', 'Solid breast pattern'],
            ['NOT black bill color', 'Black bill color'],
            ['Bill length longer than head', 'Bill length shorter than head'],
            ['NOT black wing color', 'Black wing color'],
            ['NOT solid belly pattern', 'Solid belly pattern'],
            ['NOT All purpose bill shape', 'All purpose bill shape'],
            ['NOT black upperparts color', 'Black upperparts color'],
            ['NOT white underparts color', 'White underparts color'],
        ]
        clsf_model_type = 'rn50'
    elif args.dataset == 'celeba64':
        set_of_classes = [
            ['NOT Attractive', 'Attractive'],
            ['NO Lipstick', 'Wearing Lipstick'],
            ['Mouth Closed', 'Mouth Slightly Open'],
            ['NOT Smiling', 'Smiling'],
            ['Low Cheekbones', 'High Cheekbones'],
            ['NO Makeup', 'Heavy Makeup'],
            ['Female', 'Male'],
            ['Straight Eyebrows', 'Arched Eyebrows']
        ]
        clsf_model_type = 'rn18'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Setup pseudo-labeler
    if args.pseudo_label == 'clipzs':
        print('Using CLIP zero-shot for pseudo-labels')
        pseudo_labeler = clip_pseudolabeler.CLIP_PseudoLabeler(set_of_classes, device)
    elif args.pseudo_label == 'supervised':
        print('Using supervised model for pseudo-labels')
        pseudo_labeler = clip_pseudolabeler.Sup_PseudoLabeler(
            set_of_classes, device, dataset=args.dataset, model_type=clsf_model_type
        )
    elif args.pseudo_label == 'tipzs':
        print('Using TIP Adapter for pseudo-labels')
        pseudo_labeler = clip_pseudolabeler.TIPAda_PseudoLabeler(
            set_of_classes, device, dataset=args.dataset
        )
    else:
        raise ValueError(f"Unknown pseudo-label method: {args.pseudo_label}")

    # Setup noise scheduler for forward diffusion
    noise_scheduler = DDPMNoiseSchedulerHelper(num_train_timesteps=1000)
    noise_scheduler.to(device)

    # Setup optimizers
    # Parse betas - support both list [0.5, 0.99] and string "(.5, .99)"
    betas_config = config["train_config"]["betas"]
    if isinstance(betas_config, list):
        betas = tuple(betas_config)
    elif isinstance(betas_config, str):
        betas = eval(betas_config)
    else:
        betas = betas_config  # Already a tuple

    optimizer = torch.optim.Adam(
        model.cbae.parameters(),
        lr=config["train_config"]["recon_lr"],
        betas=betas
    )
    optimizer_interv = torch.optim.Adam(
        model.cbae.parameters(),
        lr=config["train_config"]["conc_lr"],
        betas=betas
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    train_loader, test_loader = get_dataset(config)

    # If subset_size is specified, limit the dataset
    if config["dataset"].get("subset_size"):
        subset_size = config["dataset"]["subset_size"]
        print(f"Using subset of {subset_size} images for testing")
        from torch.utils.data import Subset
        import random

        # Get the underlying dataset from the dataloader
        full_dataset = train_loader.dataset
        indices = random.sample(range(len(full_dataset)), min(subset_size, len(full_dataset)))
        subset_dataset = Subset(full_dataset, indices)

        dataloader = DataLoader(
            subset_dataset,
            batch_size=config["dataset"]["batch_size"],
            shuffle=True,
            num_workers=config["dataset"].get("num_workers", 4),
            drop_last=True
        )
    else:
        dataloader = train_loader

    print(f"Dataset size: {len(dataloader.dataset)}, batches per epoch: {len(dataloader)}")

    # Training loop
    for epoch in range(config["train_config"]["epochs"]):
        start_time = time.time()

        losses = train_epoch(
            model, dataloader, noise_scheduler, pseudo_labeler,
            optimizer, optimizer_interv, config, device,
            epoch, writer, set_of_classes, args
        )

        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch}/{config['train_config']['epochs']} - "
              f"Time: {elapsed:.2f}s")
        print(f"  Total Loss: {losses['total_loss']:.4f}")
        print(f"  Recon Loss: {losses['recon_loss']:.4f}")
        print(f"  Concept Loss: {losses['concept_loss']:.4f}")
        print(f"  Noise Loss: {losses['noise_loss']:.4f}")
        print(f"  Interv Loss: {losses['interv_loss']:.4f}")

        # Save checkpoint
        if config["train_config"]["save_model"]:
            torch.save(
                model.cbae.state_dict(),
                f"models/checkpoints/{save_model_name}_cbae.pt"
            )
            print(f"Saved checkpoint to models/checkpoints/{save_model_name}_cbae.pt")

    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    main()
