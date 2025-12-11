"""
Training script for CB-AE with DDPM models - Custom Dataset Version.

This script implements the DDPM-specific training procedure with custom dataset loading:
1. Load clean images x_0 from custom annotation files
2. Sample early timesteps t <= max_timestep
3. Add noise to get x_t
4. Extract noisy latent w_t from UNet encoder
5. Apply CB-AE to get reconstructed latent
6. Compute losses:
   - Latent reconstruction loss
   - Concept alignment loss (using real labels from annotations)
   - DDPM noise prediction loss
   - Intervention losses
"""

import os
import sys
sys.path.append('.')
from utils.utils import create_image_grid, get_concept_index
import argparse
import numpy as np
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import torch.nn.functional as F
from models.cbae_ddpm import cbAE_DDPM_Trainable, DDPMNoiseSchedulerHelper
from models import clip_pseudolabeler
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from diffusers import DDIMScheduler
from PIL import Image
from torchvision import transforms


class CelebAConceptDataset(Dataset):
    """
    Custom Dataset for CelebA-HQ with concept annotations.

    Reads annotation files in format: filename.jpg label1 label2 ...
    Applies transforms to normalize images to [-1, 1].
    Returns (image, labels_tensor).
    """

    def __init__(self, anno_file, data_path, img_size=256, num_concepts=2):
        """
        Args:
            anno_file: Path to annotation txt file
            data_path: Root directory containing images
            img_size: Target image size
            num_concepts: Number of concepts
        """
        self.data_path = data_path
        self.img_size = img_size
        self.num_concepts = num_concepts

        # Define transforms: Resize -> ToTensor -> Normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)  # Maps [0, 1] to [-1, 1]
        ])

        # Load annotations
        self.samples = []
        with open(anno_file, 'r') as f:
            lines = f.readlines()

            # Skip first line (count) and second line (concept names)
            for line in lines[2:]:
                parts = line.strip().split()
                if len(parts) < num_concepts + 1:
                    continue

                filename = parts[0]
                labels = [int(parts[i+1]) for i in range(num_concepts)]
                self.samples.append((filename, labels))

        print(f"Loaded {len(self.samples)} samples from {anno_file}")
        if len(self.samples) > 0:
            print(f"  Sample format: {self.samples[0]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, labels = self.samples[idx]

        # Load image - handle different filename formats
        # Try original filename first
        img_path = os.path.join(self.data_path, filename)
        if not os.path.exists(img_path):
            # Try with 'images/' subdirectory
            img_path = os.path.join(self.data_path, 'images', filename)
        if not os.path.exists(img_path):
            # Try removing leading zeros (e.g., "007405.jpg" -> "7405.jpg")
            basename = os.path.basename(filename)
            name_no_ext, ext = os.path.splitext(basename)
            name_no_zeros = str(int(name_no_ext)) + ext
            img_path = os.path.join(self.data_path, 'images', name_no_zeros)

        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        image = self.transform(image)

        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return image, labels_tensor


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
    Compute concept alignment loss using pseudo-labels or real labels.

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

        if use_pl_thresh and pseudolabel_probs is not None:
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
    args,
    use_real_labels=False
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
        # Get clean images x_0 and labels
        real_labels = None
        if isinstance(batch, (tuple, list)):
            clean_images = batch[0].to(device)
            if len(batch) > 1:
                # Real labels from dataset
                # batch[1] is a tensor of shape [batch_size, num_concepts]
                if isinstance(batch[1], torch.Tensor):
                    batch_labels_tensor = batch[1].to(device)
                    num_concepts = batch_labels_tensor.shape[1]
                    real_labels = []
                    for c_idx in range(num_concepts):
                        real_labels.append(batch_labels_tensor[:, c_idx].long())
                    if batch_idx == 0 and epoch == 0:
                        print(f"[INFO] Using real labels (tensor format): {num_concepts} concepts")
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
        if real_labels is not None and use_real_labels:
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

        # IMPROVED: Cycle through all concepts systematically to ensure balanced intervention
        # This ensures every concept gets equal intervention training
        rand_concept = batch_idx % len(set_of_classes)
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
            if real_labels is not None and use_real_labels:
                # Use real labels as base
                intervened_label = [rl.detach().clone() for rl in real_labels]
                intervened_label[rand_concept] = (
                    torch.ones((batch_size_actual,), device=device) * concept_value
                ).long()
            else:
                # Get fresh pseudo-labels for intervention
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

        # Compute L_i1: external consistency loss
        intervened_pseudo_label_loss = torch.tensor(0.0, device=device)
        for cdx, (curr_logits, actual_pl) in enumerate(zip(pred_logits, intervened_label)):
            curr_loss = ce_loss(curr_logits, actual_pl)
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

    # Load dataset using custom Dataset class
    print(f"Loading dataset: {args.dataset}")

    # Create custom datasets
    if use_real_labels and "train_anno" in config["dataset"]:
        print("Using custom CelebAConceptDataset with annotation files")

        train_anno = config["dataset"]["train_anno"]
        test_anno = config["dataset"].get("test_anno", None)
        data_path = config["dataset"]["data_path"]
        img_size = config["dataset"]["img_size"]
        num_concepts = len(config["model"]["concepts"]["concept_names"])

        # Create train dataset
        train_dataset = CelebAConceptDataset(
            anno_file=train_anno,
            data_path=data_path,
            img_size=img_size,
            num_concepts=num_concepts
        )

        # Create test dataset if annotation file exists
        if test_anno and os.path.exists(test_anno):
            test_dataset = CelebAConceptDataset(
                anno_file=test_anno,
                data_path=data_path,
                img_size=img_size,
                num_concepts=num_concepts
            )
        else:
            test_dataset = None

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["dataset"]["batch_size"],
            shuffle=True,
            num_workers=config["dataset"].get("num_workers", 4),
            drop_last=True
        )

        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=config["dataset"]["test_batch_size"],
                shuffle=False,
                num_workers=config["dataset"].get("num_workers", 4),
                drop_last=False
            )
        else:
            test_loader = None

        print(f"Train dataset: {len(train_dataset)} samples")
        if test_dataset:
            print(f"Test dataset: {len(test_dataset)} samples")
    else:
        # Fall back to default get_dataset
        print("Using default get_dataset function")
        from utils.utils import get_dataset
        train_loader, test_loader = get_dataset(config)
        use_real_labels = False  # No real labels available

    dataloader = train_loader
    print(f"Dataset size: {len(dataloader.dataset)}, batches per epoch: {len(dataloader)}")

    # Training loop
    for epoch in range(config["train_config"]["epochs"]):
        start_time = time.time()

        losses = train_epoch(
            model, dataloader, noise_scheduler, pseudo_labeler,
            optimizer, optimizer_interv, config, device,
            epoch, writer, set_of_classes, args, use_real_labels
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
