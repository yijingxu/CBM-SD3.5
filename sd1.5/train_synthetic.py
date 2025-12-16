"""
Training script for CB-AE with Stable Diffusion 1.5 using SYNTHETIC data.

This script follows the paper's original approach:
1. Generate face images using SD1.5 with random prompts
2. Use CLIP pseudo-labeler to get concept labels for generated images
3. Train CB-AE on these synthetic image-label pairs

This eliminates the need for any real labeled dataset (like CelebA).

Key advantages:
- No need for real data or manual labels
- CB-AE learns from the same distribution it will be used on
- Can generate unlimited training data
- Works with any text prompts at inference time
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import random
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.cbae_sd15 import SD15WithCBAE, get_concept_index
from models.pseudo_labeler import CLIP_PseudoLabeler

import wandb


# ============================================================
# PROMPT TEMPLATES FOR SYNTHETIC DATA GENERATION
# ============================================================

# Base templates for generating diverse face images
FACE_PROMPT_TEMPLATES = [
    "a photo of a {attributes} person",
    "a portrait of a {attributes} person",
    "a close-up photo of a {attributes} person",
    "a professional photo of a {attributes} person",
    "a high quality photo of a {attributes} person",
    "a photo of a {attributes} individual",
    "a headshot of a {attributes} person",
]

# UNDERSPECIFIED templates - these don't mention specific attributes
# The model must learn to predict attributes from visual features alone
UNDERSPECIFIED_TEMPLATES = [
    "a photo of a person",
    "a portrait photo",
    "a headshot",
    "a face photo",
    "a professional headshot",
    "a photo of someone",
    "a portrait of a person",
    "a close-up of a face",
    "a photo of a human face",
    "a realistic portrait",
    "photograph of a person looking at the camera",
    "a high quality face photo",
]

# Attribute variations for diverse generation
ATTRIBUTE_VARIATIONS = {
    "smiling": ["smiling", "happy", "grinning", "cheerful", "joyful"],
    "not_smiling": ["serious", "neutral expression", "stoic", "stern", "solemn"],
    "male": ["male", "man", "gentleman"],
    "female": ["female", "woman", "lady"],
    "young": ["young", "youthful"],
    "old": ["elderly", "old", "aged", "senior"],
    "eyeglasses": ["wearing glasses", "with eyeglasses", "bespectacled"],
    "no_eyeglasses": ["without glasses"],
    "heavy_makeup": ["with heavy makeup", "with glamorous makeup"],
    "no_makeup": ["with natural look", "without makeup"],
    "bald": ["bald", "with no hair"],
    "bangs": ["with bangs"],
    "black_hair": ["with black hair"],
    "blond_hair": ["with blonde hair", "with blond hair"],
    "brown_hair": ["with brown hair"],
    "gray_hair": ["with gray hair", "with grey hair"],
    "wavy_hair": ["with wavy hair", "with curly hair"],
    "straight_hair": ["with straight hair"],
    "beard": ["with beard", "bearded"],
    "mustache": ["with mustache"],
    "goatee": ["with goatee"],
    "wearing_hat": ["wearing a hat", "with a hat"],
    "wearing_earrings": ["wearing earrings"],
    "attractive": ["attractive", "good-looking", "beautiful", "handsome"],
}

# Negative prompt to improve face quality
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
    "bad proportions, extra limbs, cloned face, disfigured, "
    "out of frame, watermark, signature, text"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CB-AE for SD1.5 using synthetic generated data"
    )
    
    # Config file argument (takes priority)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. Command line args override config file values."
    )
    
    # Model arguments
    parser.add_argument(
        "--sd-model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion model ID"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Image size for generation (512 recommended for SD1.5)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1024,
        help="Hidden dimension for CB-AE"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of layers in CB-AE encoder/decoder"
    )
    parser.add_argument(
        "--unsupervised-dim",
        type=int,
        default=64,
        help="Dimension of unsupervised concept embedding"
    )
    parser.add_argument(
        "--max-timestep",
        type=int,
        default=400,
        help="Maximum timestep for training (concepts visible at low noise)"
    )
    
    # Concept/Attribute arguments
    parser.add_argument(
        "--attributes",
        type=str,
        nargs="+",
        default=["Smiling", "Male", "Eyeglasses", "Young", "Bald", 
                 "Bangs", "Heavy_Makeup", "Pale_Skin"],
        help="Attributes to predict"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (smaller due to generation overhead)"
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=500,
        help="Number of training steps per epoch"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lr-interv",
        type=float,
        default=1e-4,
        help="Learning rate for intervention training"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=20,
        help="Number of denoising steps for image generation"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--underspecified-ratio",
        type=float,
        default=0.3,
        help="Ratio of prompts that don't specify attributes (0.0-1.0). "
             "Higher values train the model to predict attributes not in the prompt."
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt for better image quality. "
             "Example: 'blurry, low quality, distorted, deformed'. "
             "If not specified, uses standard unconditional embedding (recommended)."
    )
    
    # Logging arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cbae-sd15-synthetic",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Wandb run name"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Logging interval (steps)"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval (epochs)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Checkpoint save interval (epochs)"
    )
    parser.add_argument(
        "--log-images",
        action="store_true",
        default=True,
        help="Log generated images to wandb"
    )
    parser.add_argument(
        "--log-images-interval",
        type=int,
        default=100,
        help="How often to log images (steps)"
    )
    
    # Other arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=False,
        help="Use mixed precision training"
    )
    
    # First parse to get config file path
    args, remaining = parser.parse_known_args()
    
    # Load config file if specified
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Convert config keys from snake_case to match argparse
        # e.g., sd_model -> sd-model for argparse, but we store as sd_model
        for key, value in config.items():
            # Convert underscores to match argparse internal storage
            attr_name = key.replace('-', '_')
            if hasattr(args, attr_name):
                setattr(args, attr_name, value)
            else:
                # Handle special cases like wandb_project -> wandb_project
                setattr(args, attr_name, value)
    
    # Re-parse to let command line args override config
    args = parser.parse_args()
    
    # Load config again and set defaults, but command line takes priority
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            attr_name = key.replace('-', '_')
            # Only set if the argument wasn't explicitly provided on command line
            # We detect this by checking if it's still the parser default
            if not any(f'--{key}' in arg or f'--{key.replace("_", "-")}' in arg for arg in sys.argv):
                setattr(args, attr_name, value)
    
    return args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_concept_classes(attributes: List[str]) -> List[List[str]]:
    """
    Create concept class names for CLIP pseudo-labeling.
    
    Args:
        attributes: List of attribute names
        
    Returns:
        List of [negative_class, positive_class] for each attribute
    """
    concept_classes = []
    for attr in attributes:
        formatted = attr.replace('_', ' ').lower()
        concept_classes.append([f"not {formatted}", formatted])
    return concept_classes


def generate_random_prompt(underspecified_ratio: float = 0.3) -> Tuple[str, bool]:
    """
    Generate a random prompt for face image generation.
    
    Args:
        underspecified_ratio: Probability of generating an underspecified prompt
                              (one that doesn't mention specific attributes)
    
    Returns:
        prompt: A text prompt describing a person
        is_underspecified: Whether this is an underspecified prompt
    """
    # With some probability, use an underspecified prompt
    # This is CRUCIAL for learning to predict attributes not in the prompt
    if random.random() < underspecified_ratio:
        return random.choice(UNDERSPECIFIED_TEMPLATES), True
    
    # Otherwise, generate a prompt with random attributes
    attributes = []
    
    # Gender (always include one)
    if random.random() > 0.5:
        attributes.append(random.choice(ATTRIBUTE_VARIATIONS["male"]))
    else:
        attributes.append(random.choice(ATTRIBUTE_VARIATIONS["female"]))
    
    # Age
    if random.random() > 0.7:
        if random.random() > 0.5:
            attributes.append(random.choice(ATTRIBUTE_VARIATIONS["young"]))
        else:
            attributes.append(random.choice(ATTRIBUTE_VARIATIONS["old"]))
    
    # Expression
    if random.random() > 0.5:
        attributes.append(random.choice(ATTRIBUTE_VARIATIONS["smiling"]))
    elif random.random() > 0.7:
        attributes.append(random.choice(ATTRIBUTE_VARIATIONS["not_smiling"]))
    
    # Glasses
    if random.random() > 0.7:
        attributes.append(random.choice(ATTRIBUTE_VARIATIONS["eyeglasses"]))
    
    # Hair
    if random.random() > 0.6:
        hair_colors = ["black_hair", "blond_hair", "brown_hair", "gray_hair"]
        hair_color = random.choice(hair_colors)
        attributes.append(random.choice(ATTRIBUTE_VARIATIONS[hair_color]))
    
    if random.random() > 0.7:
        hair_styles = ["wavy_hair", "straight_hair", "bangs", "bald"]
        hair_style = random.choice(hair_styles)
        if hair_style in ATTRIBUTE_VARIATIONS:
            attributes.append(random.choice(ATTRIBUTE_VARIATIONS[hair_style]))
    
    # Makeup
    if random.random() > 0.8:
        attributes.append(random.choice(ATTRIBUTE_VARIATIONS["heavy_makeup"]))
    
    # Facial hair (for variety)
    if random.random() > 0.85:
        facial_hair = random.choice(["beard", "mustache", "goatee"])
        attributes.append(random.choice(ATTRIBUTE_VARIATIONS[facial_hair]))
    
    # Accessories
    if random.random() > 0.9:
        attributes.append(random.choice(ATTRIBUTE_VARIATIONS["wearing_hat"]))
    if random.random() > 0.85:
        attributes.append(random.choice(ATTRIBUTE_VARIATIONS["wearing_earrings"]))
    
    # Build the prompt
    random.shuffle(attributes)
    attr_string = ", ".join(attributes) if attributes else "person"
    template = random.choice(FACE_PROMPT_TEMPLATES)
    prompt = template.format(attributes=attr_string)
    
    return prompt, False


def generate_batch_prompts(batch_size: int, underspecified_ratio: float = 0.3) -> Tuple[List[str], List[bool]]:
    """
    Generate a batch of random prompts.
    
    Returns:
        prompts: List of prompt strings
        is_underspecified: List of booleans indicating if each prompt is underspecified
    """
    results = [generate_random_prompt(underspecified_ratio) for _ in range(batch_size)]
    prompts = [r[0] for r in results]
    is_underspecified = [r[1] for r in results]
    return prompts, is_underspecified


@torch.no_grad()
def generate_images_for_training(
    model: SD15WithCBAE,
    prompts: List[str],
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    device: str = "cuda",
    negative_prompt: str = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate images using SD1.5 for training.
    
    This function generates images AND returns intermediate latents/timesteps
    needed for CB-AE training.
    
    Args:
        model: SD15WithCBAE model
        prompts: List of text prompts
        num_inference_steps: Denoising steps
        guidance_scale: CFG scale
        device: Device
        negative_prompt: Optional negative prompt for better image quality.
                        If None, uses empty string (standard unconditional).
                        Example: "blurry, low quality, distorted"
        
    Returns:
        images: Generated images [B, 3, H, W] in range [-1, 1]
        training_latents: Noisy latents at a random training timestep
        training_timesteps: The timesteps for training
        text_embeddings: Text embeddings for the prompts
    """
    from diffusers import DDIMScheduler
    
    batch_size = len(prompts)
    
    # Setup scheduler for generation
    scheduler = DDIMScheduler.from_pretrained(
        model.pipe.scheduler.config._name_or_path if hasattr(model.pipe.scheduler.config, '_name_or_path') 
        else "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )
    scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Encode prompts
    text_embeddings = model.encode_text(prompts)
    
    # Encode negative/unconditional prompt for CFG
    # Using null (empty) embedding is standard and works well
    # Negative prompts are optional - they can improve quality but aren't required
    if negative_prompt is not None and negative_prompt.strip():
        uncond_embeddings = model.encode_text([negative_prompt] * batch_size)
    else:
        uncond_embeddings = model.null_text_embedding.repeat(batch_size, 1, 1)
    
    # Concatenate for CFG
    text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings])
    
    # Initialize random latents
    latent_shape = (batch_size, 4, model.img_size // 8, model.img_size // 8)
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32)
    latents = latents * scheduler.init_noise_sigma
    
    # Denoising loop
    for t in scheduler.timesteps:
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        timestep_batch = torch.tensor([t] * batch_size * 2, device=device)
        
        # Run full UNet (not through CB-AE for generation)
        noise_pred = model.unet(
            latent_model_input,
            timestep_batch,
            encoder_hidden_states=text_embeddings_cfg,
        ).sample
        
        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Denoise step
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode final latents to images
    images = model.decode_latents(latents)
    
    # Now prepare training data:
    # Sample random timesteps in training range and add noise
    training_timesteps = torch.randint(
        0, model.max_timestep + 1, (batch_size,), device=device
    ).long()
    
    # Re-encode the clean generated images
    clean_latents = model.encode_images(images)
    
    # Add noise for training
    noise = torch.randn_like(clean_latents)
    training_latents = model.add_noise(clean_latents, noise, training_timesteps)
    
    return images, clean_latents, training_latents, training_timesteps, text_embeddings, noise


def train_step(
    model: SD15WithCBAE,
    pseudo_labeler: CLIP_PseudoLabeler,
    optimizer: torch.optim.Optimizer,
    optimizer_interv: torch.optim.Optimizer,
    args,
    device: str,
) -> Dict[str, float]:
    """
    Perform one training step with synthetic data.
    
    Returns:
        Dictionary of losses
    """
    model.cbae.train()
    
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    batch_size = args.batch_size
    underspecified_ratio = getattr(args, 'underspecified_ratio', 0.3)
    
    # =====================
    # Step 1: Generate synthetic images
    # =====================
    prompts, is_underspecified = generate_batch_prompts(batch_size, underspecified_ratio)
    
    with torch.no_grad():
        (
            images,           # Generated images [-1, 1]
            clean_latents,    # Clean VAE latents
            noisy_latents,    # Noisy latents for training
            timesteps,        # Training timesteps
            text_embeddings,  # Text embeddings
            noise,            # The noise that was added
        ) = generate_images_for_training(
            model=model,
            prompts=prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            device=device,
            negative_prompt=getattr(args, 'negative_prompt', None),
        )
        
        # Get pseudo-labels from generated images
        # IMPORTANT: Labels come from CLIP looking at the IMAGE, not from the prompt!
        # This is how we learn to predict attributes not mentioned in the prompt.
        images_normalized = (images + 1) / 2  # [0, 1] for CLIP
        images_normalized = torch.clamp(images_normalized, 0, 1)
        
        probs_list, labels_list = pseudo_labeler.get_pseudo_labels(
            images_normalized, return_prob=True
        )
        
        # Stack labels into tensor [B, n_concepts]
        labels = torch.stack(labels_list, dim=1)
    
    # =====================
    # Phase 1: Reconstruction + Concept + Noise Loss
    # =====================
    optimizer.zero_grad()
    
    # Get bottleneck features with text conditioning
    with torch.no_grad():
        bottleneck, cache = model.get_unet_bottleneck(
            noisy_latents, timesteps, text_embeddings
        )
    
    # Apply CB-AE
    concepts, bottleneck_recon = model.cbae(bottleneck)
    
    # Predict noise from reconstructed bottleneck
    with torch.no_grad():
        noise_pred = model.run_unet_decoder(bottleneck_recon, cache)
    
    # Loss 1: Bottleneck reconstruction
    recon_loss = mse_loss(bottleneck_recon, bottleneck)
    
    # Loss 2: Concept alignment with pseudo-labels
    concept_loss = torch.tensor(0.0, device=device)
    for c in range(model.n_concepts):
        start, end = get_concept_index(model, c)
        c_logits = concepts[:, start:end]
        c_labels = labels[:, c]
        concept_loss = concept_loss + ce_loss(c_logits, c_labels)
    concept_loss = concept_loss / model.n_concepts
    
    # Loss 3: Noise prediction
    noise_loss = mse_loss(noise_pred, noise)
    
    # Total loss
    loss = recon_loss + concept_loss + noise_loss
    
    loss.backward()
    optimizer.step()
    
    # =====================
    # Phase 2: Intervention Training
    # =====================
    optimizer_interv.zero_grad()
    
    # Sample new timesteps
    timesteps_interv = torch.randint(
        0, args.max_timestep + 1, (batch_size,), device=device
    ).long()
    
    noise_interv = torch.randn_like(clean_latents)
    noisy_latents_interv = model.add_noise(clean_latents, noise_interv, timesteps_interv)
    
    with torch.no_grad():
        bottleneck_interv, cache_interv = model.get_unet_bottleneck(
            noisy_latents_interv, timesteps_interv, text_embeddings
        )
    
    # Random concept to intervene
    rand_concept = torch.randint(0, model.n_concepts, (1,)).item()
    concept_value = torch.randint(0, 2, (1,)).item()
    
    with torch.no_grad():
        concepts_interv = model.cbae.encode(bottleneck_interv)
        
        # Swap logits for intervention
        start_idx, end_idx = get_concept_index(model, rand_concept)
        intervened_concepts = concepts_interv.clone()
        curr_c = intervened_concepts[:, start_idx:end_idx]
        
        old_vals = curr_c[:, concept_value].clone()
        max_val, max_ind = torch.max(curr_c, dim=1)
        curr_c[:, concept_value] = max_val
        for i, (curr_ind, curr_old) in enumerate(zip(max_ind, old_vals)):
            curr_c[i, curr_ind] = curr_old
        
        intervened_concepts[:, start_idx:end_idx] = curr_c
        intervened_concepts = intervened_concepts.detach()
        
        # Target labels
        intervened_target_labels = labels.clone()
        intervened_target_labels[:, rand_concept] = concept_value
    
    # Decode intervened concepts
    intervened_bottleneck = model.cbae.decode(intervened_concepts)
    
    # Get noise prediction for intervened bottleneck
    with torch.no_grad():
        noise_pred_interv = model.run_unet_decoder(intervened_bottleneck, cache_interv)
    
    # Estimate clean image from intervened prediction
    with torch.no_grad():
        pred_original_latent = model.predict_x0_from_noise(
            noisy_latents_interv, noise_pred_interv, timesteps_interv
        )
        pred_original_latent = torch.clamp(pred_original_latent, -1.0, 1.0)
        
        # Decode to pixel space
        intervened_images = model.decode_latents(pred_original_latent)
        intervened_images_normalized = (intervened_images + 1) / 2
        intervened_images_normalized = torch.clamp(intervened_images_normalized, 0, 1)
    
    # L_i1: External consistency - verify with pseudo-labeler
    pred_logits = pseudo_labeler.get_soft_pseudo_labels(intervened_images_normalized)
    
    li1_loss = torch.tensor(0.0, device=device)
    for c in range(model.n_concepts):
        c_logits = pred_logits[c]
        c_target = intervened_target_labels[:, c]
        li1_loss = li1_loss + ce_loss(c_logits, c_target)
    li1_loss = li1_loss / model.n_concepts
    
    # L_i2: Internal consistency - re-encode
    recon_interv_concepts = model.cbae.encode(intervened_bottleneck)
    
    li2_loss = torch.tensor(0.0, device=device)
    for c in range(model.n_concepts):
        start, end = get_concept_index(model, c)
        c_logits = recon_interv_concepts[:, start:end]
        c_target = intervened_target_labels[:, c]
        li2_loss = li2_loss + ce_loss(c_logits, c_target)
    li2_loss = li2_loss / model.n_concepts
    
    # Total intervention loss
    interv_loss = li1_loss + li2_loss
    
    if not torch.isnan(interv_loss) and interv_loss.item() > 0:
        interv_loss.backward()
        optimizer_interv.step()
    else:
        interv_loss = torch.tensor(0.0, device=device)
        li1_loss = torch.tensor(0.0, device=device)
        li2_loss = torch.tensor(0.0, device=device)
    
    return {
        "total_loss": loss.item(),
        "recon_loss": recon_loss.item(),
        "concept_loss": concept_loss.item(),
        "noise_loss": noise_loss.item(),
        "interv_loss": interv_loss.item(),
        "li1_loss": li1_loss.item(),
        "li2_loss": li2_loss.item(),
        "images": images,  # For logging
        "prompts": prompts,
        "labels": labels,
        "is_underspecified": is_underspecified,
    }


@torch.no_grad()
def evaluate(
    model: SD15WithCBAE,
    pseudo_labeler: CLIP_PseudoLabeler,
    args,
    device: str,
    num_samples: int = 100,
) -> Dict[str, float]:
    """
    Evaluate concept prediction accuracy on synthetic data.
    
    Tests on BOTH specified and underspecified prompts to ensure
    the model can predict attributes not mentioned in the prompt.
    """
    model.cbae.eval()
    
    n_concepts = model.n_concepts
    
    # Track accuracy separately for specified and underspecified prompts
    correct_all = torch.zeros(n_concepts, device=device)
    total_all = torch.zeros(n_concepts, device=device)
    correct_underspec = torch.zeros(n_concepts, device=device)
    total_underspec = torch.zeros(n_concepts, device=device)
    
    num_batches = num_samples // args.batch_size
    
    for _ in tqdm(range(num_batches), desc="Evaluating"):
        # Use 50% underspecified for evaluation to properly test generalization
        prompts, is_underspecified = generate_batch_prompts(args.batch_size, underspecified_ratio=0.5)
        
        # Generate images
        (
            images, clean_latents, _, _, text_embeddings, _
        ) = generate_images_for_training(
            model=model,
            prompts=prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            device=device,
        )
        
        # Get pseudo-labels
        images_normalized = (images + 1) / 2
        _, labels_list = pseudo_labeler.get_pseudo_labels(images_normalized, return_prob=True)
        labels = torch.stack(labels_list, dim=1)
        
        # Predict concepts at timestep 0
        timesteps = torch.zeros(args.batch_size, device=device, dtype=torch.long)
        bottleneck, _ = model.get_unet_bottleneck(clean_latents, timesteps, text_embeddings)
        concepts = model.cbae.encode(bottleneck)
        
        # Compute accuracy
        for c in range(n_concepts):
            start, end = get_concept_index(model, c)
            c_logits = concepts[:, start:end]
            c_preds = c_logits.argmax(dim=-1)
            c_labels = labels[:, c]
            
            correct_mask = (c_preds == c_labels)
            
            # Overall accuracy
            correct_all[c] += correct_mask.sum()
            total_all[c] += c_labels.shape[0]
            
            # Accuracy on underspecified prompts only
            for i, is_under in enumerate(is_underspecified):
                if is_under:
                    correct_underspec[c] += correct_mask[i].float()
                    total_underspec[c] += 1
    
    model.cbae.train()
    
    # Compute accuracies
    per_concept_acc = (correct_all / total_all.clamp(min=1)).cpu().numpy()
    overall_acc = (correct_all.sum() / total_all.sum()).item()
    
    per_concept_acc_underspec = (correct_underspec / total_underspec.clamp(min=1)).cpu().numpy()
    overall_acc_underspec = (correct_underspec.sum() / total_underspec.sum()).item() if total_underspec.sum() > 0 else 0.0
    
    return {
        "overall": overall_acc,
        "per_concept": per_concept_acc,
        "overall_underspecified": overall_acc_underspec,
        "per_concept_underspecified": per_concept_acc_underspec,
    }
    
    per_concept_acc = (correct / total.clamp(min=1)).cpu().numpy()
    overall_acc = (correct.sum() / total.sum()).item()
    
    return {
        "overall": overall_acc,
        "per_concept": per_concept_acc,
    }


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    print(f"Training CB-AE with SYNTHETIC data generation")
    print(f"Attributes: {args.attributes}")
    
    # Initialize wandb
    wandb_name = args.wandb_name or f"cbae-sd15-synthetic-{len(args.attributes)}attr"
    wandb.init(
        project=args.wandb_project,
        name=wandb_name,
        config=vars(args),
    )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    print("Creating SD1.5 with CB-AE...")
    concept_dims = [2] * len(args.attributes)
    
    model = SD15WithCBAE(
        model_id=args.sd_model,
        n_concepts=len(args.attributes),
        concept_dims=concept_dims,
        unsupervised_dim=args.unsupervised_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        img_size=args.img_size,
        max_timestep=args.max_timestep,
        device=device,
    )
    model.to(device)
    
    # Create pseudo-labeler
    print("Creating CLIP pseudo-labeler...")
    concept_classes = get_concept_classes(args.attributes)
    pseudo_labeler = CLIP_PseudoLabeler(concept_classes, device=device)
    
    # Create optimizers
    optimizer = torch.optim.Adam(
        model.cbae.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    
    optimizer_interv = torch.optim.Adam(
        model.cbae.parameters(),
        lr=args.lr_interv,
        betas=(0.5, 0.999),
    )
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.cbae.load_state_dict(checkpoint["cbae_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_interv.load_state_dict(checkpoint["optimizer_interv_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    
    # Log model info
    n_params = sum(p.numel() for p in model.cbae.parameters())
    print(f"CB-AE parameters: {n_params:,}")
    wandb.log({"model/cbae_params": n_params})
    
    # Training loop
    best_acc = 0.0
    global_step = 0
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Accumulators
        epoch_losses = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "concept_loss": 0.0,
            "noise_loss": 0.0,
            "interv_loss": 0.0,
        }
        
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}")
        
        for step in pbar:
            global_step += 1
            
            # Train step
            losses = train_step(
                model=model,
                pseudo_labeler=pseudo_labeler,
                optimizer=optimizer,
                optimizer_interv=optimizer_interv,
                args=args,
                device=device,
            )
            
            # Accumulate
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total_loss']:.4f}",
                "recon": f"{losses['recon_loss']:.4f}",
                "concept": f"{losses['concept_loss']:.4f}",
                "Li1": f"{losses['li1_loss']:.4f}",
            })
            
            # Log to wandb
            if step % args.log_interval == 0:
                wandb.log({
                    "train/total_loss": losses["total_loss"],
                    "train/recon_loss": losses["recon_loss"],
                    "train/concept_loss": losses["concept_loss"],
                    "train/noise_loss": losses["noise_loss"],
                    "train/interv_loss": losses["interv_loss"],
                    "train/Li1_external": losses["li1_loss"],
                    "train/Li2_cyclic": losses["li2_loss"],
                    "train/step": global_step,
                })
            
            # Log images
            if args.log_images and step % args.log_images_interval == 0:
                images = losses["images"]
                prompts = losses["prompts"]
                labels = losses["labels"]
                
                # Log first image with its prompt and predicted labels
                img = (images[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
                img = np.clip(img, 0, 1)
                
                caption = f"Prompt: {prompts[0]}\nLabels: {labels[0].cpu().tolist()}"
                wandb.log({
                    "images/generated": wandb.Image(img, caption=caption),
                    "train/step": global_step,
                })
        
        elapsed = time.time() - start_time
        
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= args.steps_per_epoch
        
        print(f"\nEpoch {epoch}/{args.epochs} - Time: {elapsed:.2f}s")
        print(f"  Total Loss: {epoch_losses['total_loss']:.4f}")
        print(f"  Recon Loss: {epoch_losses['recon_loss']:.4f}")
        print(f"  Concept Loss: {epoch_losses['concept_loss']:.4f}")
        print(f"  Noise Loss: {epoch_losses['noise_loss']:.4f}")
        print(f"  Interv Loss: {epoch_losses['interv_loss']:.4f}")
        
        wandb.log({
            "epoch/total_loss": epoch_losses["total_loss"],
            "epoch/recon_loss": epoch_losses["recon_loss"],
            "epoch/concept_loss": epoch_losses["concept_loss"],
            "epoch/noise_loss": epoch_losses["noise_loss"],
            "epoch/interv_loss": epoch_losses["interv_loss"],
            "epoch/time": elapsed,
            "epoch": epoch,
        })
        
        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            print("Evaluating...")
            acc_results = evaluate(
                model=model,
                pseudo_labeler=pseudo_labeler,
                args=args,
                device=device,
                num_samples=100,
            )
            
            print(f"  Overall Accuracy: {acc_results['overall']:.4f}")
            print(f"  Overall Accuracy (underspecified prompts): {acc_results['overall_underspecified']:.4f}")
            print(f"  Per-concept accuracy:")
            for i, acc in enumerate(acc_results["per_concept"]):
                acc_under = acc_results["per_concept_underspecified"][i]
                print(f"    {args.attributes[i]}: {acc:.4f} (underspec: {acc_under:.4f})")
            
            wandb.log({
                "eval/overall_accuracy": acc_results["overall"],
                "eval/overall_accuracy_underspecified": acc_results["overall_underspecified"],
                "epoch": epoch,
            })
            for i, acc in enumerate(acc_results["per_concept"]):
                wandb.log({
                    f"eval/accuracy_{args.attributes[i]}": acc,
                    f"eval/accuracy_{args.attributes[i]}_underspecified": acc_results["per_concept_underspecified"][i],
                    "epoch": epoch,
                })
            
            # Save best
            if acc_results["overall"] > best_acc:
                best_acc = acc_results["overall"]
                torch.save(
                    {
                        "epoch": epoch,
                        "cbae_state_dict": model.cbae.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "optimizer_interv_state_dict": optimizer_interv.state_dict(),
                        "accuracy": best_acc,
                        "attributes": args.attributes,
                    },
                    os.path.join(args.checkpoint_dir, "best_cbae.pt"),
                )
                print(f"  Saved best model (accuracy: {best_acc:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "cbae_state_dict": model.cbae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "optimizer_interv_state_dict": optimizer_interv.state_dict(),
                    "attributes": args.attributes,
                },
                os.path.join(args.checkpoint_dir, f"cbae_epoch_{epoch}.pt"),
            )
            print(f"  Saved checkpoint")
    
    # Final save
    torch.save(
        {
            "epoch": args.epochs - 1,
            "cbae_state_dict": model.cbae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_interv_state_dict": optimizer_interv.state_dict(),
            "attributes": args.attributes,
        },
        os.path.join(args.checkpoint_dir, "cbae_final.pt"),
    )
    
    wandb.finish()
    print("Training completed!")


if __name__ == "__main__":
    main()