# python YijingCode/inference.py

import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import json
import argparse
from PIL import Image
from diffusers import StableDiffusion3Pipeline

from YijingCode.CBmodel import Text_CB_AE, Image_CB_AE
from apps.env_utils import get_env_var

# ==========================================
# Configuration
# ==========================================

def load_config(config_path):
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_cbm_models(config, checkpoint_path, device):
    """Load trained CBM models from checkpoint."""
    # Initialize models with same config as training
    text_model = Text_CB_AE(
        seq_len=config['seq_len'],
        embed_dim=config['embed_dim'],
        concept_dim=config['concept_dim'],
        hidden_dim=config['hidden_dim'],
        use_adversarial=config.get('use_adversarial', False)
    ).to(device)
    
    image_model = Image_CB_AE(
        num_patches=config['num_patches'],
        patch_dim=config['embed_dim'],
        concept_dim=config['concept_dim'],
        use_adversarial=config.get('use_adversarial', False)
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    text_model.load_state_dict(checkpoint['text_model_state_dict'])
    image_model.load_state_dict(checkpoint['image_model_state_dict'])
    
    # Set to eval mode
    text_model.eval()
    image_model.eval()
    
    print(f"Loaded CBM models from {checkpoint_path}")
    print(f"  Text model parameters: {sum(p.numel() for p in text_model.parameters()):,}")
    print(f"  Image model parameters: {sum(p.numel() for p in image_model.parameters()):,}")
    
    return text_model, image_model

# ==========================================
# Concept Target Mapping
# ==========================================

# Concept A: Food (Ice Cream=0.0, Spaghetti=1.0)
# Concept B: Container (Plate=0.0, Cone=1.0)
CONCEPT_MAP_A = {"Ice Cream": 0.0, "Spaghetti": 1.0}
CONCEPT_MAP_B = {"Plate": 0.0, "Cone": 1.0}

def get_target_concept_vector(concept_a_target, concept_b_target):
    """
    Get target concept vector as tensor.
    
    Args:
        concept_a_target: "Ice Cream" or "Spaghetti"
        concept_b_target: "Plate" or "Cone"
    
    Returns:
        torch.Tensor: [concept_a_value, concept_b_value]
    """
    val_a = CONCEPT_MAP_A[concept_a_target]
    val_b = CONCEPT_MAP_B[concept_b_target]
    return torch.tensor([val_a, val_b], dtype=torch.float32)

# ==========================================
# Hook Implementation for CBM Intervention
# ==========================================

class CBMInterventionHook:
    """
    Hook that applies CBM intervention to transformer block outputs.
    
    Intervention Mechanism:
    -----------------------
    The hook is registered on a specific transformer block (e.g., Block 12) and can operate
    in two modes:
    
    1. Single-shot mode: Intervention happens exactly once at a specified timestep (e.g., t=6)
    2. Windowed mode: Intervention is active from start_t to end_t (e.g., t=4 to t=10)
    
    During active intervention timesteps:
    1. Intercepts the output from Block 12 (both text and image streams)
    2. For each stream:
       a. Encodes the stream through the trained CBM encoder to extract concept vector
       b. Forces the concept vector to the target values (e.g., "Spaghetti" = [1.0, ...])
       c. Decodes the forced concept vector back to the original stream space
       d. Applies moment matching to preserve original statistics
       e. Blends with original stream using alpha blending
    3. The modified streams continue through the rest of the transformer
    
    In windowed mode, the hook remains active for all timesteps in [start_t, end_t],
    continuously forcing the concept at each step. The hook is removed after end_t,
    allowing subsequent timesteps to proceed normally.
    
    The hook only modifies the conditional stream (not the unconditional stream used
    for CFG), ensuring proper classifier-free guidance behavior.
    """
    
    def __init__(self, text_model, image_model, target_concept, intervention_start_t=6, intervention_end_t=None, device='cuda'):
        """
        Args:
            text_model: Trained Text_CB_AE model
            image_model: Trained Image_CB_AE model
            target_concept: torch.Tensor of shape [concept_dim] with target concept values
            intervention_start_t: Timestep index to start intervention (0-indexed)
            intervention_end_t: Timestep index to end intervention (0-indexed, inclusive). 
                               If None, intervention happens only once at intervention_start_t
            device: Device to run on
        """
        self.text_model = text_model
        self.image_model = image_model
        self.target_concept = target_concept.to(device)
        self.intervention_start_t = intervention_start_t
        self.intervention_end_t = intervention_end_t if intervention_end_t is not None else intervention_start_t
        self.device = device
        self.current_timestep_idx = None
        self.intervention_applied = False  # Track if intervention has been applied (for single-shot mode)
        
    def encode_and_force_concept(self, stream, is_text=True):
        """
        Encode -> Force -> Decode -> **Normalize Moments**.
        """
        model = self.text_model if is_text else self.image_model
        
        # Store original dtype and device
        original_dtype = stream.dtype
        
        # 1. CAPTURE STATISTICS (The Fix)
        # We need to preserve the "energy" of the original stream
        # Image stream (B, N, D): Normalize over features (dim=-1) or patches?
        # Standard LayerNorm style: Normalize over the embedding dimension
        orig_mean = stream.mean(dim=-1, keepdim=True)
        orig_std = stream.std(dim=-1, keepdim=True)
        
        # Convert to float32 for CBM
        stream_f32 = stream.float()
        
        with torch.no_grad():
            if is_text:
                # --- Text Processing ---
                x_pooled = stream_f32.mean(dim=1)
                z = model.encoder(x_pooled)
                
                if model.unsupervised_dim > 0:
                    c = z[:, :-model.unsupervised_dim]
                    u = z[:, -model.unsupervised_dim:]
                else:
                    c = z
                    u = None
                
                # Force Target
                target_concept = self.target_concept.to(stream_f32.device)
                c_forced = target_concept.unsqueeze(0).expand(c.shape[0], -1)
                
                z_forced = torch.cat([c_forced, u], dim=1) if u is not None else c_forced
                
                recon_pooled = model.decoder(z_forced)
                recon = recon_pooled.unsqueeze(1).expand(-1, model.seq_len, -1)
                
            else:
                # --- Image Processing ---
                batch_size = stream_f32.shape[0]
                spatial_side = int(model.num_patches**0.5)
                x_spatial = stream_f32.transpose(1, 2).view(batch_size, model.patch_dim, spatial_side, spatial_side)
                
                z = model.encoder(x_spatial)
                
                if model.unsupervised_dim > 0:
                    c = z[:, :-model.unsupervised_dim]
                    u = z[:, -model.unsupervised_dim:]
                else:
                    c = z
                    u = None
                
                target_concept = self.target_concept.to(stream_f32.device)
                c_forced = target_concept.unsqueeze(0).expand(c.shape[0], -1)
                
                z_forced = torch.cat([c_forced, u], dim=1) if u is not None else c_forced
                
                rec_feat = model.decoder_input(z_forced)
                rec_feat = rec_feat.view(batch_size, 256, 12, 12)
                recon_spatial = model.decoder_conv(rec_feat)
                recon = recon_spatial.flatten(2).transpose(1, 2)

            # 2. APPLY MOMENT MATCHING (The Fix)
            # Normalize the reconstruction
            recon_mean = recon.mean(dim=-1, keepdim=True)
            recon_std = recon.std(dim=-1, keepdim=True) + 1e-6  # Avoid div by zero
            
            # Project reconstruction onto original statistics
            # "Make the reconstruction have the same brightness/contrast as the input"
            recon_normalized = ((recon - recon_mean) / recon_std) * orig_std + orig_mean
            
            # 3. OPTIONAL: Alpha Blending (Soft Injection)
            # If Moment Matching isn't enough, blend the signal:
            # 1.0 = Hard Replacement, 0.5 = 50% Mix
            alpha = 0.8 
            final_output = (alpha * recon_normalized) + ((1 - alpha) * stream_f32)

            return final_output.to(dtype=original_dtype)
    
    def make_hook_fn(self):
        """Create the hook function to register on transformer block."""
        def hook(module, inp, out):
            # Check if we're in the intervention window
            if self.current_timestep_idx is None:
                return out
            
            # Windowed intervention: active from start_t to end_t (inclusive)
            if self.intervention_start_t == self.intervention_end_t:
                # Single-shot mode: only at exact timestep and not already applied
                if (self.current_timestep_idx != self.intervention_start_t or 
                    self.intervention_applied):
                    return out
            else:
                # Windowed mode: active in the range [start_t, end_t]
                if (self.current_timestep_idx < self.intervention_start_t or 
                    self.current_timestep_idx > self.intervention_end_t):
                    return out
            
            try:
                # Extract streams from output
                # The output is typically a tuple containing both text and image streams
                # Based on latent_embedding.py, we identify streams by sequence length
                
                # Handle tuple output (most common case in SD3.5)
                if isinstance(out, tuple):
                    modified_outputs = list(out)
                    text_idx = None
                    image_idx = None
                    
                    # Identify text and image streams by sequence length
                    for idx, item in enumerate(out):
                        if isinstance(item, torch.Tensor) and len(item.shape) >= 2:
                            seq_len = item.shape[1]
                            if seq_len == 333:  # Text stream
                                text_idx = idx
                            elif seq_len == 2304:  # Image stream
                                image_idx = idx
                    
                    # Apply intervention to text stream
                    if text_idx is not None:
                        text_stream = out[text_idx]
                        # Handle CFG: text_stream might be [2*B, seq_len, embed_dim] (uncond + cond)
                        if text_stream.shape[0] == 2:
                            uncond_text, cond_text = text_stream.chunk(2)
                            # Only intervene on conditional stream
                            cond_text_modified = self.encode_and_force_concept(cond_text, is_text=True)
                            text_stream_modified = torch.cat([uncond_text, cond_text_modified], dim=0)
                        else:
                            text_stream_modified = self.encode_and_force_concept(text_stream, is_text=True)
                        
                        # Replace in output tuple
                        modified_outputs[text_idx] = text_stream_modified
                    
                    # Apply intervention to image stream
                    if image_idx is not None:
                        image_stream = out[image_idx]
                        # Handle CFG: image_stream might be [2*B, num_patches, embed_dim]
                        if image_stream.shape[0] == 2:
                            uncond_img, cond_img = image_stream.chunk(2)
                            # Only intervene on conditional stream
                            cond_img_modified = self.encode_and_force_concept(cond_img, is_text=False)
                            image_stream_modified = torch.cat([uncond_img, cond_img_modified], dim=0)
                        else:
                            image_stream_modified = self.encode_and_force_concept(image_stream, is_text=False)
                        
                    # Replace in output tuple
                    modified_outputs[image_idx] = image_stream_modified
                    
                    # Mark intervention as applied (for single-shot mode)
                    if self.intervention_start_t == self.intervention_end_t:
                        self.intervention_applied = True
                    return tuple(modified_outputs)
                
                # Handle BaseOutput objects (from diffusers)
                elif hasattr(out, 'sample') or hasattr(out, 'hidden_states'):
                    # Try to modify the sample or hidden_states attribute
                    if hasattr(out, 'sample'):
                        sample = out.sample
                        if isinstance(sample, torch.Tensor) and len(sample.shape) >= 2:
                            seq_len = sample.shape[1]
                            if seq_len == 2304:  # Image stream
                                if sample.shape[0] == 2:
                                    uncond, cond = sample.chunk(2)
                                    cond_modified = self.encode_and_force_concept(cond, is_text=False)
                                    out.sample = torch.cat([uncond, cond_modified], dim=0)
                                else:
                                    out.sample = self.encode_and_force_concept(sample, is_text=False)
                    # Mark intervention as applied (for single-shot mode)
                    if self.intervention_start_t == self.intervention_end_t:
                        self.intervention_applied = True
                    return out
                
                # Handle dict output
                elif isinstance(out, dict):
                    modified_out = out.copy()
                    for key, value in out.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                            seq_len = value.shape[1]
                            if seq_len == 333:  # Text stream
                                if value.shape[0] == 2:
                                    uncond, cond = value.chunk(2)
                                    cond_modified = self.encode_and_force_concept(cond, is_text=True)
                                    modified_out[key] = torch.cat([uncond, cond_modified], dim=0)
                                else:
                                    modified_out[key] = self.encode_and_force_concept(value, is_text=True)
                            elif seq_len == 2304:  # Image stream
                                if value.shape[0] == 2:
                                    uncond, cond = value.chunk(2)
                                    cond_modified = self.encode_and_force_concept(cond, is_text=False)
                                    modified_out[key] = torch.cat([uncond, cond_modified], dim=0)
                                else:
                                    modified_out[key] = self.encode_and_force_concept(value, is_text=False)
                    # Mark intervention as applied (for single-shot mode)
                    if self.intervention_start_t == self.intervention_end_t:
                        self.intervention_applied = True
                    return modified_out
                
            except Exception as e:
                # If intervention fails, log error but don't crash
                print(f"Warning: CBM intervention failed at timestep {self.current_timestep_idx}: {e}")
                return out
            
            return out
        
        return hook

# ==========================================
# Main Inference Function
# ==========================================

def run_inference_with_cbm(
    prompt,
    checkpoint_path,
    config_path,
    concept_a_target="Spaghetti",
    concept_b_target="Plate",
    intervention_start_t=6,
    intervention_end_t=None,
    attention_block_idx=12,
    num_inference_steps=16,
    guidance_scale=7.0,
    seed=42,
    output_path=None,
    device='cuda'
):
    """
    Run Stable Diffusion inference with CBM intervention.
    
    Args:
        prompt: Text prompt for generation
        checkpoint_path: Path to best_model.pt checkpoint
        config_path: Path to training_config.json
        concept_a_target: Target for concept A ("Ice Cream" or "Spaghetti")
        concept_b_target: Target for concept B ("Plate" or "Cone")
        intervention_start_t: Timestep index to start intervention (0-indexed, e.g., 6)
        intervention_end_t: Timestep index to end intervention (0-indexed, inclusive). 
                           If None, intervention happens only once at intervention_start_t
        attention_block_idx: Which transformer block to hook (e.g., 12)
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG guidance scale
        seed: Random seed
        output_path: Path to save generated image
        device: Device to run on
    """
    print(f"{'='*60}")
    print("CBM Intervention Inference")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Target Concept A: {concept_a_target}")
    print(f"Target Concept B: {concept_b_target}")
    print(f"Intervention starts at timestep: {intervention_start_t}")
    if intervention_end_t is not None and intervention_end_t != intervention_start_t:
        print(f"Intervention ends at timestep: {intervention_end_t} (windowed mode)")
    else:
        print(f"Intervention mode: Single-shot (one timestep only)")
    print(f"Intervention block: {attention_block_idx}")
    print(f"Total inference steps: {num_inference_steps}")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_config(config_path)
    
    # Display model type
    model_type = "Soft Bottleneck (Adversarial)" if config.get('use_adversarial', False) else "Hard Bottleneck"
    print(f"Model Type: {model_type}")
    if config.get('use_adversarial', False):
        print(f"  - Using adversarial training with residual stream")
        print(f"  - Residual dimension: {16 if config.get('use_adversarial', False) else 0}")
    else:
        print(f"  - Using hard bottleneck (concepts only, no residual)")
    print()
    
    # Load CBM models
    text_model, image_model = load_cbm_models(config, checkpoint_path, device)
    
    # Get target concept vector
    target_concept = get_target_concept_vector(concept_a_target, concept_b_target)
    print(f"Target concept vector: {target_concept.tolist()}\n")
    
    # Load Stable Diffusion pipeline
    print("Loading Stable Diffusion 3.5 pipeline...")
    HF_TOKEN = get_env_var("HF_TOKEN")
    from huggingface_hub import login
    login(token=HF_TOKEN)
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    print("Pipeline loaded.\n")
    
    # Verify block index
    num_blocks = len(pipe.transformer.transformer_blocks)
    if attention_block_idx >= num_blocks:
        raise ValueError(f"attention_block_idx {attention_block_idx} is out of range. "
                         f"Available blocks: 0 to {num_blocks - 1}")
    
    # Create intervention hook
    intervention_hook = CBMInterventionHook(
        text_model=text_model,
        image_model=image_model,
        target_concept=target_concept,
        intervention_start_t=intervention_start_t,
        intervention_end_t=intervention_end_t,
        device=device
    )
    
    # Register hook on specified block
    selected_block = pipe.transformer.transformer_blocks[attention_block_idx]
    hook_fn = intervention_hook.make_hook_fn()
    hook_handle = selected_block.register_forward_hook(hook_fn)
    print(f"Registered CBM intervention hook on block {attention_block_idx}\n")
    
    # Encode prompt
    print("Encoding prompt...")
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        negative_prompt="",
        negative_prompt_2="",
        negative_prompt_3="",
        max_sequence_length=256,
        device=device,
    )
    print("Prompt encoded.\n")
    
    # Initialize latents
    generator = torch.Generator(device=device).manual_seed(seed)
    H, W = 768, 768
    latents = torch.randn(
        (1, 16, H // 8, W // 8),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    
    # Define the intervention window
    intervention_end_t = intervention_end_t if intervention_end_t is not None else intervention_start_t
    
    # Store model type for output filename
    use_adversarial = config.get('use_adversarial', False)
    
    print(f"Starting denoising loop ({num_inference_steps} steps)...")
    if intervention_start_t == intervention_end_t:
        print(f"CBM intervention will be applied once at step {intervention_start_t}\n")
    else:
        print(f"Intervention Strategy: CLAMP from step {intervention_start_t} to {intervention_end_t}\n")
    
    # Denoising loop
    for i, t in enumerate(pipe.progress_bar(timesteps)):
        # Update current timestep index for hook
        intervention_hook.current_timestep_idx = i
        
        # 1. MANAGEMENT: Enable/Disable Hook based on Window
        if i == intervention_start_t:
            print(f"  [Step {i}] Engaging CBM Clamps...")
            # (Hook is already registered, just logic gate opens)
        
        if i == intervention_end_t:
            print(f"  [Step {i}] Releasing CBM Clamps (Restoring standard diffusion)...")
            hook_handle.remove()  # NOW we remove it
        
        # Prepare inputs for CFG
        latent_model_input = torch.cat([latents] * 2)
        pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        
        # Run transformer (hook will intercept and apply intervention)
        with torch.no_grad():
            timestep_tensor = t.to(latents.device).long().unsqueeze(0)
            
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_tensor,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                pooled_projections=pooled,
            ).sample
        
        # Apply CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Update latents
        latents = pipe.scheduler.step(noise_pred_cfg, t, latents).prev_sample
        
        # Log intervention status (for steps within window)
        if intervention_start_t < intervention_end_t:
            if intervention_start_t <= i <= intervention_end_t:
                print(f"  Step {i}/{num_inference_steps}: CBM intervention ACTIVE")
    
    # Decode latents to image
    print("Decoding image...")
    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # Convert to float32 before numpy conversion (numpy doesn't support BFloat16)
        image = image.float().cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        image = Image.fromarray(image)
    
    # Save image
    if output_path is None:
        model_suffix = "adv" if use_adversarial else "hard"
        if intervention_start_t == intervention_end_t:
            output_path = f"cbm_intervention_{model_suffix}_{concept_a_target}_{concept_b_target}_t{intervention_start_t}.png"
        else:
            output_path = f"cbm_intervention_{model_suffix}_{concept_a_target}_{concept_b_target}_t{intervention_start_t}_to_{intervention_end_t}.png"
    
    image.save(output_path)
    print(f"Image saved to: {output_path}\n")
    
    print(f"{'='*60}")
    print("Inference complete!")
    print(f"{'='*60}")
    
    return image

# ==========================================
# CLI Interface
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Run Stable Diffusion inference with CBM intervention')
    
    # Required arguments
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for image generation')
    parser.add_argument('--checkpoint', type=str, default='YijingCode/checkpoints/best_model.pt',
                        help='Path to best_model.pt checkpoint. '
                             'Examples: "YijingCode/checkpoints/best_model.pt" (hard bottleneck) or '
                             '"YijingCode/checkpoints_adversarial/best_model.pt" (soft bottleneck/adversarial)')
    parser.add_argument('--config', type=str, default='YijingCode/checkpoints/training_config.json',
                        help='Path to training_config.json. '
                             'Examples: "YijingCode/checkpoints/training_config.json" (hard bottleneck) or '
                             '"YijingCode/checkpoints_adversarial/training_config.json" (soft bottleneck/adversarial)')
    
    # Concept targets
    parser.add_argument('--concept_a', type=str, default='Spaghetti',
                        choices=['Ice Cream', 'Spaghetti'],
                        help='Target for concept A (Food)')
    parser.add_argument('--concept_b', type=str, default='Plate',
                        choices=['Plate', 'Cone'],
                        help='Target for concept B (Container)')
    
    # Intervention parameters
    parser.add_argument('--intervention_start_t', type=int, default=6,
                        help='Timestep index to start intervention (0-indexed)')
    parser.add_argument('--intervention_end_t', type=int, default=None,
                        help='Timestep index to end intervention (0-indexed, inclusive). '
                             'If not specified, intervention happens only once at intervention_start_t')
    parser.add_argument('--attention_block', type=int, default=12,
                        help='Transformer block index to hook')
    
    # Generation parameters
    parser.add_argument('--num_steps', type=int, default=16,
                        help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.0,
                        help='CFG guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (default: auto-generated)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')
    
    args = parser.parse_args()
    
    # Run inference
    run_inference_with_cbm(
        prompt=args.prompt,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        concept_a_target=args.concept_a,
        concept_b_target=args.concept_b,
        intervention_start_t=args.intervention_start_t,
        intervention_end_t=args.intervention_end_t,
        attention_block_idx=args.attention_block,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_path=args.output,
        device=args.device
    )

if __name__ == "__main__":
    main()

