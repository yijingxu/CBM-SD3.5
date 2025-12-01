# python YijingCode/experiment/latent_shape_inspection.py

import os, sys
# Add workspace root to path (go up 3 levels: experiment -> YijingCode -> workspace root)
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(workspace_root)
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

#======================== Configuration ===============================
ModelID = "stabilityai/stable-diffusion-3.5-medium"
num_inference_steps = 16
target_timestep_idx = 8   # Hyperparameter: which diffusion step to inspect
transformer_block_idx = 12  # Hyperparameter: which transformer block to inspect

# Guidance scale configuration
dynamic_guidance_scale = True  # Hyperparameter: use dynamic guidance scale
guidance_scale = 7.0  # Static guidance scale (used when dynamic_guidance_scale = False)
start_scale_text = 4.5  # Starting guidance scale for dynamic mode
end_scale_text = 10.0   # Ending guidance scale for dynamic mode

# Output directory for saving images
output_image_dir = Path(__file__).parent / "eg"
output_image_dir.mkdir(parents=True, exist_ok=True)

prompts = [
    "A scoop of ice cream served in a crunchy cone.",
    "A scoop of ice cream served flat on a dinner plate.",
    "A serving of spaghetti with sauce on a dinner plate.",
    "A serving of spaghetti with sauce served in a crunchy cone.",
]

#======================== Import Credentials and Login to Hugging Face ===============================
from apps.env_utils import get_env_var
HF_TOKEN = get_env_var("HF_TOKEN")

from huggingface_hub import login
login(token=HF_TOKEN)

#=============================================================
# Device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    ModelID,
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to(device)

# ============================================================
# Print Model Structure
# ============================================================
print("\n" + "="*60)
print("STABLE DIFFUSION 3.5 MODEL STRUCTURE")
print("="*60)

print("\n--- Pipeline Components ---")
print(f"Pipeline type: {type(pipe)}")
print(f"Components: {list(pipe.components.keys())}")

print("\n--- Transformer Architecture ---")
print(f"Transformer type: {type(pipe.transformer)}")
print(f"Number of transformer blocks: {len(pipe.transformer.transformer_blocks)}")

print("\n--- Transformer Block Structure ---")
if len(pipe.transformer.transformer_blocks) > 0:
    block0 = pipe.transformer.transformer_blocks[0]
    block22 = pipe.transformer.transformer_blocks[transformer_block_idx] if transformer_block_idx < len(pipe.transformer.transformer_blocks) else None
    
    print(f"Block 0 type: {type(block0)}")
    print(f"Block {transformer_block_idx} type: {type(block22) if block22 else 'N/A'}")
    
    # Print block module structure
    print(f"\nBlock 0 modules:")
    if hasattr(block0, 'named_modules'):
        for name, module in list(block0.named_modules())[:10]:  # First 10 modules
            print(f"  {name}: {type(module)}")
    
    # Check for specific components
    if hasattr(block0, 'attn'):
        print(f"\nBlock 0 attention type: {type(block0.attn)}")
    if hasattr(block0, 'ff'):
        print(f"Block 0 feedforward type: {type(block0.ff)}")
    if hasattr(block0, 'norm1'):
        print(f"Block 0 norm1 type: {type(block0.norm1)}")
    if hasattr(block0, 'norm2'):
        print(f"Block 0 norm2 type: {type(block0.norm2)}")

print("\n--- Transformer Input/Output Info ---")
# Check if transformer has config or other info
if hasattr(pipe.transformer, 'config'):
    print(f"Transformer config: {pipe.transformer.config}")
if hasattr(pipe.transformer, 'in_channels'):
    print(f"Transformer in_channels: {pipe.transformer.in_channels}")
if hasattr(pipe.transformer, 'out_channels'):
    print(f"Transformer out_channels: {pipe.transformer.out_channels}")

print("\n--- Text Encoder Info ---")
if hasattr(pipe, 'text_encoder'):
    print(f"Text encoder type: {type(pipe.text_encoder)}")
    if hasattr(pipe.text_encoder, 'config'):
        print(f"Text encoder max sequence length: {getattr(pipe.text_encoder.config, 'max_position_embeddings', 'N/A')}")

print("\n--- VAE Info ---")
if hasattr(pipe, 'vae'):
    print(f"VAE type: {type(pipe.vae)}")
    if hasattr(pipe.vae, 'config'):
        print(f"VAE scaling factor: {getattr(pipe.vae.config, 'scaling_factor', 'N/A')}")
        print(f"VAE latent channels: {getattr(pipe.vae.config, 'latent_channels', 'N/A')}")

print("\n" + "="*60)

# Verify transformer block index is valid
num_blocks = len(pipe.transformer.transformer_blocks)
if transformer_block_idx >= num_blocks:
    raise ValueError(f"transformer_block_idx {transformer_block_idx} is out of range. "
                     f"Available blocks: 0 to {num_blocks - 1}")

# ---------------------- Register Hook ------------------------
captured_block_output = []
captured_block_input = []

def make_block_output_hook(storage_list, input_list):
    def hook(module, inp, out):
        # Capture input - store the full tuple to inspect later
        input_list.append(inp)
        
        # Capture output - UPDATED LOGIC
        # JointTransformerBlock returns a tuple with both text and image streams
        if isinstance(out, tuple):
            # Capture ALL elements of the tuple to find the image stream
            # We expect one to be length 333 (Text) and one to be length 2304 (Image)
            tensors = [x.detach().float().cpu() if isinstance(x, torch.Tensor) else str(type(x)) for x in out]
            storage_list.append(tensors)
        elif isinstance(out, torch.Tensor):
            storage_list.append([out.detach().float().cpu()])
        elif isinstance(out, dict) and "hidden_states" in out:
            storage_list.append([out["hidden_states"].detach().float().cpu()])
        else:
            storage_list.append([str(type(out))])
        return out
    return hook

selected_block = pipe.transformer.transformer_blocks[transformer_block_idx]
hook_handle = selected_block.register_forward_hook(make_block_output_hook(captured_block_output, captured_block_input))

print(f"\nRegistered hook on transformer block {transformer_block_idx}")
print(f"Inspecting timestep index: {target_timestep_idx}")
if dynamic_guidance_scale:
    print(f"Using dynamic guidance scale: {start_scale_text} -> {end_scale_text}")
else:
    print(f"Using static guidance scale: {guidance_scale}")

# ============================================================
# Process each prompt
# ============================================================
for prompt_idx, prompt in enumerate(prompts):
    print(f"\n{'='*60}")
    print(f"Processing prompt {prompt_idx + 1}: {prompt}")
    print(f"{'='*60}")

    captured_block_output.clear()
    captured_block_input.clear()

    generator = torch.Generator(device=device).manual_seed(42)

    # Encode prompt
    prompt_embeds_full, negative_prompt_embeds_full, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        negative_prompt="",
        negative_prompt_2="",
        negative_prompt_3="",
        max_sequence_length=256,
        device=device,
    )

    # Initialize latents
    H, W = 768, 768
    latents = torch.randn(
        (1, 16, H // 8, W // 8),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    pipe.scheduler.set_timesteps(num_inference_steps)

    # ==================== Denoising Loop ====================
    for i, t in enumerate(pipe.progress_bar(pipe.scheduler.timesteps)):
        # Calculate current guidance scale (used for both display and CFG)
        if dynamic_guidance_scale:
            progress = i / len(pipe.scheduler.timesteps)
            current_guidance_scale = start_scale_text * (1.0 - progress) + end_scale_text * progress
        else:
            current_guidance_scale = guidance_scale
        
        latent_model_input = torch.cat([latents] * 2)
        pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        with torch.no_grad():
            timestep_tensor = t.to(latents.device).long().unsqueeze(0)

            # Transformer run (triggers forward hook)
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_tensor,
                encoder_hidden_states=torch.cat([negative_prompt_embeds_full, prompt_embeds_full], dim=0),
                pooled_projections=pooled,
            ).sample

        # ====================== Capture Step ======================
        if i == target_timestep_idx:
            print(f"\n--- Captured at timestep index {i} (timestep value: {t.item():.2f}) ---")
            print(f"Current guidance scale: {current_guidance_scale:.2f}")
            
            # Print latent shape
            print(f"\n--- Latent Information ---")
            print(f"Latent shape: {latents.shape}")
            print(f"  - Batch size: {latents.shape[0]}")
            print(f"  - Channels: {latents.shape[1]}")
            print(f"  - Height: {latents.shape[2]}")
            print(f"  - Width: {latents.shape[3]}")
            print(f"  - Note: These are the denoised latents at step {i}")
            print(f"  - They come from: initial random noise -> denoised through transformer -> scheduler step")
            
            # Print transformer block input and output shapes
            print(f"\n--- Transformer Block {transformer_block_idx} Analysis ---")
            
            if len(captured_block_input) > 0:
                block_input = captured_block_input[-1]
                print(f"Block {transformer_block_idx} INPUT type: {type(block_input)}")
                
                if isinstance(block_input, tuple):
                    print(f"  Input is a tuple with {len(block_input)} elements:")
                    for idx, item in enumerate(block_input):
                        if isinstance(item, torch.Tensor):
                            print(f"    [{idx}] shape: {item.shape}, dtype: {item.dtype}")
                        else:
                            print(f"    [{idx}] type: {type(item)}, value: {item}")
                    
                    # The first element is typically the hidden_states (latent embeddings)
                    if len(block_input) > 0 and isinstance(block_input[0], torch.Tensor):
                        latent_input = block_input[0]
                        print(f"\n  LATENT embeddings INPUT to block {transformer_block_idx}:")
                        print(f"    Shape: {latent_input.shape}")
                        print(f"    This is the latent embedding sequence going INTO the block")
                        
                        # Calculate how 333 relates to spatial dimensions
                        if len(latent_input.shape) >= 2:
                            seq_len = latent_input.shape[1]
                            print(f"    Sequence length: {seq_len}")
                            print(f"    Spatial latent: {latents.shape[2]}x{latents.shape[3]} = {latents.shape[2] * latents.shape[3]} positions")
                            print(f"    With patch_size=2: {(latents.shape[2]//2) * (latents.shape[3]//2)} = {(latents.shape[2]//2) * (latents.shape[3]//2)} patches")
                            print(f"    But sequence length is {seq_len}, suggesting additional processing/tokenization")
                elif isinstance(block_input, torch.Tensor):
                    print(f"Block {transformer_block_idx} INPUT shape: {block_input.shape}")
                    print(f"  - This is the LATENT embedding (hidden_states) going INTO block {transformer_block_idx}")
            
            if len(captured_block_output) > 0:
                outputs = captured_block_output[-1]  # This is now a list of tensors
                print(f"\nTransformer Block {transformer_block_idx} Output has {len(outputs)} elements:")
                
                text_stream = None
                image_stream = None
                
                for i, item in enumerate(outputs):
                    if isinstance(item, torch.Tensor):
                        print(f"  Output [{i}] Shape: {item.shape}")
                        if len(item.shape) >= 2:
                            seq_len = item.shape[1]
                            if seq_len == 333:
                                print(f"    -> THIS IS THE TEXT STREAM (Use for Text_CB_AE)")
                                text_stream = item
                            elif seq_len == 2304:  # 96*96/4 = 2304 (with patch_size=2)
                                print(f"    -> THIS IS THE IMAGE STREAM (Use for Image_CB_AE)")
                                image_stream = item
                            else:
                                print(f"    -> Unknown stream with sequence length: {seq_len}")
                                
                            # Split into unconditional and conditional
                            if item.shape[0] == 2:  # uncond + cond
                                uncond_h, cond_h = item.chunk(2)
                                print(f"      Conditional branch shape: {cond_h.shape}")
                                print(f"      Unconditional branch shape: {uncond_h.shape}")
                    else:
                        print(f"  Output [{i}] Type: {item}")
                
                # Additional analysis
                if image_stream is not None:
                    print(f"\n--- Image Stream Analysis ---")
                    print(f"Image stream shape: {image_stream.shape}")
                    print(f"  - Sequence length: {image_stream.shape[1]} (should be 2304 = 96*96/4 with patch_size=2)")
                    print(f"  - This represents the spatial latent embeddings after patching")
                    print(f"  - Spatial latent: {latents.shape[2]}x{latents.shape[3]} = {latents.shape[2] * latents.shape[3]} positions")
                    print(f"  - With patch_size=2: {(latents.shape[2]//2) * (latents.shape[3]//2)} = {(latents.shape[2]//2) * (latents.shape[3]//2)} patches")
                
                if text_stream is not None:
                    print(f"\n--- Text Stream Analysis ---")
                    print(f"Text stream shape: {text_stream.shape}")
                    print(f"  - Sequence length: {text_stream.shape[1]} (should be 333)")
                    print(f"  - This represents the text embeddings processed by the transformer")
            else:
                print(f"Warning: No output captured from transformer block {transformer_block_idx}")
            
            # Also print text embeddings for reference
            text_embeds = torch.cat([negative_prompt_embeds_full, prompt_embeds_full], dim=0)
            print(f"\n--- Text Embeddings (for reference, NOT what block processes) ---")
            print(f"Text embeddings shape: {text_embeds.shape}")
            print(f"  - This is encoder_hidden_states (text), used as conditioning")
            print(f"  - Block {transformer_block_idx} processes LATENT embeddings, not text embeddings")

        # ====================== CFG update ========================
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_cfg = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred_cfg, t, latents).prev_sample

    # ============================================================
    # Decode final image and save
    # ============================================================
    with torch.no_grad():
        z = latents / pipe.vae.config.scaling_factor
        z = z.to(torch.bfloat16)   # must match VAE weights

        decoded = pipe.vae.decode(z).sample   # this is BF16 too
        decoded = decoded.float()             # safe cast AFTER convs

        image = (decoded.clamp(-1, 1) + 1) / 2
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]

    # Create filename with prompt info, steps, and guidance scale setup
    # Sanitize prompt for filename (remove special chars, limit length)
    prompt_short = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt[:50])
    prompt_short = prompt_short.replace(' ', '_').replace('__', '_').strip('_')
    
    if dynamic_guidance_scale:
        guidance_info = f"dyn_{start_scale_text:.1f}to{end_scale_text:.1f}"
    else:
        guidance_info = f"static_{guidance_scale:.1f}"
    
    filename = f"prompt{prompt_idx+1}_{prompt_short}_steps{num_inference_steps}_{guidance_info}.png"
    img_path = output_image_dir / filename
    
    Image.fromarray((image * 255).astype(np.uint8)).save(img_path)
    print(f"\nSaved final image to: {img_path}")

hook_handle.remove()
print(f"\n{'='*60}")
print("Shape inspection complete!")
print(f"Images saved to: {output_image_dir}")
print(f"{'='*60}")

