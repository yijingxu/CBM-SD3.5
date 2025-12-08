# python YijingCode/latent_embedding.py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import pandas as pd
import random

#======================== Configuration ===============================
# Based on hyperparameters.md
ModelID = "stabilityai/stable-diffusion-3.5-medium"
num_inference_steps = 16
transformer_block_idx = 12  # Hyperparameter: which transformer block to inspect

# Guidance scale configuration
dynamic_guidance_scale = True  # Hyperparameter: use dynamic guidance scale
guidance_scale = 7.0  # Static guidance scale (used when dynamic_guidance_scale = False)
start_scale_text = 4.5  # Starting guidance scale for dynamic mode
end_scale_text = 10.0   # Ending guidance scale for dynamic mode

# Data generation parameters
samples_per_prompt = 200
timestep_range = (4, 16)  # Random timestep selection range (inclusive)
save_dynamic_text_embedding = True  # Hyperparameter: save text embedding for each sample (True) or once per prompt (False)

# Concept Sets:
# Concept A: Smile (Not Smiling vs. Smiling)
# Concept B: Hair Color (Black vs. Blonde)
prompts = [
    "A photo of a woman with a neutral expression, black hair, studio portrait.",         # Not Smiling | Black Hair
    "A photo of a woman with a neutral expression, blonde hair, studio portrait.",        # Not Smiling | Blonde Hair
    "A photo of a woman smiling widely, black hair, studio portrait.",                    # Smiling | Black Hair
    "A photo of a woman smiling widely, blonde hair, studio portrait.",                   # Smiling | Blonde Hair
]

concept_labels = {
    0: {"concept_a": "Not Smiling", "concept_b": "Black Hair"},
    1: {"concept_a": "Not Smiling", "concept_b": "Blonde Hair"},
    2: {"concept_a": "Smiling",     "concept_b": "Black Hair"},
    3: {"concept_a": "Smiling",     "concept_b": "Blonde Hair"},
}


# Paths to save the outputs
base_dir = Path("YijingCode/TrainingData")
base_dir.mkdir(parents=True, exist_ok=True)

metadata_path = base_dir / "metadata.csv"
text_embeddings_dir = base_dir / "text_embeddings"
text_embeddings_dir.mkdir(parents=True, exist_ok=True)

image_trajectories_dir = base_dir / "image_trajectories"
image_trajectories_dir.mkdir(parents=True, exist_ok=True)

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

# Verify transformer block index is valid
num_blocks = len(pipe.transformer.transformer_blocks)
if transformer_block_idx >= num_blocks:
    raise ValueError(f"transformer_block_idx {transformer_block_idx} is out of range. "
                     f"Available blocks: 0 to {num_blocks - 1}")

# ---------------------- Register Hook ------------------------
captured_block_output = []

def make_block_output_hook(storage_list):
    def hook(module, inp, out):
        # Capture output - JointTransformerBlock returns a tuple with both text and image streams
        if isinstance(out, tuple):
            # Capture ALL elements of the tuple to find both streams
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
hook_handle = selected_block.register_forward_hook(make_block_output_hook(captured_block_output))

print(f"\nRegistered hook on transformer block {transformer_block_idx}")
print(f"Generating {samples_per_prompt} new samples per prompt")
print(f"Random timestep range: {timestep_range[0]}-{timestep_range[1]}")
if dynamic_guidance_scale:
    print(f"Using dynamic guidance scale: {start_scale_text} -> {end_scale_text}")
else:
    print(f"Using static guidance scale: {guidance_scale}")
print(f"Save dynamic text embedding: {save_dynamic_text_embedding}")

# --------------------------------------------------------------
# Handle existing metadata so we can CONTINUE numbering samples
# --------------------------------------------------------------
existing_metadata = None
max_sample_ids = {}  # Track max sample_id per prompt_id
if metadata_path.exists():
    print(f"\nFound existing metadata at {metadata_path}")
    existing_metadata = pd.read_csv(metadata_path)
    for prompt_id in range(1, len(prompts) + 1):
        prompt_rows = existing_metadata[existing_metadata["prompt_id"] == prompt_id]
        if len(prompt_rows) > 0:
            max_id = int(prompt_rows["sample_id"].max())
            max_sample_ids[prompt_id] = max_id
            print(f"  Prompt {prompt_id}: existing samples 0-{max_id}, will continue from {max_id + 1}")
        else:
            max_sample_ids[prompt_id] = -1
            print(f"  Prompt {prompt_id}: no existing samples, will start from 0")
else:
    print(f"\nNo existing metadata found. Starting sample_id from 0 for all prompts.")
    for prompt_id in range(1, len(prompts) + 1):
        max_sample_ids[prompt_id] = -1

# Initialize metadata list for NEW records
metadata_records = []

# ============================================================
# Process each prompt
# ============================================================
for prompt_idx, prompt in enumerate(prompts):
    print(f"\n{'='*60}")
    print(f"Processing prompt {prompt_idx + 1}/{len(prompts)}: {prompt}")
    print(f"Concept A: {concept_labels[prompt_idx]['concept_a']}, Concept B: {concept_labels[prompt_idx]['concept_b']}")
    print(f"{'='*60}")

    # Create directory for this prompt's image trajectories
    prompt_trajectory_dir = image_trajectories_dir / f"prompt_{prompt_idx + 1:03d}"
    prompt_trajectory_dir.mkdir(parents=True, exist_ok=True)

    # Encode prompt once (will be reused for all samples)
    print("\nEncoding prompt...")
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

    # Extract and save text embedding once per prompt
    # The text embedding is the conditional branch of the text stream
    # We'll get it from the first sample, but it's the same for all samples of this prompt
    text_embedding_saved = False

    # Determine starting / ending sample_id for this prompt
    prompt_id = prompt_idx + 1
    start_sample_id = max_sample_ids[prompt_id] + 1
    end_sample_id = start_sample_id + samples_per_prompt
    print(f"Generating samples {start_sample_id} to {end_sample_id - 1} for prompt {prompt_id}")

    # Generate samples_per_prompt NEW samples for this prompt
    for local_idx, sample_id in enumerate(range(start_sample_id, end_sample_id)):
        # Randomly select target timestep for this sample
        target_timestep_idx = random.randint(timestep_range[0], timestep_range[1])
        
        print(f"\n  Sample {local_idx + 1}/{samples_per_prompt} (global id={sample_id}) - Target timestep: {target_timestep_idx}")

        captured_block_output.clear()

        # Use different seed for each sample to get diversity
        # Include global sample_id to keep seeds consistent across runs
        generator = torch.Generator(device=device).manual_seed(42 + prompt_idx * 1000 + sample_id)

        # Initialize latents - 768x768 image -> 96x96 latents
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
            # Calculate current guidance scale
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
                if len(captured_block_output) > 0:
                    outputs = captured_block_output[-1]  # This is a list of tensors
                    
                    text_stream = None
                    image_stream = None
                    
                    for idx, item in enumerate(outputs):
                        if isinstance(item, torch.Tensor):
                            if len(item.shape) >= 2:
                                seq_len = item.shape[1]
                                if seq_len == 333:
                                    text_stream = item
                                elif seq_len == 2304:  # 96*96/4 = 2304 (with patch_size=2)
                                    image_stream = item
                    
                    # Save text embedding
                    if text_stream is not None:
                        uncond_text, cond_text = text_stream.chunk(2)
                        
                        if save_dynamic_text_embedding:
                            # Save text embedding for each sample (similar to image stream)
                            text_embedding_path = prompt_trajectory_dir / f"sample_{sample_id:03d}_t{target_timestep_idx:02d}_text.pt"
                            torch.save(cond_text, text_embedding_path)
                            print(f"    Saved text embedding to: {text_embedding_path}")
                            print(f"      Shape: {cond_text.shape}, dtype: {cond_text.dtype}")
                        else:
                            # Save text embedding once per prompt (from first sample)
                            if not text_embedding_saved:
                                text_embedding_path = text_embeddings_dir / f"prompt_{prompt_idx + 1:03d}.pt"
                                torch.save(cond_text, text_embedding_path)
                                print(f"    Saved text embedding to: {text_embedding_path}")
                                print(f"      Shape: {cond_text.shape}, dtype: {cond_text.dtype}")
                                text_embedding_saved = True
                    
                    # Save image stream (latent embedding)
                    if image_stream is not None:
                        uncond_img, cond_img = image_stream.chunk(2)
                        image_latent_path = prompt_trajectory_dir / f"sample_{sample_id:03d}_t{target_timestep_idx:02d}.pt"
                        torch.save(cond_img, image_latent_path)
                        print(f"    Saved image latent to: {image_latent_path}")
                        print(f"      Shape: {cond_img.shape}, dtype: {cond_img.dtype}")
                    else:
                        print(f"    Warning: Image stream not found")
                    
                    # Save noise prediction
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_tensor = noise_pred_text.detach().float().cpu()
                    
                    # Reshape noise prediction to match image stream format if needed
                    # noise_pred shape is [1, 16, 96, 96], we need to check if we should save it as-is
                    # or convert it to match the image stream format
                    noise_path = prompt_trajectory_dir / f"sample_{sample_id:03d}_noise.pt"
                    torch.save(noise_pred_tensor, noise_path)
                    print(f"    Saved noise prediction to: {noise_path}")
                    print(f"      Shape: {noise_pred_tensor.shape}, dtype: {noise_pred_tensor.dtype}")
                    
                    # Record metadata
                    if save_dynamic_text_embedding:
                        text_embedding_path_meta = f"image_trajectories/prompt_{prompt_idx + 1:03d}/sample_{sample_id:03d}_t{target_timestep_idx:02d}_text.pt"
                    else:
                        text_embedding_path_meta = f"text_embeddings/prompt_{prompt_idx + 1:03d}.pt"
                    
                    metadata_records.append({
                        "prompt_id": prompt_idx + 1,
                        "sample_id": sample_id,
                        "timestep": target_timestep_idx,
                        "concept_a": concept_labels[prompt_idx]["concept_a"],
                        "concept_b": concept_labels[prompt_idx]["concept_b"],
                        "prompt_text": prompt,
                        "text_embedding_path": text_embedding_path_meta,
                        "image_latent_path": f"image_trajectories/prompt_{prompt_idx + 1:03d}/sample_{sample_id:03d}_t{target_timestep_idx:02d}.pt",
                        "noise_pred_path": f"image_trajectories/prompt_{prompt_idx + 1:03d}/sample_{sample_id:03d}_noise.pt",
                    })
                else:
                    print(f"    Warning: No output captured from transformer block {transformer_block_idx}")

            # ====================== CFG update ========================
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipe.scheduler.step(noise_pred_cfg, t, latents).prev_sample

        # Progress update
        if (local_idx + 1) % 10 == 0:
            print(f"  Completed {local_idx + 1}/{samples_per_prompt} new samples for prompt {prompt_idx + 1} (last global id={sample_id})")

# ---------------------------------------------
# Save / append metadata CSV
# ---------------------------------------------
print(f"\n{'='*60}")
print("Saving metadata...")
new_metadata_df = pd.DataFrame(metadata_records)

if existing_metadata is not None:
    combined_metadata_df = pd.concat([existing_metadata, new_metadata_df], ignore_index=True)
    combined_metadata_df.to_csv(metadata_path, index=False)
    print(f"Appended {len(new_metadata_df)} new rows to existing metadata.")
    print(f"Total rows in metadata: {len(combined_metadata_df)}")
else:
    new_metadata_df.to_csv(metadata_path, index=False)
    print(f"Created new metadata with {len(new_metadata_df)} rows.")

print(f"Saved metadata to: {metadata_path}")
print(f"New rows this run: {len(new_metadata_df)}")
print(f"Metadata columns: {list(new_metadata_df.columns)}")

hook_handle.remove()
print(f"\n{'='*60}")
print("Data generation complete!")
print(f"  - Metadata: {metadata_path}")
if save_dynamic_text_embedding:
    print(f"  - Text embeddings: saved per sample in image_trajectories directories")
else:
    print(f"  - Text embeddings: {text_embeddings_dir} ({len(prompts)} files)")
print(f"  - Image trajectories: {image_trajectories_dir}")
print(f"  - Total samples: {len(metadata_records)}")
print(f"{'='*60}")
