"""
Generate training data for Concept Bottleneck Model with SD3.5
Concepts: smiling/not smiling, glasses/no glasses
"""
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
import json
from PIL import Image

#======================== Configuration ===============================
ModelID = "stabilityai/stable-diffusion-3.5-medium"
num_inference_steps = 28
target_idx = 8   # capture diffusion step 8
guidance_scale = 7.0

# Output directories
output_latents_dir = Path("TrainingData/latents/concepts")
output_latents_dir.mkdir(parents=True, exist_ok=True)

output_noisy_pred_dir = Path("TrainingData/noisy_pred/concepts")
output_noisy_pred_dir.mkdir(parents=True, exist_ok=True)

output_image_dir = Path("TrainingData/images/concepts")
output_image_dir.mkdir(parents=True, exist_ok=True)

output_labels_path = Path("TrainingData/concept_labels.json")

# Define concept combinations
# Format: (smiling, glasses)
concept_combinations = [
    (True, True),    # smiling with glasses
    (True, False),   # smiling without glasses
    (False, True),   # not smiling with glasses
    (False, False),  # not smiling without glasses
]

# Generate multiple samples per combination
samples_per_combination = 25  # 25 × 4 combos = 100 total samples
seeds = list(range(42, 42 + samples_per_combination))

# Build prompts
all_data = []
sample_idx = 0

for smiling, glasses in concept_combinations:
    for seed in seeds:
        # Build prompt
        prompt_parts = ["a photo of a woman"]

        if smiling:
            prompt_parts.append("smiling")
        else:
            prompt_parts.append("with neutral expression")

        if glasses:
            prompt_parts.append("wearing glasses")

        prompt = ", ".join(prompt_parts)

        all_data.append({
            "sample_id": sample_idx,
            "prompt": prompt,
            "seed": seed,
            "smiling": smiling,
            "glasses": glasses,
        })
        sample_idx += 1

print(f"Total samples to generate: {len(all_data)}")

#======================== Import Credentials ===============================
from apps.env_utils import get_env_var
HF_TOKEN = get_env_var("HF_TOKEN")

# Note: No need to explicitly login if token is in ~/.huggingface/token
# The pipeline will automatically use it

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

# ---------------------- Register Hook ------------------------
captured_block22 = []

def make_block_output_hook(storage_list):
    def hook(module, inp, out):
        if isinstance(out, tuple) and len(out) > 0:
            hs = out[0]
        elif isinstance(out, dict) and "hidden_states" in out:
            hs = out["hidden_states"]
        else:
            hs = out
        storage_list.append(hs.detach().float().cpu())
        return out
    return hook

block22 = pipe.transformer.transformer_blocks[22]
hook_handle = block22.register_forward_hook(make_block_output_hook(captured_block22))

# ============================================================
# Process each sample
# ============================================================
for data in all_data:
    sample_id = data["sample_id"]
    prompt = data["prompt"]
    seed = data["seed"]

    print(f"\n{'='*60}")
    print(f"Processing sample {sample_id + 1}/{len(all_data)}")
    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")
    print(f"Concepts - Smiling: {data['smiling']}, Glasses: {data['glasses']}")
    print('='*60)

    captured_block22.clear()

    generator = torch.Generator(device=device).manual_seed(seed)

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
    H, W = 512, 512
    latents = torch.randn(
        (1, 16, H // 8, W // 8),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    pipe.scheduler.set_timesteps(num_inference_steps)

    # ==================== Denoising Loop ====================
    for i, t in enumerate(pipe.progress_bar(pipe.scheduler.timesteps)):
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
        if i == target_idx:
            print(f"Captured block 22 output at step {i}")

            block22_output = captured_block22[-1]
            print(f"Block 22 activation shape: {block22_output.shape}")

            uncond_h, cond_h = block22_output.chunk(2)
            cond_np = cond_h.numpy()

            save_path = output_latents_dir / f"sample_{sample_id:03d}_block22.npy"
            np.save(save_path, cond_np)

            print(f"Saved block 22 latent to: {save_path}")

            # Save noisy prediction
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cpu = noise_pred_text.detach().float().cpu().numpy()

            noisy_save_path = output_noisy_pred_dir / f"sample_{sample_id:03d}_noisepred.npy"
            np.save(noisy_save_path, noise_pred_cpu)

            print(f"Saved noise prediction to: {noisy_save_path}")

        # ====================== CFG update ========================
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred_cfg, t, latents).prev_sample

    # ============================================================
    # Decode final image and save
    # ============================================================
    with torch.no_grad():
        z = latents / pipe.vae.config.scaling_factor
        z = z.to(torch.bfloat16)

        decoded = pipe.vae.decode(z).sample
        decoded = decoded.float()

        image = (decoded.clamp(-1, 1) + 1) / 2
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]

    img_path = output_image_dir / f"sample_{sample_id:03d}.png"
    Image.fromarray((image * 255).astype(np.uint8)).save(img_path)

    # Add image path to data
    data["image_path"] = str(img_path)
    data["latent_path"] = str(output_latents_dir / f"sample_{sample_id:03d}_block22.npy")
    data["noisypred_path"] = str(output_noisy_pred_dir / f"sample_{sample_id:03d}_noisepred.npy")

    print(f"Saved final image to: {img_path}")

hook_handle.remove()

# Save metadata
with open(output_labels_path, 'w') as f:
    json.dump(all_data, f, indent=2)

print(f"\n{'='*60}")
print("All samples generated successfully!")
print(f"Total samples: {len(all_data)}")
print(f"Labels saved to: {output_labels_path}")
print('='*60)
