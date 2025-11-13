import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
import pandas as pd

#======================== Configuration ===============================
ModelID = "stabilityai/stable-diffusion-3.5-medium"
num_inference_steps = 10
target_idx = 8   # capture diffusion step 8
guidance_scale = 7.0 # guidance scale for the CFG

# Path to save the latents
output_latents_dir = Path("TrainingData/latents/eg")
output_latents_dir.mkdir(parents=True, exist_ok=True)

output_noisy_pred_dir = Path("TrainingData/noisy_pred/eg")
output_noisy_pred_dir.mkdir(parents=True, exist_ok=True)

output_image_dir = Path("TrainingData/images/eg")
output_image_dir.mkdir(parents=True, exist_ok=True)

prompts = [
    "a cozy reading nook with warm light",
    "cyberpunk city street at night, rain",
    "a watercolor painting of mountains",
    "intricate steampunk airship in the sky",
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
# Process each prompt
# ============================================================
for prompt_idx, prompt in enumerate(prompts):
    print(f"\nProcessing prompt {prompt_idx + 1}: {prompt}")

    captured_block22.clear()

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
    H, W = 256, 256
    latents = torch.randn(
        (1, 16, H // 8, W // 8),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    pipe.scheduler.set_timesteps(num_inference_steps)

    # ==================== Denoising Loop ====================
    noisy_pred_saved = False

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
            print(f"\nCaptured block 22 output at step {i} for prompt {prompt_idx+1}")

            block22_output = captured_block22[-1]  # on CPU already
            print("Block 22 activation shape:", block22_output.shape)

            uncond_h, cond_h = block22_output.chunk(2)
            cond_np = cond_h.numpy()

            save_path = output_latents_dir / f"prompt_{prompt_idx + 1}_step{target_idx}_block22.npy"
            np.save(save_path, cond_np)

            print(f"Saved block 22 hidden-state to: {save_path}")

            # ---- Save noisy prediction too ----
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cpu = noise_pred_text.detach().float().cpu().numpy()

            noisy_save_path = output_noisy_pred_dir / f"prompt_{prompt_idx+1}_step{target_idx}_noisepred.npy"
            np.save(noisy_save_path, noise_pred_cpu)

            print(f"Saved noise prediction to: {noisy_save_path}")
            noisy_pred_saved = True

        # ====================== CFG update ========================
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

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

    img_path = output_image_dir / f"prompt_{prompt_idx+1}.png"
    from PIL import Image
    Image.fromarray((image * 255).astype(np.uint8)).save(img_path)

    print(f"Saved final image to: {img_path}")

hook_handle.remove()
print("\nAll Block 22 activations + noisy preds + final images saved!")
