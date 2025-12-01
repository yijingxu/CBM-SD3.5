# python YijingCode/experiment/sd35_understanding.py
# Test hypothesis: Transformer output is velocity prediction (rectified flow) not noise prediction

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
target_timestep_idx = 8   # Which timestep to analyze in detail

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

# Check scheduler type
print(f"\n{'='*80}")
print("SCHEDULER INFORMATION")
print(f"{'='*80}")
print(f"Scheduler type: {type(pipe.scheduler)}")
print(f"Scheduler class: {pipe.scheduler.__class__.__name__}")

# Check if it's a flow matching scheduler
if hasattr(pipe.scheduler, 'config'):
    print(f"Scheduler config: {pipe.scheduler.config}")
if hasattr(pipe.scheduler, 'prediction_type'):
    print(f"Prediction type: {pipe.scheduler.prediction_type}")

print(f"\n{'='*80}")

# ============================================================
# Single timestep analysis
# ============================================================
prompt = "A scoop of ice cream served in a crunchy cone."
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

# Storage for analysis
timestep_data = []

# ==================== Denoising Loop ====================
for i, t in enumerate(pipe.progress_bar(pipe.scheduler.timesteps)):
    # Store latents BEFORE transformer call
    latents_before = latents.clone()
    
    latent_model_input = torch.cat([latents] * 2)
    pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    with torch.no_grad():
        timestep_tensor = t.to(latents.device).long().unsqueeze(0)

        # Transformer run - this is what we're testing
        transformer_output = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep_tensor,
            encoder_hidden_states=torch.cat([negative_prompt_embeds_full, prompt_embeds_full], dim=0),
            pooled_projections=pooled,
        )
        noise_pred = transformer_output.sample  # This might be velocity, not noise!

    # Apply CFG
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred_cfg = noise_pred_uncond + 7.0 * (noise_pred_text - noise_pred_uncond)
    
    # Store latents AFTER transformer but BEFORE scheduler step
    latents_after_transformer = latents.clone()
    
    # Scheduler step - this is where we see how the output is used
    scheduler_result = pipe.scheduler.step(noise_pred_cfg, t, latents)
    latents_after_scheduler = scheduler_result.prev_sample
    
    # ====================== Detailed Analysis at Target Timestep ======================
    if i == target_timestep_idx:
        print(f"\n{'='*80}")
        print(f"DETAILED ANALYSIS AT TIMESTEP INDEX {i}")
        print(f"Timestep value: {t.item():.4f}")
        print(f"{'='*80}")
        
        # Convert to float for analysis (avoid bfloat16 precision issues)
        latents_before_f = latents_before.float()
        noise_pred_cfg_f = noise_pred_cfg.float()
        latents_after_scheduler_f = latents_after_scheduler.float()
        
        print(f"\n--- Tensor Shapes ---")
        print(f"Latents (input to transformer): {latents_before_f.shape}")
        print(f"Transformer output (noise_pred_cfg): {noise_pred_cfg_f.shape}")
        print(f"Latents (after scheduler step): {latents_after_scheduler_f.shape}")
        
        print(f"\n--- Statistical Analysis ---")
        print(f"Input latents stats:")
        print(f"  Mean: {latents_before_f.mean().item():.6f}")
        print(f"  Std: {latents_before_f.std().item():.6f}")
        print(f"  Min: {latents_before_f.min().item():.6f}")
        print(f"  Max: {latents_before_f.max().item():.6f}")
        
        print(f"\nTransformer output (noise_pred_cfg) stats:")
        print(f"  Mean: {noise_pred_cfg_f.mean().item():.6f}")
        print(f"  Std: {noise_pred_cfg_f.std().item():.6f}")
        print(f"  Min: {noise_pred_cfg_f.min().item():.6f}")
        print(f"  Max: {noise_pred_cfg_f.max().item():.6f}")
        
        print(f"\nOutput latents (after scheduler) stats:")
        print(f"  Mean: {latents_after_scheduler_f.mean().item():.6f}")
        print(f"  Std: {latents_after_scheduler_f.std().item():.6f}")
        print(f"  Min: {latents_after_scheduler_f.min().item():.6f}")
        print(f"  Max: {latents_after_scheduler_f.max().item():.6f}")
        
        # ========== KEY TEST: Is transformer output a velocity prediction? ==========
        print(f"\n{'='*80}")
        print("HYPOTHESIS TEST: Is transformer output a VELOCITY prediction?")
        print(f"{'='*80}")
        
        # In rectified flow matching:
        # x_{t_next} = x_t + (t_next - t) * v_θ(x_t, t)
        # So: v_θ = (x_{t_next} - x_t) / (t_next - t)
        # Note: In flow matching, we typically go from t=1 (noise) to t=0 (data)
        # So t_next < t, and dt = t_next - t is negative
        
        # Get next timestep (which is actually smaller in value for denoising)
        if i < len(pipe.scheduler.timesteps) - 1:
            t_next = pipe.scheduler.timesteps[i + 1]
            dt = t_next.item() - t.item()  # This will be negative (going from high to low timestep)
            
            print(f"\nTimestep difference: dt = {t_next.item():.4f} - {t.item():.4f} = {dt:.4f}")
            
            # Compute actual change in latents
            # latents_after_scheduler is the latents at the NEXT timestep (after scheduler step)
            actual_change = latents_after_scheduler_f - latents_before_f
            
            print(f"\nActual change in latents (x_t_next - x_t):")
            print(f"  Shape: {actual_change.shape}")
            print(f"  Mean: {actual_change.mean().item():.6f}")
            print(f"  Std: {actual_change.std().item():.6f}")
            print(f"  Note: This is the change from timestep {t.item():.4f} to {t_next.item():.4f}")
            
            # If transformer output is velocity, then:
            # actual_change ≈ dt * transformer_output
            # where dt = t_next - t (negative for denoising)
            predicted_change_from_velocity = dt * noise_pred_cfg_f
            
            print(f"\nPredicted change if transformer output is VELOCITY (dt * transformer_output):")
            print(f"  Shape: {predicted_change_from_velocity.shape}")
            print(f"  Mean: {predicted_change_from_velocity.mean().item():.6f}")
            print(f"  Std: {predicted_change_from_velocity.std().item():.6f}")
            
            # Compare actual vs predicted
            difference = actual_change - predicted_change_from_velocity
            relative_error = torch.abs(difference) / (torch.abs(actual_change) + 1e-8)
            
            print(f"\nComparison (actual_change vs dt * transformer_output):")
            print(f"  Absolute difference mean: {difference.mean().item():.6f}")
            print(f"  Absolute difference std: {difference.std().item():.6f}")
            print(f"  Relative error mean: {relative_error.mean().item():.100f}")
            print(f"  Relative error median: {relative_error.median().item():.100f}")
            print(f"  Relative error max: {relative_error.max().item():.100f}")
            
            # Correlation analysis
            actual_flat = actual_change.flatten()
            predicted_flat = predicted_change_from_velocity.flatten()
            correlation = torch.corrcoef(torch.stack([actual_flat, predicted_flat]))[0, 1]
            print(f"  Correlation coefficient: {correlation.item():.6f}")
            
            # Check if they're approximately equal (within some tolerance)
            tolerance = 0.1
            close_match = torch.abs(difference) < tolerance * torch.abs(actual_change)
            match_percentage = (close_match.float().mean() * 100).item()
            print(f"  Percentage of values within {tolerance*100}% tolerance: {match_percentage:.2f}%")
            
            # ========== Alternative: Is it a noise prediction? ==========
            print(f"\n{'='*80}")
            print("ALTERNATIVE TEST: Is transformer output a NOISE prediction?")
            print(f"{'='*80}")
            
            # In DDPM-style diffusion:
            # The scheduler would use noise prediction differently
            # Let's check what the scheduler actually does
            print(f"\nScheduler step function analysis:")
            print(f"  Input: noise_pred_cfg shape {noise_pred_cfg_f.shape}, timestep {t.item():.4f}, latents shape {latents_before_f.shape}")
            print(f"  Output: prev_sample shape {latents_after_scheduler_f.shape}")
            
            # Check scheduler internals if available
            if hasattr(scheduler_result, 'pred_original_sample'):
                print(f"  Scheduler also returned pred_original_sample: {scheduler_result.pred_original_sample.shape}")
            
            # For rectified flow, the update is typically:
            # x_{t_next} = x_t + dt * v_θ
            # where dt = t_next - t (negative for denoising)
            # So if transformer output is velocity, we should see this relationship
            
            # Compute what velocity would be if we reverse-engineer from the change
            if abs(dt) > 1e-8:
                reverse_engineered_velocity = actual_change / dt
                print(f"\nReverse-engineered velocity (actual_change / dt):")
                print(f"  Mean: {reverse_engineered_velocity.mean().item():.6f}")
                print(f"  Std: {reverse_engineered_velocity.std().item():.6f}")
                print(f"  Min: {reverse_engineered_velocity.min().item():.6f}")
                print(f"  Max: {reverse_engineered_velocity.max().item():.6f}")
                
                # Compare to transformer output
                vel_diff = reverse_engineered_velocity - noise_pred_cfg_f
                vel_relative_error = torch.abs(vel_diff) / (torch.abs(reverse_engineered_velocity) + 1e-8)
                print(f"\nComparison (reverse_engineered_velocity vs transformer_output):")
                print(f"  Absolute difference mean: {vel_diff.mean().item():.6f}")
                print(f"  Absolute difference std: {vel_diff.std().item():.6f}")
                print(f"  Relative error mean: {vel_relative_error.mean().item():.100f}")
                print(f"  Relative error median: {vel_relative_error.median().item():.100f}")
                
                vel_correlation = torch.corrcoef(torch.stack([reverse_engineered_velocity.flatten(), noise_pred_cfg_f.flatten()]))[0, 1]
                print(f"  Correlation coefficient: {vel_correlation.item():.6f}")
                
                # Check if they match closely
                vel_close_match = torch.abs(vel_diff) < 0.1 * torch.abs(reverse_engineered_velocity)
                vel_match_percentage = (vel_close_match.float().mean() * 100).item()
                print(f"  Percentage of values within 10% tolerance: {vel_match_percentage:.2f}%")
                
                # ========== Calculate Scaling Factor ==========
                print(f"\n{'='*80}")
                print("SCALING FACTOR ANALYSIS")
                print(f"{'='*80}")
                
                # The transformer output appears to be scaled velocity
                # Calculate the scaling factor: transformer_output / reverse_engineered_velocity
                # But we need to avoid division by zero
                non_zero_mask = torch.abs(reverse_engineered_velocity) > 1e-8
                if non_zero_mask.sum() > 0:
                    scaling_factors = noise_pred_cfg_f[non_zero_mask] / reverse_engineered_velocity[non_zero_mask]
                    print(f"\nScaling factor (transformer_output / reverse_engineered_velocity):")
                    print(f"  Mean: {scaling_factors.mean().item():.6f}")
                    print(f"  Median: {scaling_factors.median().item():.6f}")
                    print(f"  Std: {scaling_factors.std().item():.6f}")
                    print(f"  Min: {scaling_factors.min().item():.6f}")
                    print(f"  Max: {scaling_factors.max().item():.6f}")
                    print(f"\n  Interpretation: transformer_output ≈ {scaling_factors.median().item():.2f} × velocity")
                
                # Alternative: Check if there's a simple multiplicative relationship
                # If transformer_output = k * velocity, then transformer_output / velocity should be constant
                print(f"\n--- Verification: Check if scaling is consistent ---")
                if non_zero_mask.sum() > 0:
                    # Check coefficient of variation (std/mean) - should be small if scaling is consistent
                    cv = scaling_factors.std() / (torch.abs(scaling_factors.mean()) + 1e-8)
                    print(f"  Coefficient of variation: {cv.item():.6f}")
                    if cv.item() < 0.1:
                        print(f"  ✓ Scaling factor is CONSISTENT (low variation)")
                    else:
                        print(f"  ⚠ Scaling factor varies (may be timestep-dependent)")
                
                # Check if scaled transformer output matches reverse-engineered velocity
                if non_zero_mask.sum() > 0:
                    median_scale = scaling_factors.median()
                    scaled_transformer_output = noise_pred_cfg_f / median_scale
                    scaled_diff = scaled_transformer_output - reverse_engineered_velocity
                    scaled_relative_error = torch.abs(scaled_diff) / (torch.abs(reverse_engineered_velocity) + 1e-8)
                    print(f"\n--- After applying scaling factor ---")
                    print(f"  Scaled transformer output vs reverse-engineered velocity:")
                    print(f"    Relative error mean: {scaled_relative_error.mean().item():.100f}")
                    print(f"    Relative error median: {scaled_relative_error.median().item():.100f}")
                    scaled_correlation = torch.corrcoef(torch.stack([scaled_transformer_output.flatten(), reverse_engineered_velocity.flatten()]))[0, 1]
                    print(f"    Correlation: {scaled_correlation.item():.6f}")
        else:
            print(f"\nNote: This is the last timestep, cannot compute next timestep for comparison")
        
        # ========== Inspect scheduler step function ==========
        print(f"\n{'='*80}")
        print("SCHEDULER STEP FUNCTION INSPECTION")
        print(f"{'='*80}")
        
        # Try to understand what the scheduler does
        print(f"Scheduler class: {pipe.scheduler.__class__.__name__}")
        print(f"Scheduler step method signature:")
        import inspect
        sig = inspect.signature(pipe.scheduler.step)
        print(f"  {sig}")
        
        # Check if we can see the step implementation
        if hasattr(pipe.scheduler, 'step'):
            step_source = inspect.getsource(pipe.scheduler.step)
            # Print first few lines to understand
            lines = step_source.split('\n')[:20]
            print(f"\nFirst 20 lines of scheduler.step implementation:")
            for line in lines:
                print(f"  {line}")
        
        # Store data for this timestep
        timestep_data.append({
            'timestep_idx': i,
            'timestep_value': t.item(),
            'latents_before': latents_before_f.cpu(),
            'transformer_output': noise_pred_cfg_f.cpu(),
            'latents_after': latents_after_scheduler_f.cpu(),
        })
    
    # Update latents for next iteration
    latents = latents_after_scheduler

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

# ========== Summary ==========
print(f"\n--- Summary ---")
print(f"Analyzed timestep {target_timestep_idx} out of {num_inference_steps} total steps")
print(f"\nKey findings:")
print(f"1. Transformer output shape matches input latents shape")
print(f"2. Check correlation between actual_change and dt * transformer_output")
print(f"3. If correlation is high (>0.9), transformer is likely predicting VELOCITY")
print(f"4. If correlation is low, transformer might be predicting something else (e.g., noise)")

