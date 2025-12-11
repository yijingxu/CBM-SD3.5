"""
DDPM-specific wrapper for CB-AE/CC training and inference.

This module handles the specific requirements of training CB-AE with DDPM models,
including noising/denoising process, early timestep selection, and noise prediction.
"""

import torch
from torch import nn
from typing import Tuple, Optional
from models.basic import Basic
from models.cbae_core import build_cbae_for_ddpm, build_cc_for_ddpm
from models.cbae_unet2d import UNet2DModel


class cbAE_DDPM_Trainable(Basic):
    """
    CB-AE wrapper for DDPM that handles training-specific logic.

    Key differences from GAN training:
    1. Uses noisy latents w_t from early timesteps (t <= max_timestep)
    2. Pseudo-labels come from CLEAN images x_0, not generated images
    3. Includes DDPM noise prediction loss
    4. Only applies CB-AE during early timesteps
    """

    def _build_model(self):
        """Build DDPM UNet and CB-AE components."""
        pretrained_model_path = self.config['model']['pretrained']
        print(f'Loading DDPM from {pretrained_model_path}')
        self.gen = UNet2DModel.from_pretrained(pretrained_model_path)

        # Build CB-AE using core module
        self.cbae = build_cbae_for_ddpm(
            latent_shape=self.config['model']['latent_shape'],
            hidden_dim=self.config['model']['latent_noise_dim'],
            concept_dim=sum(self.concepts_output)
        )

        # Store configuration
        self.latent_shape = self.config['model']['latent_shape']
        self.max_timestep = self.config['model'].get('max_timestep', 400)
        print(f'CB-AE will be applied only for t <= {self.max_timestep}')

    def forward_with_cbae(
        self,
        noisy_sample: torch.Tensor,
        timestep: torch.Tensor,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with CB-AE intervention.

        Args:
            noisy_sample: Noisy image x_t of shape [B, C, H, W]
            timestep: Current timestep t
            return_components: If True, return intermediate components

        Returns:
            noise_pred: Predicted noise epsilon_theta
            components: Optional dict with intermediate values
        """
        # Get noisy latent from UNet encoder (g1)
        latent_t, t_emb, unet_residual = self.gen.forward_part1(noisy_sample, timestep)

        # Pass through CB-AE
        concepts = self.cbae.encode(latent_t)  # Encode to concept space
        recon_latent_t = self.cbae.decode(concepts)  # Decode back

        # Predict noise using reconstructed latent (g2)
        noise_pred = self.gen.forward_part2(
            recon_latent_t,
            emb=t_emb,
            down_block_res_samples=unet_residual,
            return_dict=False
        )

        if return_components:
            components = {
                'latent_t': latent_t,
                'concepts': concepts,
                'recon_latent_t': recon_latent_t,
                't_emb': t_emb,
                'unet_residual': unet_residual
            }
            return noise_pred, components
        else:
            return noise_pred, None

    def forward_without_cbae(
        self,
        noisy_sample: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard DDPM forward without CB-AE.

        Used for timesteps > max_timestep where image is too noisy.
        """
        return self.gen(noisy_sample, timestep, return_dict=False)

    def get_noisy_latent(
        self,
        noisy_sample: torch.Tensor,
        timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Extract noisy latent from UNet encoder.

        Args:
            noisy_sample: Noisy image x_t
            timestep: Current timestep t

        Returns:
            latent_t: Noisy latent from UNet mid-block
            t_emb: Time embedding
            unet_residual: Residual connections from down blocks
        """
        return self.gen.forward_part1(noisy_sample, timestep)

    def predict_noise_from_latent(
        self,
        latent: torch.Tensor,
        t_emb: torch.Tensor,
        unet_residual: Tuple
    ) -> torch.Tensor:
        """
        Predict noise from latent (UNet decoder g2).

        Args:
            latent: Latent representation
            t_emb: Time embedding
            unet_residual: Residual connections

        Returns:
            Predicted noise
        """
        return self.gen.forward_part2(
            latent,
            emb=t_emb,
            down_block_res_samples=unet_residual,
            return_dict=False
        )


class CC_DDPM_Trainable(Basic):
    """
    CC (encoder-only) wrapper for DDPM.

    Similar to CB-AE but without decoder - only predicts concepts.
    """

    def _build_model(self):
        """Build DDPM UNet and CC components."""
        pretrained_model_path = self.config['model']['pretrained']
        print(f'Loading DDPM from {pretrained_model_path}')
        self.gen = UNet2DModel.from_pretrained(pretrained_model_path)

        # Build CC using core module
        self.cbae = build_cc_for_ddpm(
            latent_shape=self.config['model']['latent_shape'],
            hidden_dim=self.config['model']['latent_noise_dim'],
            concept_dim=sum(self.concepts_output)
        )

        # Store configuration
        self.latent_shape = self.config['model']['latent_shape']
        self.max_timestep = self.config['model'].get('max_timestep', 400)
        print(f'CC will be applied only for t <= {self.max_timestep}')

    def get_noisy_latent(
        self,
        noisy_sample: torch.Tensor,
        timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Extract noisy latent from UNet encoder."""
        return self.gen.forward_part1(noisy_sample, timestep)


class DDPMNoiseSchedulerHelper:
    """
    Helper class for DDPM forward process (adding noise to clean images).

    This implements the forward diffusion process: q(x_t | x_0) for training.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        self.num_train_timesteps = num_train_timesteps

        # Generate beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # From Ho et al. (2020)
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        # Pre-compute alpha values for efficient sampling
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to clean images according to forward process.

        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_0: Clean images [B, C, H, W]
            noise: Gaussian noise [B, C, H, W]
            timesteps: Timestep indices [B]

        Returns:
            Noisy images x_t [B, C, H, W]
        """
        # Move schedule to same device as input
        device = x_0.device
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device)

        # Reshape for broadcasting: [B, 1, 1, 1]
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(x_0.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(x_0.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Add noise
        noisy_x = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

        return noisy_x

    def to(self, device):
        """Move noise schedule to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self
