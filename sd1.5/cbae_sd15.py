"""
Concept Bottleneck Autoencoder for Stable Diffusion 1.5.

This module implements the CB-AE architecture that can be inserted into
the UNet of Stable Diffusion 1.5 for interpretable image generation.

The CB-AE operates on the bottleneck features of the UNet:
- g1: UNet encoder (down_blocks) -> bottleneck features
- CB-AE: bottleneck features -> concepts -> reconstructed features  
- g2: UNet decoder (up_blocks) -> noise prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from typing import Optional, Tuple, List, Dict, Any
import math


class ConceptBottleneckAE(nn.Module):
    """
    Concept Bottleneck Autoencoder for SD1.5 UNet bottleneck features.
    
    The UNet bottleneck in SD1.5 has shape [B, 1280, H/64, W/64] for 512x512 images,
    which becomes [B, 1280, 8, 8] = [B, 81920] when flattened.
    
    For 256x256 images: [B, 1280, 4, 4] = [B, 20480]
    """
    
    def __init__(
        self,
        bottleneck_channels: int = 1280,
        bottleneck_spatial: int = 8,  # 8x8 for 512x512, 4x4 for 256x256
        n_concepts: int = 8,
        concept_dims: List[int] = None,  # Dimensions per concept (2 for binary)
        unsupervised_dim: int = 64,
        hidden_dim: int = 1024,
        num_layers: int = 4,
    ):
        super().__init__()
        
        self.bottleneck_channels = bottleneck_channels
        self.bottleneck_spatial = bottleneck_spatial
        self.n_concepts = n_concepts
        self.unsupervised_dim = unsupervised_dim
        
        # Default to binary concepts
        if concept_dims is None:
            concept_dims = [2] * n_concepts
        self.concept_dims = concept_dims
        
        # Total concept dimension = sum of all concept dims + unsupervised
        self.total_concept_dim = sum(concept_dims) + unsupervised_dim
        
        # Input dimension from flattened bottleneck
        self.input_dim = bottleneck_channels * bottleneck_spatial * bottleneck_spatial
        
        # Encoder: bottleneck features -> concepts
        encoder_layers = []
        in_dim = self.input_dim
        for i in range(num_layers - 1):
            out_dim = hidden_dim
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_dim = out_dim
        encoder_layers.append(nn.Linear(in_dim, self.total_concept_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: concepts -> bottleneck features
        decoder_layers = []
        in_dim = self.total_concept_dim
        for i in range(num_layers - 1):
            out_dim = hidden_dim
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_dim = out_dim
        decoder_layers.append(nn.Linear(in_dim, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode bottleneck features to concept space.
        
        Args:
            x: Bottleneck features [B, C, H, W]
            
        Returns:
            concepts: Concept predictions [B, total_concept_dim]
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        concepts = self.encoder(x_flat)
        return concepts
    
    def decode(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Decode concepts back to bottleneck features.
        
        Args:
            concepts: Concept vector [B, total_concept_dim]
            
        Returns:
            x_recon: Reconstructed bottleneck features [B, C, H, W]
        """
        batch_size = concepts.shape[0]
        x_flat = self.decoder(concepts)
        x_recon = x_flat.view(
            batch_size, 
            self.bottleneck_channels, 
            self.bottleneck_spatial, 
            self.bottleneck_spatial
        )
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.
        
        Returns:
            concepts: Predicted concepts
            x_recon: Reconstructed bottleneck features
        """
        concepts = self.encode(x)
        x_recon = self.decode(concepts)
        return concepts, x_recon
    
    def get_concept_predictions(self, concepts: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract per-concept predictions from the full concept vector.
        
        Args:
            concepts: Full concept vector [B, total_concept_dim]
            
        Returns:
            List of tensors, one per concept with shape [B, concept_dim]
        """
        predictions = []
        idx = 0
        for dim in self.concept_dims:
            predictions.append(concepts[:, idx:idx+dim])
            idx += dim
        return predictions
    
    def get_concept_probabilities(self, concepts: torch.Tensor) -> List[torch.Tensor]:
        """
        Get softmax probabilities for each concept.
        """
        predictions = self.get_concept_predictions(concepts)
        return [F.softmax(pred, dim=1) for pred in predictions]


class SD15WithCBAE(nn.Module):
    """
    Stable Diffusion 1.5 with Concept Bottleneck Autoencoder.
    
    This class wraps the SD1.5 UNet and inserts a CB-AE at the bottleneck.
    The UNet encoder (g1) and decoder (g2) are frozen, only CB-AE is trained.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        n_concepts: int = 8,
        concept_dims: List[int] = None,
        unsupervised_dim: int = 64,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        img_size: int = 256,
        max_timestep: int = 400,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.device = device
        self.n_concepts = n_concepts
        self.max_timestep = max_timestep
        self.img_size = img_size
        
        # Calculate spatial size at bottleneck
        # SD1.5 downsamples by factor of 8 in VAE, then by 8 in UNet encoder
        # Total: 64x downsampling from image to UNet bottleneck
        self.bottleneck_spatial = img_size // 64
        if self.bottleneck_spatial < 1:
            self.bottleneck_spatial = 1
            
        print(f"Image size: {img_size}, Bottleneck spatial: {self.bottleneck_spatial}")
        
        # Load SD1.5 pipeline
        print(f"Loading Stable Diffusion 1.5 from {model_id}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        
        # Extract components
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        
        # Freeze all SD components
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Setup noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        
        # Move SD components to device BEFORE creating null embedding
        # This fixes the device mismatch error
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(device)
        self.text_encoder = self.text_encoder.to(device)
        
        # Get null text embedding for unconditional generation
        self._setup_null_embedding()
        
        # Create CB-AE
        # UNet bottleneck has 1280 channels
        self.cbae = ConceptBottleneckAE(
            bottleneck_channels=1280,
            bottleneck_spatial=self.bottleneck_spatial,
            n_concepts=n_concepts,
            concept_dims=concept_dims,
            unsupervised_dim=unsupervised_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
        # Move CB-AE to device
        self.cbae = self.cbae.to(device)
        
        # Store concept dimensions for loss computation
        self.concept_dims = concept_dims if concept_dims else [2] * n_concepts
        
    def _setup_null_embedding(self):
        """Create null text embedding for unconditional denoising."""
        with torch.no_grad():
            # Empty prompt
            text_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            self.null_text_embedding = self.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]
    
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts to embeddings.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            text_embeddings: [B, seq_len, hidden_dim]
        """
        with torch.no_grad():
            text_input = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]
        return text_embeddings
            
    def to(self, device):
        """Move model to device."""
        self.device = device
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.cbae = self.cbae.to(device)
        if hasattr(self, 'null_text_embedding'):
            self.null_text_embedding = self.null_text_embedding.to(device)
        return self
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to VAE latent space.
        
        Args:
            images: Images in range [-1, 1], shape [B, 3, H, W]
            
        Returns:
            latents: VAE latents, shape [B, 4, H/8, W/8]
        """
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode VAE latents to images.
        
        Args:
            latents: VAE latents, shape [B, 4, H/8, W/8]
            
        Returns:
            images: Images in range [-1, 1], shape [B, 3, H, W]
        """
        with torch.no_grad():
            latents = latents / self.vae.config.scaling_factor
            images = self.vae.decode(latents).sample
        return images
    
    def get_unet_bottleneck(
        self, 
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run the UNet encoder (g1) to get bottleneck features.
        
        This extracts the features at the UNet bottleneck, which is between
        the down_blocks and up_blocks.
        
        Args:
            noisy_latents: Noisy VAE latents [B, 4, H/8, W/8]
            timesteps: Diffusion timesteps [B]
            encoder_hidden_states: Text conditioning [B, seq_len, dim]
                                   If None, uses null (unconditional) embedding
            
        Returns:
            bottleneck: Bottleneck features [B, 1280, H/64, W/64]
            cache: Dictionary with intermediate values for decoder
        """
        if encoder_hidden_states is None:
            batch_size = noisy_latents.shape[0]
            encoder_hidden_states = self.null_text_embedding.repeat(batch_size, 1, 1)
        
        # Get time embedding
        t_emb = self.unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=noisy_latents.dtype)
        emb = self.unet.time_embedding(t_emb)
        
        # Initial convolution
        sample = self.unet.conv_in(noisy_latents)
        
        # Down blocks (encoder)
        down_block_res_samples = (sample,)
        for downsample_block in self.unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                )
            down_block_res_samples += res_samples
            
        # Mid block (bottleneck)
        if self.unet.mid_block is not None:
            sample = self.unet.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        
        # Cache for decoder
        cache = {
            "down_block_res_samples": down_block_res_samples,
            "emb": emb,
            "encoder_hidden_states": encoder_hidden_states,
        }
        
        return sample, cache
    
    def run_unet_decoder(
        self,
        bottleneck: torch.Tensor,
        cache: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Run the UNet decoder (g2) from bottleneck features.
        
        Args:
            bottleneck: Bottleneck features [B, 1280, H/64, W/64]
            cache: Dictionary with intermediate values from encoder
            
        Returns:
            noise_pred: Predicted noise [B, 4, H/8, W/8]
        """
        sample = bottleneck
        down_block_res_samples = cache["down_block_res_samples"]
        emb = cache["emb"]
        encoder_hidden_states = cache["encoder_hidden_states"]
        
        # Up blocks (decoder)
        for i, upsample_block in enumerate(self.unet.up_blocks):
            # Get skip connections
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )
        
        # Final layers
        if self.unet.conv_norm_out is not None:
            sample = self.unet.conv_norm_out(sample)
            sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        
        return sample
    
    def forward_with_cbae(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through UNet with CB-AE at bottleneck.
        
        Args:
            noisy_latents: Noisy VAE latents [B, 4, H/8, W/8]
            timesteps: Diffusion timesteps [B]
            encoder_hidden_states: Text conditioning (optional)
            
        Returns:
            noise_pred: Predicted noise from reconstructed bottleneck
            concepts: Predicted concepts
            bottleneck_recon: Reconstructed bottleneck features
        """
        # Get bottleneck features (g1)
        with torch.no_grad():
            bottleneck, cache = self.get_unet_bottleneck(
                noisy_latents, timesteps, encoder_hidden_states
            )
        
        # Apply CB-AE
        concepts, bottleneck_recon = self.cbae(bottleneck)
        
        # Run decoder (g2)
        with torch.no_grad():
            noise_pred = self.run_unet_decoder(bottleneck_recon, cache)
        
        return noise_pred, concepts, bottleneck_recon, bottleneck
    
    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to latents according to noise schedule."""
        return self.noise_scheduler.add_noise(latents, noise, timesteps)
    
    def predict_x0_from_noise(
        self,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict clean latents x0 from noisy latents and noise prediction.
        
        Uses the DDPM formula: x0 = (xt - sqrt(1-alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        """
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        alpha_prod_t = alpha_prod_t.to(noisy_latents.device, noisy_latents.dtype)
        beta_prod_t = 1 - alpha_prod_t
        
        pred_x0 = (noisy_latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        return pred_x0
    
    @torch.no_grad()
    def predict_concepts_from_image(
        self,
        images: torch.Tensor,
        text_prompt: str = None,
    ) -> torch.Tensor:
        """
        Predict concepts for input images.
        
        This is for inference - predicting concepts on real or generated images.
        
        Args:
            images: Images in range [-1, 1], shape [B, 3, H, W]
            text_prompt: Optional text prompt (uses null embedding if None)
            
        Returns:
            concepts: Predicted concepts [B, total_concept_dim]
        """
        batch_size = images.shape[0]
        
        # Encode to VAE latents
        latents = self.encode_images(images)
        
        # Get text embeddings
        if text_prompt is not None:
            text_embeddings = self.encode_text([text_prompt] * batch_size)
        else:
            text_embeddings = None
        
        # Use timestep 0 (clean image)
        timesteps = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        
        # Get bottleneck features
        bottleneck, _ = self.get_unet_bottleneck(latents, timesteps, text_embeddings)
        
        # Predict concepts
        concepts = self.cbae.encode(bottleneck)
        
        return concepts


def get_concept_index(model: SD15WithCBAE, concept_idx: int) -> Tuple[int, int]:
    """
    Get the start and end indices for a concept in the concept vector.
    
    Args:
        model: SD15WithCBAE model
        concept_idx: Index of the concept
        
    Returns:
        start_idx, end_idx: Indices into the concept vector
    """
    start_idx = sum(model.concept_dims[:concept_idx])
    end_idx = start_idx + model.concept_dims[concept_idx]
    return start_idx, end_idx