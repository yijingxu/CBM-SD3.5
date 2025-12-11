"""
Core CB-AE/CC modules shared across different generative model backends (GAN, DDPM, etc.).

This module provides the fundamental concept bottleneck autoencoder (CB-AE) and
concept controller (CC) architectures that are backend-agnostic.
"""

import torch
from torch import nn


def _weights_init(m):
    """Initialize weights for Conv, Linear, and BatchNorm layers."""
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)


class ConceptBottleneckCore(nn.Module):
    """
    Core Concept Bottleneck module with encoder and optional decoder.

    This is a backend-agnostic implementation that can be used with different
    generative models (GANs, DDPMs, etc.) by wrapping it appropriately.

    Args:
        input_dim: Dimension of input latent/feature vector
        hidden_dim: Dimension of hidden layers
        concept_dim: Dimension of concept vector (sum of all concept bins)
        num_layers: Number of layers in encoder/decoder (default: 4)
        use_decoder: Whether to include decoder (CB-AE) or encoder-only (CC)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        concept_dim: int,
        num_layers: int = 4,
        use_decoder: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.concept_dim = concept_dim
        self.num_layers = num_layers
        self.use_decoder = use_decoder

        # Build encoder
        self.encoder = self._build_mlp(input_dim, hidden_dim, concept_dim, num_layers)

        # Build decoder if needed
        if use_decoder:
            self.decoder = self._build_mlp(concept_dim, hidden_dim, input_dim, num_layers)
        else:
            self.decoder = None

        total_layers = len(self.encoder) + (len(self.decoder) if self.decoder else 0)
        print(f'CB-AE/CC initialized with {total_layers} total layers '
              f'(encoder: {len(self.encoder)}, decoder: {len(self.decoder) if self.decoder else 0})')

        self.apply(_weights_init)

    def _build_mlp(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        """Build MLP with LeakyReLU and BatchNorm."""
        layers = []

        for i in range(num_layers - 1):
            layer_in = in_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(layer_in, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(hidden_dim),
            ])

        # Final layer without activation
        layer_in = in_dim if num_layers == 1 else hidden_dim
        layers.append(nn.Linear(layer_in, out_dim))

        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to concept space.

        Args:
            x: Input tensor of shape [batch, input_dim]

        Returns:
            Concept vector of shape [batch, concept_dim]
        """
        return self.encoder(x)

    def decode(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Decode concepts back to latent space.

        Args:
            concepts: Concept tensor of shape [batch, concept_dim]

        Returns:
            Reconstructed latent of shape [batch, input_dim]
        """
        if not self.use_decoder:
            raise RuntimeError("Decoder not available (use_decoder=False)")
        return self.decoder(concepts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass (encode + decode).

        Args:
            x: Input tensor of shape [batch, input_dim]

        Returns:
            Reconstructed input of shape [batch, input_dim]
        """
        concepts = self.encode(x)
        if self.use_decoder:
            return self.decode(concepts)
        else:
            return concepts


class FlattenWrapper(nn.Module):
    """
    Wrapper that flattens/unflattens tensors for ConceptBottleneckCore.

    This handles converting between spatial features (e.g., [B, C, H, W])
    and flat vectors required by the MLP-based core.

    Args:
        core: ConceptBottleneckCore instance
        spatial_shape: Original spatial shape (e.g., [512, 8, 8] for DDPM)
    """

    def __init__(self, core: ConceptBottleneckCore, spatial_shape: tuple):
        super().__init__()
        self.core = core
        self.spatial_shape = spatial_shape

        # Validate dimensions
        flat_dim = 1
        for dim in spatial_shape:
            flat_dim *= dim
        assert flat_dim == core.input_dim, \
            f"Spatial shape {spatial_shape} flattens to {flat_dim}, " \
            f"but core expects {core.input_dim}"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial features to concepts.

        Args:
            x: Input of shape [batch, *spatial_shape]

        Returns:
            Concepts of shape [batch, concept_dim]
        """
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        return self.core.encode(x_flat)

    def decode(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Decode concepts to spatial features.

        Args:
            concepts: Concepts of shape [batch, concept_dim]

        Returns:
            Spatial features of shape [batch, *spatial_shape]
        """
        batch_size = concepts.shape[0]
        x_flat = self.core.decode(concepts)
        return x_flat.reshape(batch_size, *self.spatial_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode pass."""
        concepts = self.encode(x)
        if self.core.use_decoder:
            return self.decode(concepts)
        else:
            return concepts


class RepeatWrapper(nn.Module):
    """
    Wrapper for StyleGAN2-like models where latent is repeated across layers.

    Args:
        core: ConceptBottleneckCore instance
        num_repeats: Number of times latent is repeated (e.g., 14 for StyleGAN2)
    """

    def __init__(self, core: ConceptBottleneckCore, num_repeats: int):
        super().__init__()
        self.core = core
        self.num_repeats = num_repeats

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode repeated latent to concepts.

        Args:
            x: Input of shape [batch, num_repeats, latent_dim]

        Returns:
            Concepts of shape [batch, concept_dim]
        """
        # Average across repeats
        x_avg = torch.mean(x, dim=1)
        return self.core.encode(x_avg)

    def decode(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Decode concepts to repeated latent.

        Args:
            concepts: Concepts of shape [batch, concept_dim]

        Returns:
            Repeated latent of shape [batch, num_repeats, latent_dim]
        """
        x = self.core.decode(concepts)
        return x.unsqueeze(1).repeat(1, self.num_repeats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode pass."""
        concepts = self.encode(x)
        if self.core.use_decoder:
            return self.decode(concepts)
        else:
            return concepts


def build_cbae_for_stylegan(noise_dim: int, concept_dim: int, num_ws: int = 14) -> RepeatWrapper:
    """
    Build CB-AE for StyleGAN2 models.

    Args:
        noise_dim: Dimension of latent vector
        concept_dim: Total concept dimension
        num_ws: Number of latent repeats in StyleGAN2

    Returns:
        RepeatWrapper with CB-AE core
    """
    core = ConceptBottleneckCore(
        input_dim=noise_dim,
        hidden_dim=noise_dim,
        concept_dim=concept_dim,
        num_layers=4,
        use_decoder=True
    )
    return RepeatWrapper(core, num_ws)


def build_cbae_for_ddpm(latent_shape: tuple, hidden_dim: int, concept_dim: int) -> FlattenWrapper:
    """
    Build CB-AE for DDPM models.

    Args:
        latent_shape: Shape of UNet latent (e.g., [512, 8, 8])
        hidden_dim: Hidden dimension for MLP
        concept_dim: Total concept dimension

    Returns:
        FlattenWrapper with CB-AE core
    """
    flat_dim = 1
    for dim in latent_shape:
        flat_dim *= dim

    core = ConceptBottleneckCore(
        input_dim=flat_dim,
        hidden_dim=hidden_dim,
        concept_dim=concept_dim,
        num_layers=4,
        use_decoder=True
    )
    return FlattenWrapper(core, latent_shape)


def build_cc_for_stylegan(noise_dim: int, concept_dim: int, num_ws: int = 14) -> RepeatWrapper:
    """Build CC (encoder-only) for StyleGAN2."""
    core = ConceptBottleneckCore(
        input_dim=noise_dim,
        hidden_dim=noise_dim,
        concept_dim=concept_dim,
        num_layers=4,
        use_decoder=False
    )
    return RepeatWrapper(core, num_ws)


def build_cc_for_ddpm(latent_shape: tuple, hidden_dim: int, concept_dim: int) -> FlattenWrapper:
    """Build CC (encoder-only) for DDPM."""
    flat_dim = 1
    for dim in latent_shape:
        flat_dim *= dim

    core = ConceptBottleneckCore(
        input_dim=flat_dim,
        hidden_dim=hidden_dim,
        concept_dim=concept_dim,
        num_layers=4,
        use_decoder=False
    )
    return FlattenWrapper(core, latent_shape)
