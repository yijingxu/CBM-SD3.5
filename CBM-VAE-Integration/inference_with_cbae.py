"""
Inference script for SD3.5 with Concept Bottleneck Model
Supports concept interventions during generation
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from typing import Dict, Optional

from diffusers import StableDiffusion3Pipeline
from models.cbae_sd35 import ConceptBottleneckAE


class SD35WithCBAE:
    """
    SD3.5 pipeline with integrated Concept Bottleneck AutoEncoder
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-medium",
        cbae_checkpoint: str = None,
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
        target_block_idx: int = 22,
    ):
        """
        Args:
            model_id: HuggingFace model ID for SD3.5
            cbae_checkpoint: Path to trained CB-AE checkpoint
            device: Device to run on
            torch_dtype: Data type for model
            target_block_idx: Which transformer block to insert CB-AE (default: 22)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.target_block_idx = target_block_idx

        print(f"Using device: {self.device}")

        # Load HF credentials
        from apps.env_utils import get_env_var
        HF_TOKEN = get_env_var("HF_TOKEN")

        # Note: No need to explicitly login if token is in ~/.huggingface/token

        # Load SD3.5 pipeline
        print("Loading SD3.5 pipeline...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        self.pipe = self.pipe.to(self.device)

        # Load CB-AE model
        if cbae_checkpoint is not None:
            print(f"Loading CB-AE checkpoint from {cbae_checkpoint}...")
            self.cbae = self._load_cbae(cbae_checkpoint)
            self.cbae = self.cbae.to(self.device).eval()
            print("CB-AE loaded successfully!")
        else:
            self.cbae = None
            print("No CB-AE checkpoint provided. Running without concept bottleneck.")

        # Storage for captured activations
        self.captured_activations = []
        self.hook_handle = None

        # Intervention configuration
        self.interventions = None

    def _load_cbae(self, checkpoint_path: str) -> ConceptBottleneckAE:
        """Load CB-AE model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get concept config from checkpoint
        concept_config = checkpoint.get('concept_config', {
            'concept_names': ['smiling', 'glasses'],
            'concept_dims': [1, 1],
            'concept_types': ['binary', 'binary'],
        })

        # Get model hyperparameters from checkpoint if available
        # Defaults match the trained model: hidden_dim=512, 3 layers
        hidden_dim = checkpoint.get('hidden_dim', 512)
        num_encoder_layers = checkpoint.get('num_encoder_layers', 3)
        num_decoder_layers = checkpoint.get('num_decoder_layers', 3)

        # Create model with the same architecture as training
        model = ConceptBottleneckAE(
            latent_dim=511488,  # 333 * 1536
            hidden_dim=hidden_dim,
            concept_config=concept_config,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def _make_intervention_hook(self):
        """
        Create hook that applies CB-AE and interventions to block output
        """
        def hook(module, inp, out):
            # Extract hidden states
            if isinstance(out, tuple) and len(out) > 0:
                hs = out[0]
            elif isinstance(out, dict) and "hidden_states" in out:
                hs = out["hidden_states"]
            else:
                hs = out

            # If CB-AE is loaded, apply it
            if self.cbae is not None:
                with torch.no_grad():
                    original_shape = hs.shape
                    original_dtype = hs.dtype

                    # Convert to float32 for CB-AE processing
                    hs_float = hs.float()

                    # Encode to concepts
                    concepts = self.cbae.encode(hs_float)

                    # Apply interventions if specified
                    if self.interventions is not None:
                        for concept_name, value in self.interventions.items():
                            # Create intervention tensor with correct shape
                            batch_size = concepts.size(0)
                            intervention_value = torch.tensor(
                                [[value]], dtype=torch.float32, device=self.device
                            ).expand(batch_size, -1)

                            # Set concept
                            concepts = self.cbae.set_concept(
                                concepts, concept_name, intervention_value
                            )

                    # Store concepts for inspection (AFTER intervention)
                    self.captured_activations.append({
                        'concepts': concepts.detach().cpu(),
                        'original_shape': original_shape,
                    })

                    # Decode back to latent space
                    hs_modified = self.cbae.decode(concepts, original_shape)

                    # Convert back to original dtype
                    hs_modified = hs_modified.to(original_dtype)

                    # Replace hidden states
                    if isinstance(out, tuple):
                        out = (hs_modified,) + out[1:]
                    elif isinstance(out, dict):
                        out["hidden_states"] = hs_modified
                    else:
                        out = hs_modified

            return out

        return hook

    def register_hook(self):
        """Register hook to target transformer block"""
        if self.hook_handle is not None:
            self.hook_handle.remove()

        target_block = self.pipe.transformer.transformer_blocks[self.target_block_idx]
        self.hook_handle = target_block.register_forward_hook(self._make_intervention_hook())

        print(f"Registered hook to transformer block {self.target_block_idx}")

    def remove_hook(self):
        """Remove hook from transformer block"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def generate(
        self,
        prompt: str,
        interventions: Optional[Dict[str, float]] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
    ):
        """
        Generate image with optional concept interventions

        Args:
            prompt: Text prompt for generation
            interventions: Dict mapping concept names to values (e.g., {'smiling': 1.0, 'glasses': 0.0})
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            height: Image height
            width: Image width
            seed: Random seed for reproducibility

        Returns:
            image: Generated PIL Image
            concepts: Captured concept values (if CB-AE is loaded)
        """
        # Set interventions
        self.interventions = interventions

        # Clear captured activations
        self.captured_activations = []

        # Register hook
        self.register_hook()

        # Set random seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate image
        print(f"\nGenerating image with prompt: '{prompt}'")
        if interventions:
            print(f"Applying interventions: {interventions}")

        output = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

        image = output.images[0]

        # Remove hook
        self.remove_hook()

        # Extract captured concepts
        concepts = None
        if self.captured_activations:
            # Average concepts across all denoising steps
            all_concepts = torch.stack([cap['concepts'] for cap in self.captured_activations])
            concepts = all_concepts.mean(dim=0)  # Average over steps

        return image, concepts

    def intervene_and_compare(
        self,
        prompt: str,
        interventions: Dict[str, float],
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
    ):
        """
        Generate both original and intervened images for comparison

        Args:
            prompt: Text prompt
            interventions: Concept interventions to apply
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            height: Image height
            width: Image width
            seed: Random seed

        Returns:
            original_image: Image without interventions
            intervened_image: Image with interventions
            original_concepts: Concepts from original generation
            intervened_concepts: Concepts from intervened generation
        """
        # Generate original
        print("\n" + "="*60)
        print("Generating ORIGINAL image...")
        print("="*60)
        original_image, original_concepts = self.generate(
            prompt=prompt,
            interventions=None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
        )

        # Generate with interventions
        print("\n" + "="*60)
        print("Generating INTERVENED image...")
        print("="*60)
        intervened_image, intervened_concepts = self.generate(
            prompt=prompt,
            interventions=interventions,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
        )

        return original_image, intervened_image, original_concepts, intervened_concepts


def main():
    parser = argparse.ArgumentParser(description="Generate images with SD3.5 + CB-AE")

    # Model
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-3.5-medium',
                        help='HuggingFace model ID')
    parser.add_argument('--cbae_checkpoint', type=str, default='checkpoints/cbae_best.pt',
                        help='Path to CB-AE checkpoint')
    parser.add_argument('--target_block', type=int, default=22,
                        help='Target transformer block index')

    # Generation
    parser.add_argument('--prompt', type=str, default='a photo of a woman',
                        help='Text prompt for generation')
    parser.add_argument('--num_inference_steps', type=int, default=28,
                        help='Number of denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=7.0,
                        help='CFG guidance scale')
    parser.add_argument('--height', type=int, default=512,
                        help='Image height')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Interventions
    parser.add_argument('--intervene', action='store_true',
                        help='Enable intervention mode (compare original vs intervened)')
    parser.add_argument('--smiling', type=float, default=None,
                        help='Intervention value for smiling (0.0 or 1.0)')
    parser.add_argument('--glasses', type=float, default=None,
                        help='Intervention value for glasses (0.0 or 1.0)')

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save generated images')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create pipeline
    pipeline = SD35WithCBAE(
        model_id=args.model_id,
        cbae_checkpoint=args.cbae_checkpoint,
        target_block_idx=args.target_block,
    )

    # Build interventions dict
    interventions = {}
    if args.smiling is not None:
        interventions['smiling'] = args.smiling
    if args.glasses is not None:
        interventions['glasses'] = args.glasses

    # Generate images
    if args.intervene and interventions:
        # Compare original vs intervened
        original_img, intervened_img, orig_concepts, interv_concepts = pipeline.intervene_and_compare(
            prompt=args.prompt,
            interventions=interventions,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            seed=args.seed,
        )

        # Save images
        original_path = output_dir / f"original_seed{args.seed}.png"
        intervened_path = output_dir / f"intervened_seed{args.seed}.png"

        original_img.save(original_path)
        intervened_img.save(intervened_path)

        print(f"\n{'='*60}")
        print("Images saved!")
        print(f"  Original: {original_path}")
        print(f"  Intervened: {intervened_path}")

        # Print concept values
        if orig_concepts is not None:
            print(f"\nOriginal concepts:")
            print(f"  Smiling: {torch.sigmoid(orig_concepts[0, 0]).item():.3f}")
            print(f"  Glasses: {torch.sigmoid(orig_concepts[0, 1]).item():.3f}")

        if interv_concepts is not None:
            print(f"\nIntervened concepts:")
            print(f"  Smiling: {torch.sigmoid(interv_concepts[0, 0]).item():.3f}")
            print(f"  Glasses: {torch.sigmoid(interv_concepts[0, 1]).item():.3f}")

        print('='*60)

    else:
        # Single generation
        image, concepts = pipeline.generate(
            prompt=args.prompt,
            interventions=interventions if interventions else None,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            seed=args.seed,
        )

        # Save image
        output_path = output_dir / f"generated_seed{args.seed}.png"
        image.save(output_path)

        print(f"\n{'='*60}")
        print(f"Image saved to: {output_path}")

        # Print concept values
        if concepts is not None:
            print(f"\nDetected concepts:")
            print(f"  Smiling: {torch.sigmoid(concepts[0, 0]).item():.3f}")
            print(f"  Glasses: {torch.sigmoid(concepts[0, 1]).item():.3f}")

        print('='*60)


if __name__ == "__main__":
    main()
