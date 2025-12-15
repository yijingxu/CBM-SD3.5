"""
DDPM + CB-AE wrapper aligned to the paper's Objective 1-3.

This file assumes you are using the author's `models/cbae_unet2d.py` as-is:
  - UNet2DModel has forward_part1 / forward_part2
  - CB_AE defines enc(x)->concept logits and dec(c)->reconstructed latent

Key design choice (matches "insert CB-AE at intermediate location"):
  - We treat the output of UNet2DModel.forward_part1(...) as the "w" feature
    (this tensor is AFTER the UNet mid-block in the author's implementation).
  - Then we decode from CB-AE and pass to forward_part2 to predict epsilon.

Losses (paper notation):
  - Lr1: MSE(w, w') where w' = D(E(w))
  - Lr2: "image reconstruction" in the paper; for DDPM we implement an
         output-preservation proxy:
         MSE(eps_orig, eps_cbae) where eps_orig = g2(w), eps_cbae = g2(w')
         This preserves the frozen generator's output given the same input.
  - Lc: cross-entropy between pseudo-labels and concept logits E(w)

Intervention losses Li1/Li2 are implemented but can be disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbae_unet2d import UNet2DModel, CB_AE


# -----------------------------
# Utilities: concept bookkeeping
# -----------------------------

@dataclass
class ConceptSpec:
    """
    Defines the structure of your concept vector c.
    The paper uses:
      - binary concepts: 2 logits each (c_i^+, c_i^-)
      - categorical: N logits
      - optional unsupervised embedding appended at the end
    """
    types: List[str]                 # "bin" or "cat"
    sizes: List[int]                 # for "bin", size must be 2; for "cat", size=N
    names: Optional[List[str]] = None
    unsup_dim: int = 0               # optional extra dims (unsupervised embedding)

    def total_dim(self) -> int:
        return int(sum(self.sizes) + self.unsup_dim)

    def slices(self) -> List[slice]:
        out = []
        s = 0
        for k in self.sizes:
            out.append(slice(s, s + k))
            s += k
        # unsup (if any) is appended at end, not included in per-concept slices
        return out

    def validate(self):
        assert len(self.types) == len(self.sizes)
        for t, k in zip(self.types, self.sizes):
            if t == "bin":
                assert k == 2, "Binary concepts must have 2 logits (paper convention)."
            elif t == "cat":
                assert k >= 2, "Categorical concept must have >=2 logits."
            else:
                raise ValueError(f"Unknown concept type: {t}")


def intervene_logits(
    c: torch.Tensor,
    spec: ConceptSpec,
    concept_idx: int,
    target_class: Optional[int] = None,
) -> torch.Tensor:
    """
    Implements the paper's "swap logits" intervention.
      - For bin: swap the two logits
      - For cat: swap argmax logit with desired class logit

    Args:
      c: [B, C_total]
      concept_idx: which concept to intervene (0-based)
      target_class: for categorical only (0..N-1). If None, pick a random class.

    Returns:
      c_intervened: [B, C_total]
    """
    spec.validate()
    c_int = c.clone()
    slices = spec.slices()
    sl = slices[concept_idx]
    t = spec.types[concept_idx]
    k = spec.sizes[concept_idx]

    if t == "bin":
        # swap [c+, c-] -> [c-, c+]
        c_int[:, sl] = c_int[:, sl].flip(dims=[1])
        return c_int

    # categorical
    logits = c_int[:, sl]  # [B, k]
    with torch.no_grad():
        top = logits.argmax(dim=1)  # [B]
        if target_class is None:
            target_class = int(torch.randint(0, k, (1,), device=c.device).item())
        tgt = torch.full_like(top, fill_value=target_class)

    # swap per-sample (top <-> target)
    b = torch.arange(c.size(0), device=c.device)
    tmp = logits[b, top].clone()
    logits[b, top] = logits[b, tgt]
    logits[b, tgt] = tmp
    c_int[:, sl] = logits
    return c_int


# -----------------------------
# Main model: DDPM + CB-AE
# -----------------------------

class CBAE_DDPM(nn.Module):
    """
    Wrapper that inserts CB-AE between DDPM parts:
      eps = g2( D(E(w)) ) with w = g1(x_t, t)

    - UNet is frozen by default (paper trains only E,D) 
    - CB-AE (E,D) is trainable
    """

    def __init__(
        self,
        pretrained_model_id: str,
        concept_spec: ConceptSpec,
        hidden_dim: int = 1024,
        max_timestep: int = 400,
        freeze_unet: bool = True,
    ):
        super().__init__()

        concept_spec.validate()
        self.concept_spec = concept_spec
        self.max_timestep = int(max_timestep)

        print(f"[CBAE_DDPM] Loading UNet2DModel from `{pretrained_model_id}`")
        self.unet: UNet2DModel = UNet2DModel.from_pretrained(pretrained_model_id)

        if freeze_unet:
            for p in self.unet.parameters():
                p.requires_grad_(False)

        # Infer mid-latent shape by a tiny dummy forward_part1
        with torch.no_grad():
            # Always do shape inference on CPU to avoid device mismatch in Colab init
            cpu_device = torch.device("cpu")
            self.unet.to(cpu_device)

            x = torch.zeros(1, 3, 256, 256, device=cpu_device)
            t = torch.zeros(1, device=cpu_device, dtype=torch.long)

            mid, _, _ = self.unet.forward_part1(x, t)
            mid_shape = list(mid.shape[1:])  # [C,H,W]

        print(f"[CBAE_DDPM] Inferred mid-block latent shape: {mid_shape}")

        self.cbae: CB_AE = CB_AE(
            latent_shape=mid_shape,
            hidden_dim=hidden_dim,
            concept_dim=self.concept_spec.total_dim(),
        )

        # Move UNet back to GPU if available (and if you plan to use GPU)
        if torch.cuda.is_available():
            self.unet.to(torch.device("cuda"))

        print(
            f"[CBAE_DDPM] CB-AE concept_dim={self.concept_spec.total_dim()} | "
            f"mid latent_shape={mid_shape} | max_timestep={self.max_timestep}"
        )

    @torch.no_grad()
    def ddpm_eps_from_mid(self, mid: torch.Tensor, emb: torch.Tensor, down_res) -> torch.Tensor:
        return self.unet.forward_part2(mid, emb=emb, down_block_res_samples=down_res, return_dict=False)

    def forward(
        self,
        x_t: torch.Tensor,
        t: Union[torch.Tensor, int],
        use_cbae: bool = True,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
          x_t: [B,3,H,W]
          t: [B] long or int
        Returns:
          eps_pred: [B,3,H,W]
          components: dict (mid, recon_mid, c_logits, eps_orig) if requested
        """
        if isinstance(t, int):
            t_tensor = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        else:
            t_tensor = t.to(device=x_t.device, dtype=torch.long)

        t_max = int(t_tensor.max().item())
        mid, emb, down_res = self.unet.forward_part1(x_t, t_tensor)

        components = None

        # Gate CB-AE only for early timesteps (paper does early-step for diffusion variant)
        if use_cbae and t_max <= self.max_timestep:
            c_logits = self.cbae.enc(mid)               # E(w)
            recon_mid = self.cbae.dec(c_logits)         # D(E(w))

            eps_pred = self.unet.forward_part2(
                recon_mid,
                emb=emb,
                down_block_res_samples=down_res,
                return_dict=False,
            )

            if return_components:
                # eps_orig is useful for the DDPM proxy of Lr2
                with torch.no_grad():
                    eps_orig = self.unet.forward_part2(
                        mid,
                        emb=emb,
                        down_block_res_samples=down_res,
                        return_dict=False,
                    )
                components = {
                    "mid": mid,
                    "recon_mid": recon_mid,
                    "concept_logits": c_logits,
                    "eps_orig": eps_orig,
                }
        else:
            # Standard DDPM UNet forward (no CB-AE)
            eps_pred = self.unet(x_t, t_tensor, return_dict=False)

        return eps_pred, components

    # -----------------------------
    # Losses (paper Objective 1 & 2)
    # -----------------------------

    def loss_Lr1(self, mid: torch.Tensor, recon_mid: torch.Tensor) -> torch.Tensor:
        # latent reconstruction: Lr1(w, w')
        return F.mse_loss(recon_mid, mid)

    def loss_Lr2_proxy(self, eps_orig: torch.Tensor, eps_cbae: torch.Tensor) -> torch.Tensor:
        """
        Paper: Lr2(x, x') in image space 
        DDPM adaptation: preserve frozen generator output by matching epsilon predictions.
        """
        return F.mse_loss(eps_cbae, eps_orig)

    def loss_Lc(self, concept_logits: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy concept alignment (paper Objective 2) 

        Expected y_hat format:
          - If all concepts are binary with 2 logits each:
              y_hat is LongTensor [B, n_bin] with values in {0,1}
          - If you mix concepts, you must pass a list/dict externally and compute per-slice.
        """
        spec = self.concept_spec
        spec.validate()
        slices = spec.slices()

        losses = []
        for i, (t, sl) in enumerate(zip(spec.types, slices)):
            if t == "bin":
                # 2-way CE
                logits_i = concept_logits[:, sl]   # [B,2]
                target_i = y_hat[:, i].long()      # [B]
                losses.append(F.cross_entropy(logits_i, target_i))
            else:
                # cat: y_hat[:, i] in {0..K-1}
                logits_i = concept_logits[:, sl]
                target_i = y_hat[:, i].long()
                losses.append(F.cross_entropy(logits_i, target_i))

        return torch.stack(losses).mean()

    # -----------------------------
    # Intervention losses (Objective 3)
    # -----------------------------

    def intervention_step(
        self,
        mid: torch.Tensor,
        y_hat: torch.Tensor,
        concept_to_intervene: Optional[int] = None,
        target_class: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Creates c_intervened, w_intervened, c'_intervened (for Li2).
        To compute Li1 you still need pseudo-labels from M(x_intervened) externally.
        See paper Objective 3 for definitions. 
        """
        spec = self.concept_spec
        B = mid.size(0)

        c = self.cbae.enc(mid)  # [B, C_total]

        if concept_to_intervene is None:
            concept_to_intervene = int(torch.randint(0, len(spec.types), (1,), device=mid.device).item())

        c_int = intervene_logits(c, spec, concept_to_intervene, target_class=target_class)
        w_int = self.cbae.dec(c_int)
        c_prime_int = self.cbae.enc(w_int)

        # y_hat_intervened (paper) is y_hat modified at the intervened concept index
        y_hat_int = y_hat.clone()
        if spec.types[concept_to_intervene] == "bin":
            # swap 0<->1 for that concept
            y_hat_int[:, concept_to_intervene] = 1 - y_hat_int[:, concept_to_intervene]
        else:
            # if categorical and target_class provided, set it; else leave as-is
            if target_class is not None:
                y_hat_int[:, concept_to_intervene] = int(target_class)

        return {
            "concept_idx": torch.tensor([concept_to_intervene], device=mid.device),
            "c": c,
            "c_intervened": c_int,
            "w_intervened": w_int,
            "c_prime_intervened": c_prime_int,
            "y_hat_intervened": y_hat_int,
        }


# -----------------------------
# Forward diffusion helper
# -----------------------------

class DDPMNoiseSchedulerHelper:
    """
    Implements q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps
    (used for dataset-based training; not required if you train purely from generation).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        self.num_train_timesteps = num_train_timesteps

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        device = x_0.device
        s1 = self.sqrt_alphas_cumprod[t].to(device)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].to(device)
        while s1.ndim < x_0.ndim:
            s1 = s1.unsqueeze(-1)
            s2 = s2.unsqueeze(-1)
        return s1 * x_0 + s2 * noise

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self