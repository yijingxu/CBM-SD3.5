import os, sys
from pathlib import Path
import argparse
import random
import numpy as np

import torch
from torchvision.utils import save_image
from diffusers import DDIMScheduler

# ensure repo root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.cbae_ddpm import CBAE_DDPM, ConceptSpec
from models.clip_pseudolabeler import CLIP_PseudoLabeler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_01(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return x.mul(0.5).add_(0.5).clamp(0, 1)


def hard_concepts_from_logits(c_logits: torch.Tensor, spec: ConceptSpec) -> torch.Tensor:
    """
    Convert concatenated per-concept logits into hard labels per concept.
    Returns y_hat: [B, num_concepts] with integer class per concept.
    """
    B = c_logits.shape[0]
    y_hat = torch.empty((B, len(spec.sizes)), device=c_logits.device, dtype=torch.long)

    start = 0
    for k, sz in enumerate(spec.sizes):
        sl = c_logits[:, start:start + sz]
        y_hat[:, k] = sl.argmax(dim=-1)
        start += sz
    return y_hat


def get_c_logits_from_components(comps: dict) -> torch.Tensor:
    """
    Try common keys from your codepath.
    We need the concatenated concept logits [B, sum(spec.sizes)].
    """
    for key in ["concept_logits", "c_prime", "c_logits", "c_hat_logits", "c"]:
        if key in comps and isinstance(comps[key], torch.Tensor):
            return comps[key]
    raise KeyError(
        "Could not find concept logits in components. "
        f"Available keys: {list(comps.keys())}"
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--pretrained_model_id", type=str, default="google/ddpm-celebahq-256")
    parser.add_argument("--max_timestep", type=int, default=400)
    parser.add_argument("--hidden_dim", type=int, default=1024)

    parser.add_argument("--concept_idx", type=int, default=0)  # 0=Smiling in our spec below
    parser.add_argument("--target", type=int, default=1)       # force concept k to this value

    parser.add_argument("--num_batches", type=int, default=5)  # how many batches to evaluate
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=256)

    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--outdir", type=str, default="results/eval_mine")
    parser.add_argument("--save_batches", type=int, default=5)  # save first N batches of grids

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Concepts: MUST match training order ----
    selected_concepts = ["Smiling", "Young", "Male", "Eyeglasses"]
    spec = ConceptSpec(
        types=["bin", "bin", "bin", "bin"],
        sizes=[2, 2, 2, 2],
        names=selected_concepts,
        unsup_dim=0,
    )
    spec.validate()

    assert 0 <= args.concept_idx < len(selected_concepts), (
        f"concept_idx out of range. Must be in [0,{len(selected_concepts)-1}]"
    )
    assert args.target in [0, 1], "target must be 0 or 1 for binary concepts."

    # 1) Load model
    model = CBAE_DDPM(
        pretrained_model_id=args.pretrained_model_id,
        concept_spec=spec,
        hidden_dim=args.hidden_dim,
        max_timestep=args.max_timestep,
        freeze_unet=True,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    # Optional: silence intervention debug prints if present
    if hasattr(model, "debug_intervene"):
        model.debug_intervene = False

    # 2) CLIP pseudo-labeler for evaluation only
    pseudo = CLIP_PseudoLabeler(set_of_classes=selected_concepts, device=device)
    print("[Eval] CLIP pseudo-labeler ready.")

    # 3) DDIM scheduler (paper-style)
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(args.ddim_steps)

    # Helper: extract hard pred for concept k from concatenated 2K logits
    def pred_k_from_clip_logits(m_logits_concat: torch.Tensor, k: int) -> torch.Tensor:
        start = 2 * k
        return m_logits_concat[:, start:start + 2].argmax(dim=-1)

    # ---- Aggregate metrics (paper-style) ----
    num_target = 0.0
    num_not_target = 0.0
    num_succ = 0.0
    num_unsucc = 0.0
    num_negative = 0.0
    num_target_stays = 0.0

    for b in range(args.num_batches):
        # two parallel chains: original vs intervened
        x = torch.randn((args.batch_size, 3, args.img_size, args.img_size), device=device)
        x_int = x.clone()

        for t in scheduler.timesteps:
            t_int = int(t.item())

            # ORIGINAL PATH - use forward() with return_components=True
            eps_o, comps_o = model.forward(x, t, use_cbae=True, return_components=True)
            
            if comps_o is not None:
                x = scheduler.step(comps_o["eps_orig"], t, x).prev_sample
            else:
                x = scheduler.step(eps_o, t, x).prev_sample

            # INTERVENTION PATH
            eps_pred, comps = model.forward(x_int, t, use_cbae=True, return_components=True)

            # Only intervene when noise level is below threshold (t <= max_timestep)
            if t_int > args.max_timestep or comps is None:
                x_int = scheduler.step(eps_pred, t, x_int).prev_sample
                continue

            # Build y_hat from the model's concept logits
            c_logits = get_c_logits_from_components(comps)  # [B, sum(sizes)]
            y_hat = hard_concepts_from_logits(c_logits, spec)  # [B, num_concepts]

            ints = model.intervention_step(
                mid=comps["mid"],
                y_hat=y_hat,
                concept_to_intervene=args.concept_idx,
                target_class=args.target,
            )

            eps_int = model.ddpm_eps_from_mid(
                mid=ints["w_intervened"],
                emb=comps["emb"],
                down_res=comps["down_res"],
            )
            x_int = scheduler.step(eps_int, t, x_int).prev_sample

        # Convert final samples to [0,1] for CLIP + saving
        x_01 = to_01(x)
        xint_01 = to_01(x_int)

        # CLIP pseudo labels for *evaluation*
        logits_o = torch.cat(pseudo.get_soft_pseudo_labels(x_01), dim=-1)      # [B, 2K]
        logits_i = torch.cat(pseudo.get_soft_pseudo_labels(xint_01), dim=-1)   # [B, 2K]
        pre = pred_k_from_clip_logits(logits_o, args.concept_idx)
        post = pred_k_from_clip_logits(logits_i, args.concept_idx)

        # Save paired grids for first few batches
        if b < args.save_batches:
            save_image(x_01,    os.path.join(args.outdir, f"{b:03d}_orig.png"),  nrow=4)
            save_image(xint_01, os.path.join(args.outdir, f"{b:03d}_interv.png"), nrow=4)

        tgt = args.target
        not_tgt = 1 - tgt

        curr_tgt = (pre == tgt).sum().item()
        num_target += curr_tgt
        num_not_target += (args.batch_size - curr_tgt)

        num_succ += ((pre == not_tgt) & (post == tgt)).sum().item()
        num_unsucc += ((pre == not_tgt) & (post == not_tgt)).sum().item()
        num_target_stays += ((pre == tgt) & (post == tgt)).sum().item()
        num_negative += ((pre == tgt) & (post == not_tgt)).sum().item()

        flip = (pre != post).float().mean().item()
        p_after = (post == tgt).float().mean().item()
        print(f"[Batch {b}] P(after=target)={p_after:.3f} | flip={flip:.3f}")

    total = args.batch_size * args.num_batches

    # Normalize like the paper (conditioned on pre being not-target or target)
    succ = num_succ / max(num_not_target, 1.0)
    unsucc = num_unsucc / max(num_not_target, 1.0)
    neg = num_negative / max(num_target, 1.0)
    tgt_stays = num_target_stays / max(num_target, 1.0)

    print("\n=== Steerability-style summary (CLIP proxy) ===")
    print(f"concept={selected_concepts[args.concept_idx]} (idx={args.concept_idx}) | target={args.target}")
    print(f"successful interventions:   {succ*100:.2f}%  (pre!=target -> post==target)")
    print(f"unsuccessful interventions: {unsucc*100:.2f}% (pre!=target -> post!=target)")
    print(f"already target & no change: {tgt_stays*100:.2f}% (pre==target -> post==target)")
    print(f"negative intervention:      {neg*100:.2f}%   (pre==target -> post!=target)")
    print(f"images already target:      {int(num_target)}/{total}")
    print(f"images NOT target:          {int(num_not_target)}/{total}")
    print(f"saved grids to:             {args.outdir}")


if __name__ == "__main__":
    main()