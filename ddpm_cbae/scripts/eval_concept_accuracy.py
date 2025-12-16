#!/usr/bin/env python
"""
eval_concept_accuracy.py

Evaluates CB-AE concept accuracy on CelebA (HF: flwrlabs/celeba) using ground-truth attributes.

This matches your notebook-style Step 9 evaluation:
- Concepts: Smiling, Young, Male, Eyeglasses
- Preprocess: CenterCrop(178) -> Resize(256) -> Normalize to [-1,1]
- Logits: 2 per binary concept (total_dim = 8)
- Accuracy computed per concept and mean accuracy

Eval modes:
- random_t: add noise at random t in [0, max_timestep]
- t0: add noise at t=0 (still adds a tiny amount of noise via scheduler)
- x0: no noise; pass clean x0 directly (t=0, x_t = x0)

Note: This script assumes your repo contains models/cbae_ddpm.py (your updated implementation).
"""

import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(ds_split, batch_size, num_workers, selected_concepts):
    tfm = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # Defensive schema check (expects top-level concept columns in flwrlabs/celeba)
    ex0 = ds_split[0]
    missing = [c for c in selected_concepts if c not in ex0]
    if missing:
        raise ValueError(
            f"Dataset split is missing label columns: {missing}. "
            f"Example keys: {list(ex0.keys())[:30]}"
        )

    def collate(batch):
        imgs = [tfm(ex["image"].convert("RGB")) for ex in batch]
        ys = [[int(ex[c]) for c in selected_concepts] for ex in batch]  # 0/1
        return torch.stack(imgs, 0), torch.tensor(ys, dtype=torch.long)

    return DataLoader(
        ds_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )


@torch.no_grad()
def evaluate(model, scheduler, loader, device, max_timestep, eval_mode, eval_batches, selected_concepts):
    model.eval()

    correct = torch.zeros(len(selected_concepts), dtype=torch.long)
    total = 0

    for bi, (x0, y01) in enumerate(loader, start=1):
        if eval_batches is not None and bi > eval_batches:
            break

        x0 = x0.to(device, non_blocking=True)
        y01 = y01.to(device, non_blocking=True)  # [B,4] in {0,1}
        B = x0.size(0)

        if eval_mode == "random_t":
            t = torch.randint(0, max_timestep + 1, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = scheduler.add_noise(x0, noise, t)

        elif eval_mode == "t0":
            t = torch.zeros(B, device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = scheduler.add_noise(x0, noise, t)

        elif eval_mode == "x0":
            t = torch.zeros(B, device=device, dtype=torch.long)
            x_t = x0

        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

        _, comps = model(x_t, t, use_cbae=True, return_components=True)

        # [B, 8] -> [B, 4, 2]
        logits = comps["concept_logits"].view(B, len(selected_concepts), 2)
        pred = logits.argmax(dim=-1)  # [B,4] in {0,1}

        correct += (pred == y01).sum(dim=0).cpu()
        total += B

        if bi % 50 == 0:
            print(f"Processed {total} examples...")

    acc = (correct.float() / max(total, 1)).numpy()
    return acc


def main():
    parser = argparse.ArgumentParser()

    # Repo import path
    parser.add_argument("--repo_root", type=str, default=".")
    # HF checkpoint
    parser.add_argument("--hf_repo_id", type=str, required=True)
    parser.add_argument("--hf_ckpt_path", type=str, default="checkpoints/cbae_ddpm_midblock_v1.pt")

    # Backbone + CB-AE config (must match training)
    parser.add_argument("--pretrained_model_id", type=str, default="google/ddpm-celebahq-256")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--max_timestep", type=int, default=400)

    # Data
    parser.add_argument("--dataset_id", type=str, default="flwrlabs/celeba")
    parser.add_argument("--split", type=str, default="valid")  # IMPORTANT: flwrlabs uses 'valid'
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)

    # Eval control
    parser.add_argument("--eval_mode", type=str, default="random_t", choices=["random_t", "t0", "x0"])
    parser.add_argument("--eval_batches", type=int, default=2000,
                        help="Number of batches to evaluate. Set to 0 for full split.")
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    set_seed(args.seed)

    import sys
    if args.repo_root not in sys.path:
        sys.path.append(args.repo_root)

    from models.cbae_ddpm import CBAE_DDPM, ConceptSpec, DDPMNoiseSchedulerHelper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    selected_concepts = ["Smiling", "Young", "Male", "Eyeglasses"]
    spec = ConceptSpec(
        types=["bin", "bin", "bin", "bin"],
        sizes=[2, 2, 2, 2],  # 2 logits per binary concept
        names=selected_concepts,
        unsup_dim=0,
    )
    spec.validate()

    # Download checkpoint
    ckpt_file = hf_hub_download(
        repo_id=args.hf_repo_id,
        filename=args.hf_ckpt_path,
        repo_type="model",
    )
    ckpt = torch.load(ckpt_file, map_location="cpu")
    if "max_timestep" in ckpt:
        args.max_timestep = int(ckpt["max_timestep"])

    # Build model + load CB-AE weights
    model = CBAE_DDPM(
        pretrained_model_id=args.pretrained_model_id,
        concept_spec=spec,
        hidden_dim=args.hidden_dim,
        max_timestep=args.max_timestep,
        freeze_unet=True,
    ).to(device).eval()

    model.cbae.load_state_dict(ckpt["cbae_state"])
    print(f"Loaded checkpoint step={ckpt.get('global_step', 'NA')} from {args.hf_repo_id}/{args.hf_ckpt_path}")

    scheduler = DDPMNoiseSchedulerHelper(num_train_timesteps=1000).to(device)

    # Load dataset and split
    ds = load_dataset(args.dataset_id)
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found. Available splits: {list(ds.keys())}")

    loader = build_loader(ds[args.split], args.batch_size, args.num_workers, selected_concepts)

    eval_batches = None if args.eval_batches == 0 else args.eval_batches
    acc = evaluate(model, scheduler, loader, device, args.max_timestep, args.eval_mode, eval_batches, selected_concepts)

    print("Eval mode:", args.eval_mode)
    print("Acc per concept:", acc)
    print("Mean acc:", float(acc.mean()))


if __name__ == "__main__":
    main()
