#!/usr/bin/env python
"""
train_cbae_ddpm.py

Trains CB-AE inserted at the UNet mid-block of a pretrained DDPM UNet.
Backbone UNet is frozen; only CB-AE is optimized.

This script is a direct “script-ified” version of your Step 6–8 notebook code.
"""

import os
import time
import argparse
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset

from huggingface_hub import HfApi, hf_hub_download

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CelebAHFDataset(Dataset):
    """
    Wraps flwrlabs/celeba HF dataset split into (x0, y01) pairs.

    x0: [3,256,256] normalized to [-1, 1]
    y01: [4] in {0,1} for selected concepts
    """
    def __init__(self, hf_split, transform, selected_concepts):
        self.data = hf_split
        self.transform = transform
        self.selected_concepts = selected_concepts

        ex0 = self.data[0]
        self._has_attr_dict = ("attributes" in ex0)
        self._has_attr_cols = all(c in ex0 for c in selected_concepts)

        print(
            "[Dataset] Schema:",
            "attributes_dict" if self._has_attr_dict else "no_attributes_dict",
            "| top_level_attr_cols_ok?", self._has_attr_cols,
        )

        if (not self._has_attr_dict) and (not self._has_attr_cols):
            raise ValueError(
                "HF dataset does not expose attributes in expected format. "
                f"Example keys: {list(ex0.keys())[:25]}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        ex = self.data[idx]
        img = ex["image"].convert("RGB")
        x = self.transform(img)

        if "attributes" in ex:
            attrs = ex["attributes"]
            y = torch.tensor([int(attrs[c]) for c in self.selected_concepts], dtype=torch.long)
        else:
            y = torch.tensor([int(ex[c]) for c in self.selected_concepts], dtype=torch.long)

        return x, y


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    # Paths / repo
    parser.add_argument("--repo_root", type=str, default=".",
                        help="Local path to the posthoc-generative-cbm repo root.")
    parser.add_argument("--hf_repo_id", type=str, default=".",
                        help="HF repo to upload checkpoints to (repo_type='model').")
    parser.add_argument("--hf_ckpt_path", type=str, default="checkpoints/cbae_ddpm_midblock_v1.pt",
                        help="Path inside HF repo where checkpoint will be stored.")
    parser.add_argument("--upload_to_hf", action="store_true",
                        help="If set, upload checkpoints to Hugging Face.")

    # Model / training
    parser.add_argument("--pretrained_model_id", type=str, default="google/ddpm-celebahq-256")
    parser.add_argument("--max_timestep", type=int, default=400)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--freeze_unet", action="store_true", default=True)

    # Data
    parser.add_argument("--dataset_id", type=str, default="flwrlabs/celeba")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)

    # Loss weights (your current settings)
    parser.add_argument("--w_lr1", type=float, default=1.0)
    parser.add_argument("--w_lr2", type=float, default=1.0)
    parser.add_argument("--w_lc", type=float, default=1.0)

    # Intervention loss weights
    parser.add_argument("--w_li1", type=float, default=0.0)
    parser.add_argument("--w_li2", type=float, default=0.0)
    
    # How often to apply intervention per batch
    parser.add_argument("--p_intervene", type=float, default=0.5)
    
    # Optional: intervene a fixed concept index (-1 means random)
    parser.add_argument("--intervene_idx", type=int, default=-1)
    parser.add_argument("--intervene_target", type=int, default=-1,
                    help="If >=0, force intervened concept to this class (0/1). If -1, use default flip logic.")

    # Resume (optional)
    parser.add_argument("--resume_from_hf", action="store_true",
                        help="If set, tries to resume CB-AE weights from HF checkpoint.")
    args = parser.parse_args()
    pseudo = None

    set_seed(args.seed)

    # Put repo on path and import your implementation
    import sys
    if args.repo_root not in sys.path:
        sys.path.append(args.repo_root)

    # IMPORTANT: this expects your updated code lives here:
    #   /content/posthoc-generative-cbm/models/cbae_ddpm.py
    from models.cbae_ddpm import CBAE_DDPM, ConceptSpec, DDPMNoiseSchedulerHelper
    from models.clip_pseudolabeler import CLIP_PseudoLabeler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    pseudo = None
    if args.w_li1 > 0:
        set_of_classes = [
            ["not smiling", "smiling"],              # Smiling
            ["not young", "young"],                  # Young
            ["female", "male"],                      # Male
            ["no eyeglasses", "wearing eyeglasses"], # Eyeglasses
        ]
        pseudo = CLIP_PseudoLabeler(
            set_of_classes=set_of_classes,
            device=device,
            clip_model_name="ViT-B/16",
            clip_model_type="clip",
        ).eval()
        for p in pseudo.parameters():
            p.requires_grad = False
        print("[CLIP] Pseudo-labeler ready.")

    # ---- Concepts (fixed to your 4-concept run) ----
    selected_concepts = ["Smiling", "Young", "Male", "Eyeglasses"]
    # Paper convention: 2 logits per binary concept => total_dim=8
    spec = ConceptSpec(
        types=["bin", "bin", "bin", "bin"],
        sizes=[2, 2, 2, 2],
        names=selected_concepts,
        unsup_dim=0,
    )
    spec.validate()

    # ---- Dataset ----
    print(f"[Data] Loading HF dataset: {args.dataset_id}")
    ds = load_dataset(args.dataset_id)

    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found. Available splits: {list(ds.keys())}")

    tfm = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dataset = CelebAHFDataset(ds[args.split], tfm, selected_concepts)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ---- Model ----
    model = CBAE_DDPM(
        pretrained_model_id=args.pretrained_model_id,
        concept_spec=spec,
        hidden_dim=args.hidden_dim,
        max_timestep=args.max_timestep,
        freeze_unet=args.freeze_unet,
    ).to(device)

    scheduler = DDPMNoiseSchedulerHelper(num_train_timesteps=1000).to(device)

    # ---- Optimizer (CB-AE only) ----
    optimizer = torch.optim.Adam(model.cbae.parameters(), lr=args.lr)

    # ---- Optional resume (CB-AE weights only) ----
    global_step = 0
    if args.resume_from_hf:
        try:
            ckpt_file = hf_hub_download(
                repo_id=args.hf_repo_id,
                filename=args.hf_ckpt_path,
                repo_type="model",
            )
            ckpt = torch.load(ckpt_file, map_location="cpu")
            model.cbae.load_state_dict(ckpt["cbae_state"])
            global_step = int(ckpt.get("global_step", 0))
            print(f"[Resume] Loaded CB-AE from {args.hf_repo_id}/{args.hf_ckpt_path} (step={global_step})")
        except Exception as e:
            print(f"[Resume] Failed to resume from HF: {e}")
            print("[Resume] Starting fresh training.")

    # ---- Training loop (Step 8 logic) ----
    model.train()
    running = {"lr1": 0.0, "lr2": 0.0, "lc": 0.0, "li1": 0.0, "li2": 0.0, "total": 0.0}
    t0 = time.time()

    it = iter(train_loader)

    # local checkpoint staging
    local_ckpt_dir = os.path.join(os.getcwd(), "checkpoints_local")
    os.makedirs(local_ckpt_dir, exist_ok=True)
    local_ckpt_path = os.path.join(local_ckpt_dir, "cbae_ddpm_latest.pt")

    api = HfApi() if args.upload_to_hf else None

    print("[Train] Starting...")

    for step in range(global_step + 1, args.total_steps + 1):
        try:
            x0, y01 = next(it)
        except StopIteration:
            it = iter(train_loader)
            x0, y01 = next(it)

        x0 = x0.to(device, non_blocking=True)
        y01 = y01.to(device, non_blocking=True)

        B = x0.size(0)
        t = torch.randint(0, args.max_timestep + 1, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = scheduler.add_noise(x0, noise, t)

        optimizer.zero_grad(set_to_none=True)
        eps_pred, comps = model(x_t, t, use_cbae=True, return_components=True)

        # Your loss components
        loss_lr1 = model.loss_Lr1(comps["mid"], comps["recon_mid"])
        loss_lr2 = model.loss_Lr2_proxy(comps["eps_orig"], eps_pred)
        loss_lc = model.loss_Lc(comps["concept_logits"], y01)

        loss_li1 = torch.tensor(0.0, device=device)
        loss_li2 = torch.tensor(0.0, device=device)
        
        do_intervene = (torch.rand(()) < args.p_intervene) and (args.w_li1 > 0 or args.w_li2 > 0)
        
        if do_intervene:
            concept_idx = None if args.intervene_idx < 0 else args.intervene_idx
            target_class = None if args.intervene_target < 0 else int(args.intervene_target)

            ints = model.intervention_step(
                mid=comps["mid"],          # from return_components
                y_hat=y01,                 # your GT concept labels
                concept_to_intervene=concept_idx,
                target_class=target_class, # keeps your default flip logic
            )
        
            # 1) compute eps for intervened latent at same t
            eps_int = model.ddpm_eps_from_mid(
                mid=ints["w_intervened"],
                emb=comps["emb"],
                down_res=comps["down_res"],
            )
        
            # 2) convert to x0 prediction (image-like)
            x0_int = scheduler.predict_x0(x_t, eps_int, t).clamp(-1, 1)
        
            # 3) CLIP pseudo-label logits (list of [B,2])
            # Li1 uses external scorer M(x) (CLIP). Only run if w_li1 > 0.
            if args.w_li1 > 0:
                assert pseudo is not None, "pseudo-labeler not initialized (pseudo is None)"
                with torch.no_grad():
                    logits_list = pseudo.get_soft_pseudo_labels(x0_int)
                m_logits = torch.cat(logits_list, dim=-1)
                loss_li1 = model.loss_Lc(m_logits, ints["y_hat_intervened"])
            else:
                loss_li1 = torch.tensor(0.0, device=device)
                        
            # Li2: enforce encoder-consistency matches intervened target
            loss_li2 = model.loss_Lc(ints["c_prime_intervened"], ints["y_hat_intervened"])
            
            if args.w_li1 > 0 and (step % args.log_every == 0):
                assert pseudo is not None
                k = ints["concept_to_intervene"]
            
                # x0 from original eps (no intervention)
                x0_orig = scheduler.predict_x0(x_t, comps["eps_orig"], t).clamp(-1, 1)
            
                with torch.no_grad():
                    m_logits_o = torch.cat(pseudo.get_soft_pseudo_labels(x0_orig), dim=-1)
                    m_logits_i = torch.cat(pseudo.get_soft_pseudo_labels(x0_int),  dim=-1)
            
                def pred_k(m_logits_concat, k):
                    start = 2 * k
                    return m_logits_concat[:, start:start+2].argmax(dim=-1)
            
                pred_before = pred_k(m_logits_o, k)
                pred_after  = pred_k(m_logits_i, k)

                flip_rate = (pred_before != pred_after).float().mean().item()
                #print(f"[Dbg] intervene k={k} | CLIP flip-rate@k={flip_rate:.3f}")
                b0 = (pred_before == 0).sum().item()
                b1 = (pred_before == 1).sum().item()
                a0 = (pred_after  == 0).sum().item()
                a1 = (pred_after  == 1).sum().item()
                
                # NEW: success rate for "force smiling"
                succ = (pred_after == 1).float().mean().item()
            
                print(
                    f"[Dbg] k={k} | P(after=1)={succ:.3f} | "
                    f"before (0,1)=({b0},{b1}) -> after (0,1)=({a0},{a1}) | "
                    f"flip={flip_rate:.3f}"
                )

        # loss = args.w_lr1 * loss_lr1 + args.w_lr2 * loss_lr2 + args.w_lc * loss_lc
        loss = (
            args.w_lr1 * loss_lr1
          + args.w_lr2 * loss_lr2
          + args.w_lc  * loss_lc
          + args.w_li1 * loss_li1
          + args.w_li2 * loss_li2
        )

        loss.backward()
        optimizer.step()

        running["lr1"] += loss_lr1.item()
        running["lr2"] += loss_lr2.item()
        running["lc"] += loss_lc.item()
        running['li1'] += loss_li1.item()
        running['li2'] += loss_li2.item()
        running["total"] += loss.item()
        '''
        if step % args.log_every == 0:
            dt = time.time() - t0
            print(
                f"[Step {step}] loss={running['total']/args.log_every:.4f} | "
                f"lr1={running['lr1']/args.log_every:.4f} | "
                f"lr2={running['lr2']/args.log_every:.4f} | "
                f"lc={running['lc']/args.log_every:.4f} | "
                f"{dt:.1f}s"
            )
            for k in running:
                running[k] = 0.0
            t0 = time.time()
        '''
        if step % args.log_every == 0:
            dt = time.time() - t0
            print(
                f"[Step {step}] loss={running['total']/args.log_every:.4f} | "
                f"lr1={running['lr1']/args.log_every:.4f} | "
                f"lr2={running['lr2']/args.log_every:.4f} | "
                f"lc={running['lc']/args.log_every:.4f} | "
                f"li1={running['li1']/args.log_every:.4f} | "
                f"li2={running['li2']/args.log_every:.4f} | "
                f"{dt:.1f}s"
            )
            for k in running:
                running[k] = 0.0
            t0 = time.time()


        if step % args.save_every == 0:
            ckpt = {
                "global_step": step,
                "cbae_state": model.cbae.state_dict(),
                # Note: for your replication you decided NOT to save optimizer state.
                # "optim_state": optimizer.state_dict(),
                "loss_weights": {"w_lc": args.w_lc, "w_lr1": args.w_lr1, "w_lr2": args.w_lr2},
                "max_timestep": args.max_timestep,
            }
            torch.save(ckpt, local_ckpt_path)
            print(f"[CKPT] Saved local checkpoint: {local_ckpt_path} (step={step})")

            if args.upload_to_hf:
                api.upload_file(
                    path_or_fileobj=local_ckpt_path,
                    path_in_repo=args.hf_ckpt_path,
                    repo_id=args.hf_repo_id,
                    repo_type="model",
                )
                print(f"[HF] Uploaded checkpoint: {args.hf_repo_id}/{args.hf_ckpt_path} (step={step})")

    # Always save final checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/cbae_ddpm_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[CKPT] Saved final checkpoint to {ckpt_path}")

    print("[Train] Done.")


if __name__ == "__main__":
    main()
