#!/usr/bin/env python
"""
run_all_evals.py

Batch evaluation script that runs eval_intervention_ddpm_cs762.py 
for all concept/checkpoint combinations.

Usage:
    python scripts/run_all_evals.py --ckpt_dir checkpoints_local --results_dir results

Or in a notebook:
    !python scripts/run_all_evals.py --ckpt_dir checkpoints_local --results_dir results
"""

import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


# Concept name to index mapping (must match training order)
CONCEPTS = {
    "Smiling": 0,
    "Young": 1,
    "Male": 2,
    "Eyeglasses": 3,
}


def find_checkpoints(ckpt_dir: str) -> list:
    """Find all .pt checkpoint files in the directory."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")
    
    checkpoints = sorted(ckpt_path.glob("*.pt"))
    
    if not checkpoints:
        raise ValueError(f"No .pt files found in {ckpt_dir}")
    
    return checkpoints


def run_eval(
    ckpt_path: str,
    concept_name: str,
    concept_idx: int,
    target: int,
    results_dir: str,
    pretrained_model_id: str = "google/ddpm-celebahq-256",
    max_timestep: int = 400,
    hidden_dim: int = 1024,
    num_batches: int = 5,
    batch_size: int = 16,
    ddim_steps: int = 50,
    seed: int = 0,
) -> dict:
    """Run a single evaluation and return results."""
    
    ckpt_name = Path(ckpt_path).stem  # e.g., "cbae_ddpm_step5000"
    
    # Create output directory
    outdir = os.path.join(results_dir, ckpt_name, f"eval_{concept_name}_target{target}")
    os.makedirs(outdir, exist_ok=True)
    
    cmd = [
        "python", "-m", "scripts.eval_intervention_ddpm_cs762",
        "--ckpt", str(ckpt_path),
        "--pretrained_model_id", pretrained_model_id,
        "--max_timestep", str(max_timestep),
        "--hidden_dim", str(hidden_dim),
        "--concept_idx", str(concept_idx),
        "--target", str(target),
        "--num_batches", str(num_batches),
        "--batch_size", str(batch_size),
        "--ddim_steps", str(ddim_steps),
        "--outdir", outdir,
        "--seed", str(seed),
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {ckpt_name} | {concept_name} (idx={concept_idx}) | target={target}")
    print(f"Output: {outdir}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return {
        "checkpoint": ckpt_name,
        "concept": concept_name,
        "concept_idx": concept_idx,
        "target": target,
        "returncode": result.returncode,
        "outdir": outdir,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for all concepts and checkpoints")
    
    # Directories
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_local",
                        help="Directory containing checkpoint .pt files")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base directory for results")
    
    # Model settings
    parser.add_argument("--pretrained_model_id", type=str, default="google/ddpm-celebahq-256")
    parser.add_argument("--max_timestep", type=int, default=400)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    
    # Eval settings
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    
    # Target settings
    parser.add_argument("--targets", type=int, nargs="+", default=[0, 1],
                        help="Target values to evaluate (default: both 0 and 1)")
    
    # Filter options
    parser.add_argument("--concepts", type=str, nargs="+", default=None,
                        help="Only evaluate these concepts (default: all)")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Only evaluate these checkpoints (default: all)")
    
    args = parser.parse_args()
    
    # Find checkpoints
    all_checkpoints = find_checkpoints(args.ckpt_dir)
    print(f"Found {len(all_checkpoints)} checkpoints in {args.ckpt_dir}")
    
    # Filter checkpoints if specified
    if args.checkpoints:
        all_checkpoints = [c for c in all_checkpoints 
                          if any(f in c.stem for f in args.checkpoints)]
        print(f"Filtered to {len(all_checkpoints)} checkpoints")
    
    # Filter concepts if specified
    concepts_to_eval = CONCEPTS.copy()
    if args.concepts:
        concepts_to_eval = {k: v for k, v in CONCEPTS.items() if k in args.concepts}
    
    print(f"Concepts to evaluate: {list(concepts_to_eval.keys())}")
    print(f"Targets to evaluate: {args.targets}")
    print(f"Checkpoints: {[c.stem for c in all_checkpoints]}")
    
    # Calculate total runs
    total_runs = len(all_checkpoints) * len(concepts_to_eval) * len(args.targets)
    print(f"\nTotal evaluations to run: {total_runs}")
    
    # Run all evaluations
    results = []
    run_count = 0
    
    for ckpt_path in all_checkpoints:
        for concept_name, concept_idx in concepts_to_eval.items():
            for target in args.targets:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}]")
                
                result = run_eval(
                    ckpt_path=str(ckpt_path),
                    concept_name=concept_name,
                    concept_idx=concept_idx,
                    target=target,
                    results_dir=args.results_dir,
                    pretrained_model_id=args.pretrained_model_id,
                    max_timestep=args.max_timestep,
                    hidden_dim=args.hidden_dim,
                    num_batches=args.num_batches,
                    batch_size=args.batch_size,
                    ddim_steps=args.ddim_steps,
                    seed=args.seed,
                )
                results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r["returncode"] == 0]
    failed = [r for r in results if r["returncode"] != 0]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    
    if failed:
        print("\nFailed runs:")
        for r in failed:
            print(f"  - {r['checkpoint']} | {r['concept']} | target={r['target']}")
    
    # Save summary to file
    summary_path = os.path.join(args.results_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Evaluation Summary - {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoints: {[c.stem for c in all_checkpoints]}\n")
        f.write(f"Concepts: {list(concepts_to_eval.keys())}\n")
        f.write(f"Targets: {args.targets}\n\n")
        f.write(f"Successful: {len(successful)}/{len(results)}\n")
        f.write(f"Failed: {len(failed)}/{len(results)}\n\n")
        
        for r in results:
            status = "✓" if r["returncode"] == 0 else "✗"
            f.write(f"{status} {r['checkpoint']} | {r['concept']} | target={r['target']}\n")
            f.write(f"  Output: {r['outdir']}\n\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()