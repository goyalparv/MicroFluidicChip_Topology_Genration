"""Automated batch runner: trains Stage 1 RL sequentially across 20 distinct flow target specifications.

Executes each run completely independently from the ground up.
All results for each run are kept completely self-contained in its own separate directory:
- `design.json`: Best synthesized chip layout
- `chip_layout.png`: High-resolution visual schematic of the chip
- `rl_rewards.png`: Standalone learning curve for reward & score progression
- `rl_flow_error.png`: Standalone learning curve for hydraulic flow error convergence
- `rl_tile_count.png`: Standalone learning curve for topology compactness & tile usage
- `rl_policy_entropy.png`: Standalone learning curve for policy exploration & entropy health
- `eval_summary.json`: Statistical performance over post-training evaluation episodes
- `comsol_check/`: Automated COMSOL validation outputs (json, csv, and assembled chip plot)
- `archive.json`: Top-K distinct topology archive
- `history.json`: Episode-by-episode training history
- `config.json`: Full configuration parameters
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

from .run_stage1 import main as run_stage1_main


TARGETS_20: List[List[float]] = [
    [0.3333, 0.3334, 0.3333],  # 1: Equal split
    [0.50, 0.30, 0.20],        # 2: Left descending
    [0.20, 0.30, 0.50],        # 3: Right ascending
    [0.40, 0.40, 0.20],        # 4: Left-center dominant
    [0.20, 0.40, 0.40],        # 5: Center-right dominant
    [0.20, 0.60, 0.20],        # 6: Center peak symmetric
    [0.45, 0.10, 0.45],        # 7: Valley center symmetric
    [0.60, 0.20, 0.20],        # 8: Left heavy
    [0.20, 0.20, 0.60],        # 9: Right heavy
    [0.35, 0.45, 0.20],        # 10: Center-left bias
    [0.20, 0.45, 0.35],        # 11: Center-right bias
    [0.55, 0.25, 0.20],        # 12: High left moderate
    [0.20, 0.25, 0.55],        # 13: High right moderate
    [0.15, 0.70, 0.15],        # 14: Tall center peak
    [0.70, 0.15, 0.15],        # 15: Strong left bias
    [0.15, 0.15, 0.70],        # 16: Strong right bias
    [0.40, 0.20, 0.40],        # 17: U-shaped split
    [0.30, 0.50, 0.20],        # 18: Mild center-left
    [0.25, 0.50, 0.25],        # 19: Bell-curve 1:2:1
    [0.80, 0.10, 0.10],        # 20: Extreme left dominant
]

# Internal structural objective variation to promote topological diversity
_OBJECTIVES_SEQUENCE = [
    "compact", "tree", "low-pressure", "compact", "tree",
    "low-pressure", "compact", "tree", "low-pressure", "compact",
    "tree", "low-pressure", "compact", "tree", "low-pressure",
    "compact", "tree", "low-pressure", "compact", "tree"
]


def run_batch_automation(
    episodes: int = 2000,
    batch_size: int = 16,
    eval_episodes: int = 50,
    out_dir: str = "results/batch_20_targets",
    selected_indices: Optional[List[int]] = None,
    library_path: Optional[str] = None,
    validate: bool = True,
    validate_backend: str = "comsol",
    comsol_plugins: str = "C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins",
    allow_baseline_library: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    indices = selected_indices or list(range(1, len(TARGETS_20) + 1))

    print("\n" + "=" * 70)
    print(f"   STARTING INDEPENDENT BATCH RUNS ({len(indices)} TARGET SPECIFICATIONS)")
    print("=" * 70)
    print(f"Episodes per run : {episodes}")
    print(f"Rollouts / update: {batch_size}")
    print(f"Post-eval trials : {eval_episodes}")
    print(f"Validation mode  : {'Automated COMSOL' if validate else 'Disabled'}")
    print(f"Root output dir  : {os.path.abspath(out_dir)}\n")

    t_batch_start = time.time()

    for idx in indices:
        tgt = TARGETS_20[idx - 1]
        obj = _OBJECTIVES_SEQUENCE[(idx - 1) % len(_OBJECTIVES_SEQUENCE)]

        tag = f"run_{idx:02d}_target_{int(round(tgt[0]*100)):02d}_{int(round(tgt[1]*100)):02d}_{int(round(tgt[2]*100)):02d}"
        run_out = os.path.join(out_dir, tag)

        print(f"\n{'='*70}")
        print(f">>> [Run {idx:02d}/{len(TARGETS_20):02d}] Target: {tgt} -> Initializing new run from scratch...")
        print(f"    Output directory: {os.path.abspath(run_out)}")
        print(f"{'='*70}\n")

        cmd_argv = [
            "--targets", str(tgt[0]), str(tgt[1]), str(tgt[2]),
            "--episodes", str(episodes),
            "--batch", str(batch_size),
            "--eval-episodes", str(eval_episodes),
            "--objective", obj,
            "--out", run_out,
        ]
        if library_path:
            cmd_argv += ["--library", library_path]
        if not validate:
            cmd_argv.append("--no-validate")
        else:
            cmd_argv += [
                "--validate-backend", validate_backend,
                "--comsol-plugins", comsol_plugins,
            ]
            if allow_baseline_library:
                cmd_argv.append("--allow-baseline-library")

        try:
            run_stage1_main(cmd_argv)
            print(f"\n<<< [Run {idx:02d} Completed] Results saved independently to:\n    {os.path.abspath(run_out)}")
        except Exception as e:
            print(f"Error during run {idx}: {e}", file=sys.stderr)

    total_time = time.time() - t_batch_start
    print("\n" + "=" * 70)
    print(f"All {len(indices)} independent runs completed in {total_time / 60:.1f} minutes.")
    print(f"Each run's results are stored independently in: {os.path.abspath(out_dir)}")
    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Automate Stage 1 RL Synthesis across 20 Target Ratios Independently")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="RL training episodes per target specification (default: 2000)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Rollouts per update (default: 16)")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Post-training evaluation trials (default: 50)")
    parser.add_argument("--library", type=str, default=None,
                        help="Optional path to custom COMSOL library CSV")
    parser.add_argument("--out-dir", type=str, default="results/batch_20_targets",
                        help="Root output directory (default: results/batch_20_targets)")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Run only specific target indices (1 to 20), e.g. --indices 1 2 3")
    parser.add_argument("--no-validate", action="store_true",
                        help="Disable automatic COMSOL validation after each run")
    parser.add_argument("--validate-backend", choices=["comsol", "raster", "mock", "none"], default="comsol",
                        help="Validation backend (default: comsol)")
    parser.add_argument("--comsol-plugins", type=str, default="C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins",
                        help="Path to COMSOL plugins directory")
    parser.add_argument("--allow-baseline-library", action="store_true",
                        help="Allow validating designs with baseline non-COMSOL libraries")
    return parser.parse_args()


def main():
    args = parse_args()
    run_batch_automation(
        episodes=args.episodes,
        batch_size=args.batch,
        eval_episodes=args.eval_episodes,
        out_dir=args.out_dir,
        selected_indices=args.indices,
        library_path=args.library,
        validate=not args.no_validate,
        validate_backend=args.validate_backend,
        comsol_plugins=args.comsol_plugins,
        allow_baseline_library=args.allow_baseline_library,
    )


if __name__ == "__main__":
    main()
