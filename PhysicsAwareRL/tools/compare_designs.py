"""Design comparison tool: computes J_seed and J_target Jaccard similarities on occupied cells."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from PhysicsAwareRL.metrics import jaccard


def load_design(path: str) -> Optional[Dict[str, Any]]:
    design_file = os.path.join(path, "design.json") if os.path.isdir(path) else path
    if not os.path.exists(design_file):
        return None
    with open(design_file, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_occupied_cells(design: Dict[str, Any]) -> Set[Tuple[int, int]]:
    return {tuple(t["cell"]) for t in design.get("tiles", [])}


def compare_runs(run_paths: List[str]) -> Dict[str, Any]:
    runs_data = []
    for p in run_paths:
        d = load_design(p)
        if d is None:
            continue
        cfg_file = os.path.join(p if os.path.isdir(p) else os.path.dirname(p), "config.json")
        cfg = {}
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
            except Exception:
                pass

        cells = get_occupied_cells(d)
        target = tuple(cfg.get("targets", d.get("spec", {}).get("target_flows", [])))
        seed = cfg.get("seed", d.get("seed", 0))

        runs_data.append({
            "path": p,
            "target": target,
            "seed": seed,
            "cells": cells,
            "n_tiles": len(cells),
            "score": d.get("score", 0.0),
            "e_q": d.get("metrics", {}).get("e_q", 0.0),
            "success": d.get("metrics", {}).get("success", False),
            "topo_id": d.get("topo_id", ""),
            "design_id": d.get("design_id", ""),
        })

    if not runs_data:
        print("No valid design.json files found.")
        return {}

    j_seed_list = []
    j_target_list = []

    n = len(runs_data)
    print("\n================== DESIGN SUMMARY TABLE ==================")
    header = f"{'Path':<30} {'Seed':<5} {'Tiles':<6} {'E_Q':<8} {'Score':<8} {'Success':<8} {'Topo_ID':<14} Target"
    print(header)
    print("-" * len(header))
    for r in runs_data:
        tgt_str = str([round(x, 4) for x in r["target"]]) if r["target"] else "[]"
        print(
            f"{os.path.basename(r['path']):<30} {r['seed']:<5} {r['n_tiles']:<6} "
            f"{r['e_q']:<8.4f} {r['score']:<8.4f} {str(r['success']):<8} {r['topo_id']:<14} {tgt_str}"
        )

    # Compute pairwise Jaccard
    print("\n================ PAIRWISE JACCARD SIMILARITY ================")
    for i in range(n):
        for j in range(i + 1, n):
            r_a = runs_data[i]
            r_b = runs_data[j]
            j_val = jaccard(r_a["cells"], r_b["cells"])
            same_seed = (r_a["seed"] == r_b["seed"])
            same_tgt = (r_a["target"] == r_b["target"])

            pair_type = ""
            if same_seed and not same_tgt:
                pair_type = "J_seed (same seed, diff target)"
                j_seed_list.append(j_val)
            elif same_tgt and not same_seed:
                pair_type = "J_target (same target, diff seed)"
                j_target_list.append(j_val)
            elif same_seed and same_tgt:
                pair_type = "replicate"
            else:
                pair_type = "cross (diff seed, diff target)"

            print(
                f"{os.path.basename(r_a['path'])} vs {os.path.basename(r_b['path'])}: "
                f"J = {j_val:.4f} [{pair_type}]"
            )

    mean_j_seed = float(np.mean(j_seed_list)) if j_seed_list else None
    mean_j_target = float(np.mean(j_target_list)) if j_target_list else None

    print("\n------------------- AGGREGATE SUMMARY -------------------")
    if mean_j_seed is not None:
        print(f"Mean J_seed   (sameness under same seed, diff target) : {mean_j_seed:.4f}")
    if mean_j_target is not None:
        print(f"Mean J_target (sameness under same target, diff seed) : {mean_j_target:.4f}")

    return {
        "runs": runs_data,
        "mean_j_seed": mean_j_seed,
        "mean_j_target": mean_j_target,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare microfluidic designs across runs (J_seed & J_target)")
    parser.add_argument("runs", nargs="+", help="Run directories or design.json paths to compare")
    args = parser.parse_args()

    expanded = []
    for r in args.runs:
        matches = glob.glob(r)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(r)

    compare_runs(expanded)


if __name__ == "__main__":
    main()
