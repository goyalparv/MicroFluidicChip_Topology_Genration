"""Stage 1 Command Line: Train RL agent to synthesize tile-based microfluidic chip topology."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from .closure_v2 import check_port_degrees, find_open_ports
from .config_v2 import (
    DEFAULT_INLET_FLOW,
    PRESETS,
    DesignSpec,
    RewardWeights,
    apply_preset,
    normalize_targets,
)
from .environment_v2 import TopologyEnv
from .library_v2 import Library, PlacedTile
from .metrics_v2 import design_id, topo_id
from .train_v2 import tiles_to_board, train_agent, train_agent_ppo


def parse_ports(items: List[str]) -> List[Tuple[int, int]]:
    out = []
    for it in items:
        try:
            r, c = (int(x.strip()) for x in it.split(","))
            out.append((r, c))
        except ValueError:
            raise SystemExit(f"Invalid outlet port '{it}': expected format 'row,col', e.g. '7,3'")
    return out


def get_git_hash() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 1: Tile-Based Microfluidic RL Topology Synthesis (v3)")
    ap.add_argument("--grid", type=int, default=8, help="Grid dimension (default 8 for 8x8)")
    ap.add_argument("--inlet", type=int, nargs=2, default=[0, 3], metavar=("ROW", "COL"),
                    help="Inlet cell (must be on row 0, default 0 3)")
    ap.add_argument("--outlets", type=str, nargs="+", default=["7,0", "7,3", "7,7"],
                    help="Outlet cells on bottom row (default 7,0 7,3 7,7)")
    ap.add_argument("--inlet-flow", type=float, default=DEFAULT_INLET_FLOW,
                    help="Flow injected at the inlet (default 1e-4)")
    ap.add_argument("--targets", type=float, nargs="+", default=None,
                    help="Target outlet flows or flow fractions (normalised to sum to --inlet-flow)")
    ap.add_argument("--max-tiles", type=int, default=25, help="Max tiles placed per episode")
    ap.add_argument("--library", type=str, default=None, help="Path to COMSOL library CSV")
    ap.add_argument("--episodes", type=int, default=2000, help="Total RL training episodes")
    ap.add_argument("--algo", type=str, choices=["ppo", "reinforce"], default="ppo",
                    help="RL algorithm: 'ppo' (physics-aware GAE-PPO, default) or 'reinforce'")
    ap.add_argument("--batch", type=int, default=8, help="Rollouts per update (for PPO) or batch size (for REINFORCE)")
    ap.add_argument("--k-epochs", type=int, default=4, help="PPO mini-batch epochs per update (default 4)")
    ap.add_argument("--lr", type=float, default=None, help="Learning rate (default 0.003 for PPO, 0.01 for REINFORCE)")
    ap.add_argument("--entropy", type=float, default=0.01, help="Entropy regularisation coefficient (start)")
    ap.add_argument("--entropy-final", type=float, default=None,
                    help="Final entropy coefficient for linear decay (default None = constant)")
    ap.add_argument("--hidden", type=int, default=128, help="MLP hidden layer dimension (default 128)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--shaping", type=float, default=20.0, help="Potential-based progress shaping weight")
    ap.add_argument("--lam-conn", type=float, default=0.5, help="Connectivity progress weight")
    ap.add_argument("--lam-balance", type=float, default=0.3, help="Resistance balance reward weight")
    ap.add_argument("--curriculum-mode", type=str, choices=["sequential", "mixed", "off"],
                    default="sequential", help="Curriculum mode: 'sequential' (default), 'mixed', or 'off'")
    ap.add_argument("--no-curriculum", action="store_true", help="Disable staged curriculum learning (alias for --curriculum-mode off)")
    ap.add_argument("--cur-window", type=int, default=50, help="Curriculum sliding window size (default 50)")
    ap.add_argument("--cur-threshold", type=float, default=0.4, help="Curriculum success rate threshold (default 0.4)")
    ap.add_argument("--cur-q-gate", type=float, default=0.15, help="Curriculum flow error limit for good episode (default 0.15)")
    ap.add_argument("--cur-min-stage-episodes", type=int, default=100, help="Minimum episodes before stage advance (default 100)")
    ap.add_argument("--cur-max-stage-fraction", type=float, default=0.25, help="Maximum episode fraction per stage before timeout advance (default 0.25)")
    ap.add_argument("--cur-final-normal-fraction", type=float, default=0.4, help="Guaranteed fraction of episodes at normal/empty stage (default 0.4)")
    ap.add_argument("--cur-hard", action="store_true", help="Enable optional stage 3 (hard mode with tightened tolerance)")
    ap.add_argument("--probe-every", type=int, default=25, help="Updates between read-only evaluation probes (default 25 updates = 200 eps)")
    ap.add_argument("--probe-episodes", type=int, default=16, help="Episodes per evaluation probe (default 16, 0 to disable)")
    ap.add_argument("--best-min-stage", type=int, default=0, help="Minimum curriculum stage required for best_board eligibility (default 0)")
    ap.add_argument("--allow-open-ends", action="store_true", help="Allow open/dangling ends (sets enforce_closure=False)")
    ap.add_argument("--closure-node-limit", type=int, default=4000, help="Oracle node limit for closure search (default 4000)")
    ap.add_argument("--strict-closure", action="store_true", help="Crash immediately on any closure invariant violation")

    # Structural objective preset and overrides (Step 6)
    ap.add_argument("--objective", type=str, choices=list(PRESETS.keys()), default="compact",
                    help="Objective preset: 'compact' (default), 'legacy', 'low-pressure', or 'tree'")
    ap.add_argument("--lam-extra-tiles", type=float, default=None,
                    help="Override penalty weight for tiles beyond n_min")
    ap.add_argument("--lam-pressure", type=float, default=None,
                    help="Override penalty weight for inlet pressure ratio")
    ap.add_argument("--lam-loops", type=float, default=None,
                    help="Override penalty weight for graph cycles")
    ap.add_argument("--allow-large-struct", action="store_true",
                    help="Bypass structural weight budget limit with warning")
    ap.add_argument("--tile-potential", type=str, choices=["terminal", "dense"], default="terminal",
                    help="Tile count penalty mode: 'terminal' (default) or 'dense' (Step 8a)")
    ap.add_argument("--lam-robust", type=float, default=0.0,
                    help="Robustness error weight (default 0.0)")
    ap.add_argument("--robust-eps", type=float, default=0.10,
                    help="Channel perturbation radius for robustness (default 0.10)")
    ap.add_argument("--robust-samples", type=int, default=8,
                    help="Number of perturbed copies to evaluate for robustness (default 8)")

    # Learning signal & Archive & Evaluation (Steps 2, 5)
    ap.add_argument("--adv-norm", type=str, choices=["batch", "episode"], default="batch",
                    help="Advantage normalization: 'batch' (default) or 'episode' (legacy)")
    ap.add_argument("--archive-k", type=int, default=5,
                    help="Top-K archive size for distinct topologies (default 5)")
    ap.add_argument("--eval-episodes", type=int, default=200,
                    help="Fixed post-training evaluation episodes (default 200, 0 to disable)")

    # Automated COMSOL Validation
    ap.add_argument("--no-validate", action="store_true",
                    help="Disable automatic COMSOL / design validation after RL synthesis")
    ap.add_argument("--validate-backend", choices=["comsol", "raster", "mock", "none"], default="comsol",
                    help="Validation backend: 'comsol' (default), 'raster', 'mock', or 'none'")
    ap.add_argument("--comsol-plugins", type=str, default="C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins",
                    help="Path to COMSOL plugins directory (default: C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins)")
    ap.add_argument("--validate-out", type=str, default=None,
                    help="Output folder for validation results (default: <out>/comsol_check)")
    ap.add_argument("--validate-tol", type=float, default=0.05,
                    help="Flow rate tolerance for validation (default: 0.05)")
    ap.add_argument("--validate-timeout", type=int, default=7200,
                    help="Timeout in seconds for COMSOL solver (default: 7200)")
    ap.add_argument("--save-mph", action="store_true",
                    help="Save COMSOL .mph file upon successful validation solve")
    ap.add_argument("--allow-baseline-library", action="store_true",
                    help="Allow validation when using baseline synthetic library")

    ap.add_argument("--out", type=str, default="results/stage1_v3", help="Output directory")
    return ap.parse_args(argv)


def render_board_ascii(spec: DesignSpec, board: Dict[Tuple[int, int], PlacedTile]) -> str:
    """ASCII visualization of the 8x8 tile chip layout."""
    n = spec.grid_size
    lines = []
    lines.append("+" + "---+" * n)
    for r in range(n):
        row_str = "|"
        for c in range(n):
            if (r, c) == tuple(spec.inlet_cell):
                sym = " IN"
            elif (r, c) in set(spec.outlet_cells):
                sym = "OUT"
            elif (r, c) in board:
                v = board[(r, c)].variant
                in_str = "".join(v.inlet_sides)
                out_str = "".join(v.outlet_sides)
                sym = f"{in_str}>{out_str}"[:3].center(3)
            else:
                sym = " . "
            row_str += sym + "|"
        lines.append(row_str)
        lines.append("+" + "---+" * n)
    return "\n".join(lines)


def format_design_export(
    board: Dict[Tuple[int, int], PlacedTile],
    spec: DesignSpec,
    library: Library,
    env: TopologyEnv,
    eligible: bool,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    ev = env.evaluator.evaluate(board)
    open_ports = find_open_ports(board, spec)
    bad_degrees = check_port_degrees(board, spec)

    tiles_export = []
    for (r, c), pt in sorted(board.items()):
        combo = (tuple(sorted(pt.variant.inlet_sides)), tuple(sorted(pt.variant.outlet_sides)))
        combo_vars = library.by_combo.get(combo, [])
        k_tot = len(combo_vars)
        v_idx = 0
        for idx, cv in enumerate(combo_vars):
            if cv.variant_id == pt.variant.variant_id:
                v_idx = idx
                break
        tiles_export.append({
            "cell": [int(r), int(c)],
            "variant_id": pt.variant.variant_id,
            "variant_index": v_idx,
            "k_total": k_tot,
            "inlet_sides": list(pt.variant.inlet_sides),
            "outlet_sides": list(pt.variant.outlet_sides),
            "R_per_side": dict(pt.variant.R_per_side),
        })

    return {
        "spec": spec.to_dict(),
        "tiles": tiles_export,
        "topo_id": topo_id(board),
        "design_id": design_id(board),
        "score": float(ev.score),
        "eligible": bool(eligible),
        "n_min": int(getattr(env, "n_min", 17)),
        "p_ref": float(getattr(env, "p_ref", 1.0)),
        "curriculum": {
            "found_in_stage": meta.get("found_in_stage", 2),
            "preplaced_cells": [list(c) for c in getattr(env, "preplaced", ())],
        },
        "closure": {
            "closed": len(open_ports) == 0 and len(bad_degrees) == 0,
            "open_ports": [
                {"cell": list(c), "side": s, "kind": k} for c, s, k in open_ports
            ],
        },
        "metrics": ev.metrics(),
        "realized_flows": list(ev.solve.outlet_flows),
    }


def main(argv=None):
    args = parse_args(argv)
    outlets = parse_ports(args.outlets)

    spec = DesignSpec(
        grid_size=args.grid,
        inlet_cell=tuple(args.inlet),
        outlet_cells=outlets,
        inlet_flow=args.inlet_flow,
        max_tiles=args.max_tiles,
    )

    if args.targets:
        if len(args.targets) != len(spec.outlet_cells):
            raise SystemExit(f"--targets must provide {len(spec.outlet_cells)} values, got {len(args.targets)}")
        spec.target_flows = normalize_targets(args.targets, spec.inlet_flow)
    else:
        eq = spec.inlet_flow / len(spec.outlet_cells)
        spec.target_flows = [eq] * len(spec.outlet_cells)

    spec.validate()

    # Load library or generate baseline
    if args.library and os.path.exists(args.library):
        print(f"Loading COMSOL library from: {args.library}")
        library = Library.from_csv(args.library)
    else:
        if args.library:
            print(f"Warning: {args.library} not found; generating baseline library...")
        else:
            print("No --library passed; generating baseline library...")
        library = Library.generate_initial_library()

    print(f"Library loaded: {len(library)} tile variants across {len(library.by_combo)} side combos.")
    print("Inlet:", spec.inlet_cell, "| Outlets:", spec.outlet_cells)
    print("Target flow fractions:", [round(q / spec.inlet_flow, 4) for q in spec.target_flows])

    # Reward weights setup with presets and overrides (Step 6)
    weights = RewardWeights(
        shaping=args.shaping,
        lam_conn=args.lam_conn,
        lam_balance=args.lam_balance,
    )
    apply_preset(weights, args.objective)
    if args.lam_extra_tiles is not None:
        weights.lam_extra_tiles = float(args.lam_extra_tiles)
    if args.lam_pressure is not None:
        weights.lam_pressure = float(args.lam_pressure)
    if args.lam_loops is not None:
        weights.lam_loops = float(args.lam_loops)
    weights.lam_robust = float(args.lam_robust)
    weights.robust_eps = float(args.robust_eps)
    weights.robust_samples = int(args.robust_samples)

    env = TopologyEnv(
        spec=spec,
        library=library,
        weights=weights,
        enforce_closure=not args.allow_open_ends,
        closure_node_limit=args.closure_node_limit,
        strict_closure=args.strict_closure,
        allow_large_struct=args.allow_large_struct,
        tile_potential=args.tile_potential,
    )

    # Save config.json at startup
    os.makedirs(args.out, exist_ok=True)
    config_dict = vars(args).copy()
    config_dict["git_hash"] = get_git_hash()
    config_dict["timestamp"] = datetime.now().isoformat()
    config_dict["n_min"] = int(env.n_min)
    config_dict["p_ref"] = float(env.p_ref)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config_dict, fh, indent=2)

    lr = args.lr if args.lr is not None else (0.003 if args.algo == "ppo" else 0.01)
    cur_mode = "off" if (args.no_curriculum or args.episodes < 100) else args.curriculum_mode
    meta = {}

    if args.algo == "ppo":
        print(f"\nStarting Stage 1 PPO training ({args.episodes} episodes, rollouts/update {args.batch}, K_epochs {args.k_epochs})...\n")
        net, value_net, best_board, best_reward, history, meta = train_agent_ppo(
            env=env,
            episodes=args.episodes,
            rollouts_per_update=args.batch,
            K_epochs=args.k_epochs,
            lr_policy=lr,
            entropy_coef=args.entropy,
            entropy_final=args.entropy_final,
            hidden=args.hidden,
            seed=args.seed,
            use_curriculum=(cur_mode != "off"),
            curriculum_mode=cur_mode,
            cur_window=args.cur_window,
            cur_threshold=args.cur_threshold,
            cur_q_gate=args.cur_q_gate,
            cur_min_stage_episodes=args.cur_min_stage_episodes,
            cur_max_stage_fraction=args.cur_max_stage_fraction,
            cur_final_normal_fraction=args.cur_final_normal_fraction,
            cur_hard=args.cur_hard,
            probe_every=args.probe_every,
            probe_episodes=args.probe_episodes,
            best_min_stage=args.best_min_stage,
            adv_norm=args.adv_norm,
            archive_k=args.archive_k,
            eval_episodes=args.eval_episodes,
        )
    else:
        print(f"\nStarting Stage 1 REINFORCE training ({args.episodes} episodes, batch {args.batch})...\n")
        net, best_board, best_reward, history, meta = train_agent(
            env=env,
            episodes=args.episodes,
            batch_size=args.batch,
            lr=lr,
            entropy_coef=args.entropy,
            entropy_final=args.entropy_final,
            hidden=args.hidden,
            seed=args.seed,
            archive_k=args.archive_k,
            eval_episodes=args.eval_episodes,
        )
        value_net = None

    ev = env.evaluator.evaluate(best_board)
    q0 = spec.inlet_flow

    open_ports = find_open_ports(best_board, spec)
    bad_degrees = check_port_degrees(best_board, spec)
    if not args.allow_open_ends and (open_ports or bad_degrees):
        print("\nFATAL: Closed-topology invariant violated in final best design!", file=sys.stderr)
        if open_ports:
            print(f"Open ports found ({len(open_ports)}): {open_ports}", file=sys.stderr)
        if bad_degrees:
            print(f"Nodes with invalid degree ({len(bad_degrees)}): {bad_degrees}", file=sys.stderr)
        sys.exit(2)

    print("\n================= BEST SYNTHESIZED DESIGN =================")
    print(render_board_ascii(spec, best_board))
    print("Target flows   :", [round(q, 8) for q in spec.target_flows])
    print("Realized flows :", [round(q, 8) for q in ev.solve.outlet_flows])
    print("Target fractions  :", [round(q / q0, 4) for q in spec.target_flows])
    print("Realized fractions:", [round(q / q0, 4) for q in ev.solve.outlet_flows])
    print("Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in ev.metrics().items()})

    eligible = meta.get("eligible", False)
    design_export = format_design_export(best_board, spec, library, env, eligible, meta)

    # Save primary outputs
    with open(os.path.join(args.out, "design.json"), "w", encoding="utf-8") as fh:
        json.dump(design_export, fh, indent=2)

    with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    archive = meta.get("archive", [])
    with open(os.path.join(args.out, "archive.json"), "w", encoding="utf-8") as fh:
        json.dump(archive, fh, indent=2)

    eval_summary = meta.get("eval_summary", {})
    with open(os.path.join(args.out, "eval_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(eval_summary, fh, indent=2)

    # Save rank 2..K designs
    for rank_idx, entry in enumerate(archive[1:], start=2):
        r_board = tiles_to_board(entry["tiles"], library)
        r_export = format_design_export(r_board, spec, library, env, eligible=True, meta=meta)
        rank_filename = f"design_rank{rank_idx}.json"
        with open(os.path.join(args.out, rank_filename), "w", encoding="utf-8") as fh:
            json.dump(r_export, fh, indent=2)

    net.save(os.path.join(args.out, "policy.npz"))
    if value_net is not None:
        value_net.save(os.path.join(args.out, "value.npz"))
    library.to_csv(os.path.join(args.out, "library.csv"))

    # Automatically generate high-resolution chip visualization PNG and standalone learning curves
    try:
        from .visualize_stage1 import plot_training_curves, render_chip_image
        render_chip_image(design_export, os.path.join(args.out, "chip_layout.png"))
        plot_training_curves(
            history,
            out_dir_or_png=args.out,
            title_prefix=f"Targets: {[round(q/spec.inlet_flow, 4) for q in spec.target_flows]}",
        )
    except Exception as e:
        print(f"Note: Could not generate visualization images: {e}")

    # Automatically validate design against COMSOL (or specified backend)
    if not args.no_validate:
        print("\n" + "=" * 65)
        print("       AUTOMATIC DESIGN VALIDATION (COMSOL)")
        print("=" * 65)
        try:
            from .validate_design import validate_rl_design
            val_out = args.validate_out or os.path.join(args.out, "comsol_check")
            validate_rl_design(
                design_path=os.path.join(args.out, "design.json"),
                output_dir=val_out,
                backend=args.validate_backend,
                plugins_path=args.comsol_plugins,
                tol=args.validate_tol,
                library_path=args.library,
                timeout_s=args.validate_timeout,
                allow_baseline_library=args.allow_baseline_library or (args.library is None),
                save_mph=args.save_mph,
                verbose=True,
            )
            print(f"\n[validate] Automated validation results saved to: {os.path.abspath(val_out)}")
        except Exception as e:
            print(f"\n[validate] Note: Automated validation could not complete: {e}")

    print(f"\nSaved Stage 1 results to: {os.path.abspath(args.out)}")
    return design_export


if __name__ == "__main__":
    main()
