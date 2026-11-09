"""Reference-optimum solver for the 17-tile minimum skeleton across the 12 monotone staircase routes.

Answers: for a given target vector, what is the best flow error achievable with the minimum skeleton?
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from PhysicsAwareRL.config import DEFAULT_INLET_FLOW, DesignSpec, RewardWeights, normalize_targets
from PhysicsAwareRL.hydraulic_solver import solve_network, solve_network_v2
from PhysicsAwareRL.library import Library, PlacedTile, TileVariant


class RouteTile(NamedTuple):
    cell: Tuple[int, int]
    inlet_sides: Tuple[str, ...]
    outlet_sides: Tuple[str, ...]

    @property
    def combo(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        return (tuple(sorted(self.inlet_sides)), tuple(sorted(self.outlet_sides)))


def get_monotone_routes(spec: DesignSpec) -> Tuple[List[List[RouteTile]], List[List[RouteTile]]]:
    """Enumerate 3 left routes (4 tiles) and 4 right routes (5 tiles) for the default 8x8 spec."""
    grid_n = spec.grid_size
    r_in, c_in = spec.inlet_cell
    outlets = sorted([tuple(rc) for rc in spec.outlet_cells], key=lambda x: x[1])

    if r_in != 0 or len(outlets) != 3:
        print("unsupported spec", file=sys.stderr)
        sys.exit(1)

    left_out, mid_out, right_out = outlets
    if mid_out != (grid_n - 1, c_in) or left_out[0] != grid_n - 1 or right_out[0] != grid_n - 1:
        print("unsupported spec", file=sys.stderr)
        sys.exit(1)
    if not (left_out[1] < c_in < right_out[1]):
        print("unsupported spec", file=sys.stderr)
        sys.exit(1)

    r_split = grid_n - 2  # row 6
    r_sink = grid_n - 1   # row 7

    # Left arm: from (r_split, c_in) West to left_out (r_sink, c_left)
    # Step down can happen at c_step in [c_in - 1, ..., left_out[1]]
    left_routes: List[List[RouteTile]] = []
    c_left = left_out[1]
    for c_step in range(c_in - 1, c_left - 1, -1):
        tiles: List[RouteTile] = []
        # Row 6: from c_in - 1 down to c_step
        for c in range(c_in - 1, c_step, -1):
            tiles.append(RouteTile((r_split, c), ("E",), ("W",)))
        # Turn down at c_step
        tiles.append(RouteTile((r_split, c_step), ("E",), ("S",)))
        # Row 7: at c_step
        if c_step == c_left:
            # Reached destination column on step down
            tiles.append(RouteTile((r_sink, c_left), ("N",), ("S",)))
        else:
            tiles.append(RouteTile((r_sink, c_step), ("N",), ("W",)))
            for c in range(c_step - 1, c_left, -1):
                tiles.append(RouteTile((r_sink, c), ("E",), ("W",)))
            tiles.append(RouteTile((r_sink, c_left), ("E",), ("S",)))
        left_routes.append(tiles)

    # Right arm: from (r_split, c_in) East to right_out (r_sink, c_right)
    # Step down can happen at c_step in [c_in + 1, ..., right_out[1]]
    right_routes: List[List[RouteTile]] = []
    c_right = right_out[1]
    for c_step in range(c_in + 1, c_right + 1):
        tiles = []
        # Row 6: from c_in + 1 up to c_step
        for c in range(c_in + 1, c_step):
            tiles.append(RouteTile((r_split, c), ("W",), ("E",)))
        # Turn down at c_step
        tiles.append(RouteTile((r_split, c_step), ("W",), ("S",)))
        # Row 7: at c_step
        if c_step == c_right:
            tiles.append(RouteTile((r_sink, c_right), ("N",), ("S",)))
        else:
            tiles.append(RouteTile((r_sink, c_step), ("N",), ("E",)))
            for c in range(c_step + 1, c_right):
                tiles.append(RouteTile((r_sink, c), ("W",), ("E",)))
            tiles.append(RouteTile((r_sink, c_right), ("W",), ("S",)))
        right_routes.append(tiles)

    return left_routes, right_routes


def build_arm_combinations(
    route: List[RouteTile],
    library: Library,
) -> List[Tuple[float, List[TileVariant]]]:
    """Find achievable resistance sums and variant assignments for a route using dynamic programming."""
    # dp: list of (sum_r, [variants])
    dp: List[Tuple[float, List[TileVariant]]] = [(0.0, [])]

    for rt in route:
        combo = rt.combo
        variants = library.by_combo.get(combo, [])
        if not variants:
            raise ValueError(f"No library variants for combo {combo}")

        new_dp: Dict[float, Tuple[float, List[TileVariant]]] = {}
        for cur_r, cur_vars in dp:
            for v in variants:
                tot_r = cur_r + v.total_r
                # Round to 4 significant digits to prune duplicates
                key = float(f"{tot_r:.4g}")
                if key not in new_dp or tot_r < new_dp[key][0]:
                    new_dp[key] = (tot_r, cur_vars + [v])
        dp = list(new_dp.values())

    return dp


def solve_reference_optimum(
    spec: DesignSpec,
    library: Library,
    target_fractions: List[float],
    weights: Optional[RewardWeights] = None,
) -> Dict[str, Any]:
    """Find the optimal variant combination for the minimum 17-tile skeleton."""
    weights = weights or RewardWeights()
    left_routes, right_routes = get_monotone_routes(spec)
    n_routes = len(left_routes) * len(right_routes)

    q_in = spec.inlet_flow
    q_targets = np.array(target_fractions, dtype=float) * q_in
    denom = np.maximum(q_targets, 0.01 * q_in)

    # 1. Split tile variants
    split_combo = (("N",), ("E", "S", "W"))
    split_variants = library.by_combo.get(split_combo, [])
    if not split_variants:
        raise ValueError(f"No library variants for split tile {split_combo}")

    # 2. Middle arm variants: 1 tile at (7, 3) with combo (('N',), ('S',))
    mid_combo = (("N",), ("S",))
    mid_variants = library.by_combo.get(mid_combo, [])
    if not mid_variants:
        raise ValueError(f"No library variants for middle arm {mid_combo}")

    # Trunk tiles: straight (('N',), ('S',)) with minimum total_r
    best_trunk_v = min(mid_variants, key=lambda v: (v.total_r, v.variant_id))

    # 3. Dynamic programming for left and right routes
    left_arm_options: List[Tuple[int, float, List[TileVariant]]] = []
    for l_idx, l_route in enumerate(left_routes):
        for r_val, v_list in build_arm_combinations(l_route, library):
            left_arm_options.append((l_idx, r_val, v_list))

    right_arm_options: List[Tuple[int, float, List[TileVariant]]] = []
    for r_idx, r_route in enumerate(right_routes):
        for r_val, v_list in build_arm_combinations(r_route, library):
            right_arm_options.append((r_idx, r_val, v_list))

    r_stub = spec.R_stub

    best_eq = float("inf")
    best_choice = None

    # Vectorized evaluation per split variant
    for sv in split_variants:
        r_w = sv.R_per_side["W"]
        r_s = sv.R_per_side["S"]
        r_e = sv.R_per_side["E"]

        for mv in mid_variants:
            R1 = r_s + mv.total_r + r_stub
            g1 = 1.0 / R1

            for l_idx, l_r, l_vars in left_arm_options:
                R0 = r_w + l_r + r_stub
                g0 = 1.0 / R0

                for r_idx, r_r, r_vars in right_arm_options:
                    R2 = r_e + r_r + r_stub
                    g2 = 1.0 / R2

                    g_tot = g0 + g1 + g2
                    q0 = q_in * (g0 / g_tot)
                    q1 = q_in * (g1 / g_tot)
                    q2 = q_in * (g2 / g_tot)

                    q_real = np.array([q0, q1, q2])
                    rel_errs = np.abs(q_real - q_targets) / denom
                    eq = float(np.max(rel_errs))

                    if eq < best_eq:
                        best_eq = eq
                        best_choice = {
                            "eq": eq,
                            "rel_errs": rel_errs.tolist(),
                            "realized_q": q_real.tolist(),
                            "fractions": (q_real / q_in).tolist(),
                            "split_v": sv,
                            "mid_v": mv,
                            "left_route_idx": l_idx,
                            "left_vars": l_vars,
                            "right_route_idx": r_idx,
                            "right_vars": r_vars,
                            "R0": R0,
                            "R1": R1,
                            "R2": R2,
                        }

    assert best_choice is not None, "Failed to find any reference optimum"
    success = bool(best_choice["eq"] <= weights.tolerance)
    needs_extra_tiles = not success

    # Cross-check: build actual board and verify against solve_network_v2 to 1e-9
    grid_n = spec.grid_size
    c_in = spec.inlet_cell[1]
    board: Dict[Tuple[int, int], PlacedTile] = {}

    # Trunk: rows 0 to 5 in column c_in
    for r in range(grid_n - 2):
        board[(r, c_in)] = PlacedTile((r, c_in), best_trunk_v)

    # Split tile at (grid_n - 2, c_in)
    r_split = grid_n - 2
    board[(r_split, c_in)] = PlacedTile((r_split, c_in), best_choice["split_v"])

    # Middle arm at (grid_n - 1, c_in)
    board[(grid_n - 1, c_in)] = PlacedTile((grid_n - 1, c_in), best_choice["mid_v"])

    # Left arm tiles
    l_route = left_routes[best_choice["left_route_idx"]]
    for rt, v in zip(l_route, best_choice["left_vars"]):
        board[rt.cell] = PlacedTile(rt.cell, v)

    # Right arm tiles
    r_route = right_routes[best_choice["right_route_idx"]]
    for rt, v in zip(r_route, best_choice["right_vars"]):
        board[rt.cell] = PlacedTile(rt.cell, v)

    sr = solve_network_v2(board, spec)
    assert sr.ok and not sr.stranded and sr.connected_outlets.all(), "Cross-check solve failed"

    sim_fractions = sr.outlet_flows / q_in
    tool_fractions = np.array(best_choice["fractions"])
    diff = np.max(np.abs(sim_fractions - tool_fractions))
    if diff > 1e-9:
        raise AssertionError(
            f"Cross-check failed: difference between tool model and hydraulic solver is {diff:g} > 1e-9!\n"
            f"Tool: {tool_fractions}\nSolver: {sim_fractions}"
        )

    all_variant_ids = (
        [best_trunk_v.variant_id] * (grid_n - 2)
        + [best_choice["split_v"].variant_id, best_choice["mid_v"].variant_id]
        + [v.variant_id for v in best_choice["left_vars"]]
        + [v.variant_id for v in best_choice["right_vars"]]
    )

    return {
        "target_fractions": target_fractions,
        "n_routes": n_routes,
        "best_e_q": float(best_choice["eq"]),
        "rel_errs": [float(e) for e in best_choice["rel_errs"]],
        "realized_fractions": [float(f) for f in best_choice["fractions"]],
        "success": success,
        "needs_extra_tiles": needs_extra_tiles,
        "variant_ids": all_variant_ids,
        "board": board,
    }


def main():
    parser = argparse.ArgumentParser(description="Reference Optimum Tool for Minimum Skeleton (17 tiles)")
    parser.add_argument("--library", type=str, default=None, help="Path to COMSOL library CSV")
    parser.add_argument("--out", type=str, default="results/reference_optimum.csv", help="Output CSV path")
    args = parser.parse_args()

    spec = DesignSpec()
    if args.library and os.path.exists(args.library):
        lib = Library.from_csv(args.library)
    else:
        lib = Library.generate_initial_library()

    targets_to_run = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.5, 0.3, 0.2],
        [0.2, 0.3, 0.5],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ]

    left_routes, right_routes = get_monotone_routes(spec)
    print(f"Supported geometry verified: 8x8 grid, 1 inlet (0, 3), 3 outlets.")
    print(f"Number of routes: {len(left_routes) * len(right_routes)} (3 left x 4 right).")

    results = []
    for tgt in targets_to_run:
        spec.target_flows = normalize_targets(tgt, spec.inlet_flow)
        res = solve_reference_optimum(spec, lib, tgt)
        print(
            f"Target: {tgt} | Best E_Q: {res['best_e_q']:.4f} | "
            f"Realized: {[round(x, 4) for x in res['realized_fractions']]} | "
            f"Success: {res['success']} | Needs extra tiles: {res['needs_extra_tiles']}"
        )
        results.append({
            "target": str(tgt),
            "best_e_q": f"{res['best_e_q']:.6f}",
            "success": str(res['success']),
            "needs_extra_tiles": str(res['needs_extra_tiles']),
            "err_out0": f"{res['rel_errs'][0]:.6f}",
            "err_out1": f"{res['rel_errs'][1]:.6f}",
            "err_out2": f"{res['rel_errs'][2]:.6f}",
            "realized_frac0": f"{res['realized_fractions'][0]:.6f}",
            "realized_frac1": f"{res['realized_fractions'][1]:.6f}",
            "realized_frac2": f"{res['realized_fractions'][2]:.6f}",
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved reference optimum table to: {os.path.abspath(args.out)}")
    print("Cross-check passed with max difference < 1e-9.")


if __name__ == "__main__":
    main()
