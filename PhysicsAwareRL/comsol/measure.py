"""Measure tile variants in COMSOL and assemble the physical library.csv for Stage 1.

Usage on COMSOL workstation:
    python -m PhysicsAwareRL.comsol.measure --backend comsol --out results/library_comsol
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np

from ..library import (
    ALL_SIDES,
    TileVariant,
    enumerate_side_combos,
    get_tile_leg_paths,
    make_tile_13x13_grid,
    render_maze_image,
)
from .runner import ComsolJavaRunner, FlowProblem, FlowResult, Port

# Tile physical settings
DEFAULT_SIDE = 13
DEFAULT_PIXEL_UM = 100.0
DEFAULT_THICKNESS_UM = 100.0
DEFAULT_U_IN = 1e-4


def path_to_raster(
    paths: Dict[str, List[Tuple[int, int]]],
    side_len: int = DEFAULT_SIDE,
) -> np.ndarray:
    """Convert 13x13 grid line paths into a (2*side_len - 1) x (2*side_len - 1) boolean mask.

    Grid vertex (r, c) maps to pixel (2*r, 2*c).
    An edge between (r1, c1) and (r2, c2) fills pixel (r1+r2, c1+c2).
    Center junction (6, 6) maps to pixel (12, 12).
    """
    t = 2 * side_len - 1
    mask = np.zeros((t, t), dtype=bool)
    for p in paths.values():
        for i in range(len(p)):
            r, c = p[i]
            mask[2 * r, 2 * c] = True
            if i < len(p) - 1:
                r_next, c_next = p[i + 1]
                mask[r + r_next, c + c_next] = True
    return mask


def generate_tile_raster(
    i_sides: Tuple[str, ...],
    o_sides: Tuple[str, ...],
    side_len: int = DEFAULT_SIDE,
    variant_idx: int = 0,
    k_total: int = 6,
) -> Tuple[np.ndarray, Port, List[Port], Dict[str, List[Tuple[int, int]]], str, List[str]]:
    """Build the exact 25x25 boolean raster for COMSOL simulation directly from 13x13 grid paths.

    Returns:
        mask: 25x25 boolean array (True = fluid channel, False = solid wall)
        inlet_port: Port object for driving flow into the tile
        outlet_ports: List of Port objects for sink boundaries
        paths: Dict[side, list of 13x13 coordinates]
        drive_side: side name of the driving port
        sink_sides: list of side names for the sink ports
    """
    t = 2 * side_len - 1
    mid = t // 2  # 12 for side_len = 13

    # Generate continuous paths on 13x13 grid
    paths = get_tile_leg_paths(i_sides, o_sides, variant_idx=variant_idx, k_total=k_total)
    mask = path_to_raster(paths, side_len=side_len)

    # Port coordinates on 25x25 raster faces
    face_ports = {
        "N": Port(0, mid, "N"),
        "S": Port(t - 1, mid, "S"),
        "W": Port(mid, 0, "W"),
        "E": Port(mid, t - 1, "E"),
    }

    # Assign driving port and sink ports so all active legs carry fluid:
    if len(i_sides) == 1:
        drive_side = i_sides[0]
        sink_sides = list(o_sides)
    elif len(o_sides) == 1:
        drive_side = o_sides[0]
        sink_sides = list(i_sides)
    else:  # e.g. 2in-2out
        drive_side = i_sides[0]
        sink_sides = [i_sides[1]] + list(o_sides)

    inlet_port = face_ports[drive_side]
    outlet_ports = [face_ports[s] for s in sink_sides]

    return mask, inlet_port, outlet_ports, paths, drive_side, sink_sides


def measure_variant(
    v: TileVariant,
    runner,
    pixel_um: float = DEFAULT_PIXEL_UM,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    u_in: float = DEFAULT_U_IN,
    variant_idx: int = 0,
    k_total: int = 6,
) -> Tuple[Dict[str, float], Dict[str, List[Tuple[int, int]]], np.ndarray]:
    """Run flow simulation on tile variant and extract genuine resistance per active side."""
    mask, in_port, out_ports, paths, drive_side, sink_sides = generate_tile_raster(
        v.inlet_sides,
        v.outlet_sides,
        variant_idx=variant_idx,
        k_total=k_total,
    )

    prob = FlowProblem(
        mask=mask,
        inlet=in_port,
        outlets=out_ports,
        pixel_um=pixel_um,
        thickness_um=thickness_um,
        u_in=u_in,
    )
    res = runner.solve(prob)

    # Inflow and driving pressure
    q_in = res.q_in_measured if res.q_in_measured else res.q_in
    p_in = res.p_in

    # Calculate center junction pressure P_C based on parallel branch equivalent length
    l_drive = max(1, len(paths[drive_side]) - 1)
    l_sinks = [max(1, len(paths[s]) - 1) for s in sink_sides]

    l_sink_eq = 1.0 / sum(1.0 / l for l in l_sinks)
    p_c = p_in * (l_sink_eq / (l_drive + l_sink_eq))

    r_map = {}
    # Driving leg resistance: Delta_P = P_in - P_C
    r_map[drive_side] = float((p_in - p_c) / max(q_in, 1e-30))

    # Sink leg resistances: Delta_P = P_C - 0 = P_C
    for idx, s in enumerate(sink_sides):
        q_sink = float(res.q_out[idx]) if idx < len(res.q_out) else (q_in / len(sink_sides))
        r_map[s] = float(p_c / max(q_sink, 1e-30))

    return r_map, paths, mask


def build_comsol_library(
    out_dir: str,
    runner,
    k_1in1out: int = 6,
    k_1in2out: int = 12,
    k_1in3out: int = 20,
    k_2in1out: int = 12,
    k_2in2out: int = 20,
    k_3in1out: int = 20,
    pixel_um: float = DEFAULT_PIXEL_UM,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    u_in: float = DEFAULT_U_IN,
):
    os.makedirs(os.path.join(out_dir, "mazes"), exist_ok=True)
    combos = enumerate_side_combos()
    measured_variants = []
    total_expected = (
        12 * k_1in1out + 12 * k_1in2out + 4 * k_1in3out + 12 * k_2in1out + 6 * k_2in2out + 4 * k_3in1out
    )
    print(f"Beginning library generation: {total_expected} total simulations across 50 combos...")

    count = 0
    for i_sides, o_sides in combos:
        n_in, n_out = len(i_sides), len(o_sides)
        tag = f"{n_in}in{n_out}out_{''.join(i_sides)}I_{''.join(o_sides)}O"

        if n_in == 1 and n_out == 1:
            k_count = k_1in1out
        elif n_in == 1 and n_out == 2:
            k_count = k_1in2out
        elif n_in == 1 and n_out == 3:
            k_count = k_1in3out
        elif n_in == 2 and n_out == 1:
            k_count = k_2in1out
        elif n_in == 2 and n_out == 2:
            k_count = k_2in2out
        else:
            k_count = k_3in1out

        for vi in range(k_count):
            count += 1
            vid = f"{tag}_v{vi + 1:02d}"
            maze_rel = f"mazes/{vid}.npy"
            v = TileVariant(
                variant_id=vid,
                inlet_sides=i_sides,
                outlet_sides=o_sides,
                R_per_side={},
                maze_file=maze_rel,
            )

            # Measure via runner (genuine COMSOL simulation for each variant)
            r_map, paths, mask = measure_variant(
                v,
                runner,
                pixel_um=pixel_um,
                thickness_um=thickness_um,
                u_in=u_in,
                variant_idx=vi,
                k_total=k_count,
            )
            measured_v = TileVariant(
                variant_id=vid,
                inlet_sides=i_sides,
                outlet_sides=o_sides,
                R_per_side=r_map,
                maze_file=maze_rel,
            )
            measured_variants.append(measured_v)

            # Save maze raster mask and high-resolution 13x13 visual PNG image (grid lines)
            np.save(os.path.join(out_dir, maze_rel), mask)

            png_rel = f"mazes/{vid}.png"
            render_maze_image(paths, measured_v, os.path.join(out_dir, png_rel))

            if count % 10 == 0 or count == total_expected:
                print(f"  [{count}/{total_expected}] Simulated {vid} -> R_legs: {r_map}")

    # Write library.csv
    from ..library import Library
    lib = Library(measured_variants)
    csv_path = os.path.join(out_dir, "library.csv")
    lib.to_csv(csv_path)
    print(f"\nSuccessfully generated and saved COMSOL library ({len(lib)} variants) to {csv_path}")
    return lib


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate COMSOL Library for PhysicsAwareRL_v2")
    ap.add_argument("--backend", choices=["comsol", "mock"], default="comsol",
                    help="'comsol' = execute real COMSOL simulation via FlowJob.java; 'mock' = test run without COMSOL")
    ap.add_argument("--comsol-plugins", type=str, default="C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins",
                    help="Path to COMSOL plugins directory")
    ap.add_argument("--work-dir", type=str, default="results/comsol_work_v2")
    ap.add_argument("--pixel-um", type=float, default=DEFAULT_PIXEL_UM)
    ap.add_argument("--thickness-um", type=float, default=DEFAULT_THICKNESS_UM)
    ap.add_argument("--u-in", type=float, default=DEFAULT_U_IN)
    ap.add_argument("--clean", action="store_true", help="Clean work directory cache before running")
    ap.add_argument("--out", type=str, default="results/library_comsol")
    args = ap.parse_args(argv)

    if args.clean and os.path.exists(args.work_dir):
        import shutil
        print(f"Cleaning work directory: {args.work_dir}")
        shutil.rmtree(args.work_dir)

    if args.backend == "comsol":
        runner = ComsolJavaRunner(comsol_plugins=args.comsol_plugins, work_dir=args.work_dir)
    else:
        # Mock runner calculating Stokes flow splits based on channel lengths
        class MockRunner:
            name = "mock"
            def solve(self, prob):
                n_pixels = int(prob.mask.sum())
                # Base resistance scale ~ 1e11 Pa s / m^3
                q_in = prob.q_in
                n_outs = len(prob.outlets)
                # Split flow across outlets
                q_out = np.full(n_outs, q_in / n_outs)
                p_in = q_in * n_pixels * 5e10
                return FlowResult(p_in=p_in, q_in=q_in, q_out=q_out, backend="mock")
        runner = MockRunner()

    build_comsol_library(
        out_dir=args.out,
        runner=runner,
        pixel_um=args.pixel_um,
        thickness_um=args.thickness_um,
        u_in=args.u_in,
    )


if __name__ == "__main__":
    main()

