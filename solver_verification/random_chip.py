"""Random Chip Generator for Stage 1 Hydraulic Solver v2 Verification.

Generates a connected 8x8 tile chip connecting:
  - Fixed single inlet at (0, 3) North face
  - Fixed 3 outlets at (7, 0), (7, 3), and (7, 7) South face
Assigns physical leg lengths (600 to 3000 um) and resistances matching the tile library.
"""
from __future__ import annotations

import os
import random
from collections import deque
from typing import Dict, List, Set, Tuple

# Physical constants for water in 100 um x 100 um microchannels
VISCOSITY = 1.01e-3  # Pa s
WIDTH_M = 100e-6     # m
DEPTH_M = 100e-6     # m
# Hydraulic resistance per micrometer of channel length (Pa s / m^3 / um)
R_PER_UM = (12.0 * VISCOSITY / (WIDTH_M * (DEPTH_M**3) * (1.0 - 0.63 * DEPTH_M / WIDTH_M))) * 1e-6

DEFAULT_PITCH_UM = 1200.0
DEFAULT_CHANNEL_WIDTH_UM = 100.0
DEFAULT_L_STUB_UM = 600.0
DEFAULT_INLET_FLOW = 1e-4

ALL_SIDES = ("N", "S", "E", "W")
DELTA_TO_SIDE = {
    (-1, 0): "N",
    (1, 0): "S",
    (0, -1): "W",
    (0, 1): "E",
}
SIDE_TO_DELTA = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}


def generate_random_chip(
    layout_file: str = "chip_layout.txt",
    ratio_file: str = "ratios.txt",
    random_lengths: bool = True,
    random_ratios: bool = False,
    seed: int | None = None,
) -> Dict:
    """Generate a connected 8x8 microfluidic chip layout and write to files."""
    if seed is not None:
        random.seed(seed)

    grid_size = 8
    inlet = (0, 3)
    outlets = [(7, 0), (7, 3), (7, 7)]

    # 1. Build a connected acyclic/branching network of cells connecting inlet to all 3 outlets
    tree_nodes: Set[Tuple[int, int]] = {inlet}
    tree_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    def add_edge(u: Tuple[int, int], v: Tuple[int, int]):
        tree_edges.add((min(u, v), max(u, v)))
        tree_nodes.add(u)
        tree_nodes.add(v)

    def random_walk_to_target(start: Tuple[int, int], target: Tuple[int, int]):
        curr = start
        while curr != target:
            r, c = curr
            tr, tc = target
            candidates = []
            if tr > r: candidates.append((r + 1, c))
            if tr < r: candidates.append((r - 1, c))
            if tc > c: candidates.append((r, c + 1))
            if tc < c: candidates.append((r, c - 1))
            # Occasionally allow a lateral jog
            if random.random() < 0.25:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in candidates:
                        candidates.append((nr, nc))
            nxt = random.choice(candidates)
            add_edge(curr, nxt)
            curr = nxt

    # Path 1: Inlet (0, 3) to Outlet (7, 3)
    random_walk_to_target(inlet, (7, 3))

    # Path 2: Branch to Outlet (7, 0)
    branch_candidates_left = [n for n in tree_nodes if n[0] <= 5]
    branch1 = random.choice(branch_candidates_left)
    random_walk_to_target(branch1, (7, 0))

    # Path 3: Branch to Outlet (7, 7)
    branch_candidates_right = [n for n in tree_nodes if n[0] <= 5]
    branch2 = random.choice(branch_candidates_right)
    random_walk_to_target(branch2, (7, 7))

    # Add a few extra loops (15% chance for adjacent active cells)
    for u in list(tree_nodes):
        r, c = u
        for dr, dc in [(1, 0), (0, 1)]:
            v = (r + dr, c + dc)
            if v in tree_nodes and (min(u, v), max(u, v)) not in tree_edges:
                if random.random() < 0.15:
                    add_edge(u, v)

    # 2. Determine adjacency and flow directions using BFS depth from inlet
    adj_cells: Dict[Tuple[int, int], List[Tuple[int, int]]] = {n: [] for n in tree_nodes}
    for u, v in tree_edges:
        adj_cells[u].append(v)
        adj_cells[v].append(u)

    depth: Dict[Tuple[int, int], int] = {inlet: 0}
    queue = deque([inlet])
    while queue:
        curr = queue.popleft()
        d = depth[curr]
        for nbr in adj_cells[curr]:
            if nbr not in depth:
                depth[nbr] = d + 1
                queue.append(nbr)

    # 3. Construct placed tiles specifications
    tiles_data = []
    length_options = [600.0, 1200.0, 1800.0, 2400.0, 3000.0] if random_lengths else [600.0]

    for (r, c) in sorted(tree_nodes):
        in_sides = []
        out_sides = []

        # Inlet cell (0, 3) has external inlet on North
        if (r, c) == inlet:
            in_sides.append("N")

        # Neighboring cells
        for nbr in adj_cells[(r, c)]:
            dr, dc = nbr[0] - r, nbr[1] - c
            side = DELTA_TO_SIDE[(dr, dc)]
            if depth.get(nbr, 999) < depth.get((r, c), 0):
                # Nbr is upstream -> flow enters from side
                if side not in in_sides:
                    in_sides.append(side)
            else:
                # Nbr is downstream -> flow exits to side
                if side not in out_sides:
                    out_sides.append(side)

        # Outlet cells have external outlet on South
        if (r, c) in outlets:
            if "S" not in out_sides:
                out_sides.append("S")

        # Ensure no overlap between in_sides and out_sides
        out_sides = [s for s in out_sides if s not in in_sides]
        if not in_sides:
            # Fallback if no incoming direction was assigned
            in_sides.append("N")

        # Assign leg lengths (um)
        leg_lengths = {}
        for s in in_sides + out_sides:
            leg_lengths[s] = random.choice(length_options)

        tiles_data.append({
            "row": r,
            "col": c,
            "inlet_sides": tuple(sorted(in_sides)),
            "outlet_sides": tuple(sorted(out_sides)),
            "lengths": leg_lengths,
        })

    # 4. Outlet ratios (stubs)
    if random_ratios:
        ratios = [round(random.uniform(0.5, 2.0), 3) for _ in range(3)]
    else:
        ratios = [1.0, 1.0, 1.0]

    # 5. Write to files
    if layout_file:
        os.makedirs(os.path.dirname(os.path.abspath(layout_file)), exist_ok=True)
        with open(layout_file, "w", encoding="utf-8") as f:
            f.write("# Chip Layout Specification for Hydraulic Solver v2 Verification\n")
            f.write(f"GRID_SIZE {grid_size}\n")
            f.write(f"PITCH {DEFAULT_PITCH_UM:.1f}\n")
            f.write(f"CHANNEL_WIDTH {DEFAULT_CHANNEL_WIDTH_UM:.1f}\n")
            f.write(f"L_STUB {DEFAULT_L_STUB_UM:.1f}\n")
            f.write(f"INLET_FLOW {DEFAULT_INLET_FLOW:g}\n")
            f.write("INLET_CELL 0,3\n")
            f.write("OUTLET_CELLS 7,0 7,3 7,7\n")
            f.write("# Format: TILE <row> <col> <in_sides> <out_sides> <len_N> <len_S> <len_W> <len_E>\n")
            for t in tiles_data:
                lens = [f"{t['lengths'].get(s, 0.0):.1f}" for s in ("N", "S", "W", "E")]
                i_str = ";".join(t["inlet_sides"]) if t["inlet_sides"] else "none"
                o_str = ";".join(t["outlet_sides"]) if t["outlet_sides"] else "none"
                f.write(f"TILE {t['row']} {t['col']} {i_str} {o_str} {' '.join(lens)}\n")

    if ratio_file:
        os.makedirs(os.path.dirname(os.path.abspath(ratio_file)), exist_ok=True)
        with open(ratio_file, "w", encoding="utf-8") as f:
            f.write(f"{ratios[0]:.4f} {ratios[1]:.4f} {ratios[2]:.4f}\n")

    return {
        "tiles": tiles_data,
        "ratios": ratios,
        "n_tiles": len(tiles_data),
    }


generate_random_chip_v2 = generate_random_chip


if __name__ == "__main__":
    res = generate_random_chip("chip_layout.txt", "ratios.txt")
    print(f"Generated chip layout with {res['n_tiles']} tiles. Ratios: {res['ratios']}")

