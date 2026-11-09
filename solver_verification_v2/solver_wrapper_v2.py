"""Python solver wrapper for Stage 1 solver verification.

Loads `chip_layout.txt` and `ratios.txt`, builds the 17x17 circuit,
and runs `PhysicsAwareRL_v2.hydraulic_solver_v2.solve_network_v2`.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

try:
    from .hydraulic_solver_v2 import (
        INLET_VIRTUAL_NODE,
        TOTAL_NODES,
        DesignSpec,
        PlacedTile,
        TileVariant,
        center_node,
        outlet_port_nodes,
        port_node,
        solve_network_v2,
    )
    from .random_chip_v2 import R_PER_UM
except ImportError:
    from hydraulic_solver_v2 import (
        INLET_VIRTUAL_NODE,
        TOTAL_NODES,
        DesignSpec,
        PlacedTile,
        TileVariant,
        center_node,
        outlet_port_nodes,
        port_node,
        solve_network_v2,
    )
    from random_chip_v2 import R_PER_UM


def load_chip_layout(layout_path: str) -> Tuple[Dict, List[dict]]:
    """Parse chip_layout.txt file into config parameters and tile specifications."""
    config = {
        "grid_size": 8,
        "pitch": 1200.0,
        "channel_width": 100.0,
        "l_stub": 600.0,
        "inlet_flow": 1e-4,
        "inlet_cell": (0, 3),
        "outlet_cells": [(7, 0), (7, 3), (7, 7)],
    }
    tiles = []

    with open(layout_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cmd = parts[0].upper()
            if cmd == "GRID_SIZE":
                config["grid_size"] = int(parts[1])
            elif cmd == "PITCH":
                config["pitch"] = float(parts[1])
            elif cmd == "CHANNEL_WIDTH":
                config["channel_width"] = float(parts[1])
            elif cmd == "L_STUB":
                config["l_stub"] = float(parts[1])
            elif cmd == "INLET_FLOW":
                config["inlet_flow"] = float(parts[1])
            elif cmd == "INLET_CELL":
                r, c = parts[1].split(",")
                config["inlet_cell"] = (int(r), int(c))
            elif cmd == "OUTLET_CELLS":
                outs = []
                for p in parts[1:]:
                    r, c = p.split(",")
                    outs.append((int(r), int(c)))
                config["outlet_cells"] = outs
            elif cmd == "TILE":
                # TILE <row> <col> <in_sides> <out_sides> <len_N> <len_S> <len_W> <len_E>
                r, c = int(parts[1]), int(parts[2])
                i_str = parts[3]
                o_str = parts[4]
                in_sides = tuple(s for s in i_str.split(";") if s and s != "none")
                out_sides = tuple(s for s in o_str.split(";") if s and s != "none")
                lens = {
                    "N": float(parts[5]),
                    "S": float(parts[6]),
                    "W": float(parts[7]),
                    "E": float(parts[8]),
                }
                tiles.append({
                    "row": r,
                    "col": c,
                    "inlet_sides": in_sides,
                    "outlet_sides": out_sides,
                    "lengths": lens,
                })

    return config, tiles


def load_ratios(ratio_path: str) -> List[float]:
    """Read outlet stub ratios from ratios.txt (default [1.0, 1.0, 1.0])."""
    if not os.path.exists(ratio_path):
        return [1.0, 1.0, 1.0]
    with open(ratio_path, "r", encoding="utf-8") as f:
        line = f.read().strip()
        if not line:
            return [1.0, 1.0, 1.0]
        return [float(x) for x in line.split()[:3]]


def simulate_chip_v2(
    layout_path: str = "chip_layout.txt",
    ratio_path: str = "ratios.txt",
) -> Tuple[np.ndarray | None, List[float] | None]:
    """Run hydraulic_solver_v2 on the chip specified in layout_path.

    Returns:
        pressures: nodal pressure array (or None if failed)
        outlet_flows: list of 3 outlet flows [Q1, Q2, Q3] in m/s (same scale as velocity)
    """
    config, tiles = load_chip_layout(layout_path)
    ratios = load_ratios(ratio_path)

    # Base stub resistance matching 600 um channel length
    r_stub_base = R_PER_UM * config["l_stub"]

    # Build TileVariant and PlacedTile for each tile
    board: Dict[Tuple[int, int], PlacedTile] = {}
    for t in tiles:
        r, c = t["row"], t["col"]
        i_sides = t["inlet_sides"]
        o_sides = t["outlet_sides"]
        active = set(i_sides) | set(o_sides)

        # Resistance proportional to length matching library values
        r_map = {}
        for s in active:
            length_um = t["lengths"].get(s, config["pitch"] / 2.0)
            r_map[s] = float(R_PER_UM * max(length_um, 10.0))

        vid = f"tile_r{r}c{c}"
        variant = TileVariant(
            variant_id=vid,
            inlet_sides=i_sides,
            outlet_sides=o_sides,
            R_per_side=r_map,
        )
        board[(r, c)] = PlacedTile(cell=(r, c), variant=variant)

    spec = DesignSpec(
        grid_size=config["grid_size"],
        inlet_cell=config["inlet_cell"],
        outlet_cells=config["outlet_cells"],
        inlet_flow=config["inlet_flow"],
        R_stub=r_stub_base,
    )

    # If outlet ratios are all equal (1.0), use solve_network_v2 directly
    if all(abs(r - 1.0) < 1e-4 for r in ratios):
        res = solve_network_v2(board, spec)
        if not res.ok or res.stranded:
            return None, None
        return res.pressures, [float(q) for q in res.outlet_flows]

    # If outlet ratios differ, solve with individual outlet stub resistances
    # R_stub_j = R_stub_base * ratio_j
    spec.validate()
    n_out = len(spec.outlet_cells)
    out_ports = outlet_port_nodes(spec)
    inlet_port = port_node(spec.inlet_cell[0], spec.inlet_cell[1], "N")

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(TOTAL_NODES)]
    for (r, c), placed in board.items():
        c_node = center_node(r, c)
        variant = placed.variant
        for s in variant.active_sides:
            p_node = port_node(r, c, s)
            g = 1.0 / variant.R_per_side[s]
            adj[c_node].append((p_node, g))
            adj[p_node].append((c_node, g))

    g_in_stub = 1.0 / r_stub_base
    adj[INLET_VIRTUAL_NODE].append((inlet_port, g_in_stub))
    adj[inlet_port].append((INLET_VIRTUAL_NODE, g_in_stub))

    # Connected component
    visited = np.zeros(TOTAL_NODES, dtype=bool)
    visited[INLET_VIRTUAL_NODE] = True
    queue = [INLET_VIRTUAL_NODE]
    while queue:
        curr = queue.pop(0)
        for nxt, _ in adj[curr]:
            if not visited[nxt]:
                visited[nxt] = True
                queue.append(nxt)

    connected_outlets = np.array([bool(visited[p]) for p in out_ports], dtype=bool)
    if not any(connected_outlets):
        return None, None

    free_nodes = [i for i in range(TOTAL_NODES) if visited[i]]
    node_to_idx = {node: idx for idx, node in enumerate(free_nodes)}
    n_free = len(free_nodes)

    A = np.zeros((n_free, n_free), dtype=float)
    b = np.zeros(n_free, dtype=float)

    # Outlet discharge conductances
    g_stub_outs = [1.0 / (r_stub_base * max(ratios[j], 1e-4)) for j in range(n_out)]

    for node in free_nodes:
        i = node_to_idx[node]
        for nbr, g in adj[node]:
            A[i, i] += g
            if nbr in node_to_idx:
                A[i, node_to_idx[nbr]] -= g
        if node in out_ports:
            j = out_ports.index(node)
            A[i, i] += g_stub_outs[j]

    b[node_to_idx[INLET_VIRTUAL_NODE]] = spec.inlet_flow

    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None

    pressures = np.zeros(TOTAL_NODES, dtype=float)
    for node in free_nodes:
        pressures[node] = sol[node_to_idx[node]]

    outlet_flows = [
        float(pressures[p] * g_stub_outs[j]) if connected_outlets[j] else 0.0
        for j, p in enumerate(out_ports)
    ]

    return pressures, outlet_flows


if __name__ == "__main__":
    import sys
    l_path = sys.argv[1] if len(sys.argv) > 1 else "chip_layout.txt"
    r_path = sys.argv[2] if len(sys.argv) > 2 else "ratios.txt"
    p, q = simulate_chip_v2(l_path, r_path)
    if q:
        print(f"Python Solver Outlet Flows [O1, O2, O3]: {q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}")
        print(f"Total Outlet Flow: {sum(q):.6f}")
    else:
        print("Solver failed.")

