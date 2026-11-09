"""17x17 Fine-Grid Hydraulic Circuit Solver for Stage 1.

Each of the 8x8 coarse grid cells (r, c) maps to:
  - Center node: (2r+1, 2c+1)
  - North port:  (2r,   2c+1)
  - South port:  (2r+2, 2c+1)
  - West port:   (2r+1, 2c)
  - East port:   (2r+1, 2c+2)

Shared ports between neighbors share the same fine-grid node.
Total fine grid nodes = 17 * 17 = 289.
External virtual inlet node = 289 (Total nodes = 290).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from .config_v2 import DesignSpec
from .library_v2 import ALL_SIDES, PlacedTile

FINE_GRID_N = 17
TOTAL_FINE_NODES = 289
INLET_VIRTUAL_NODE = 289
TOTAL_NODES = 290


def fine_node_id(row: int, col: int) -> int:
    return int(row) * FINE_GRID_N + int(col)


def fine_node_rc(node_id: int) -> Tuple[int, int]:
    return divmod(int(node_id), FINE_GRID_N)


def center_node(r: int, c: int) -> int:
    return fine_node_id(2 * r + 1, 2 * c + 1)


def port_node(r: int, c: int, side: str) -> int:
    if side == "N":
        return fine_node_id(2 * r, 2 * c + 1)
    elif side == "S":
        return fine_node_id(2 * r + 2, 2 * c + 1)
    elif side == "W":
        return fine_node_id(2 * r + 1, 2 * c)
    elif side == "E":
        return fine_node_id(2 * r + 1, 2 * c + 2)
    else:
        raise ValueError(f"Unknown side {side}, expected one of {ALL_SIDES}")


def outlet_port_nodes(spec: DesignSpec) -> List[int]:
    """Fine-grid node IDs for each outlet cell's South port."""
    out = []
    for r, c in spec.outlet_cells:
        out.append(port_node(r, c, "S"))
    return out


@dataclass
class SolveResult:
    ok: bool
    pressures: np.ndarray          # shape (290,) fine-grid nodal pressures
    outlet_flows: np.ndarray       # flow leaving each outlet port
    connected_outlets: np.ndarray  # bool per outlet indicating connectivity to inlet
    mass_residual: float           # max KCL residual on internal nodes
    p_in: float                    # pressure at external inlet node
    stranded: bool                 # true if inlet has no path to any outlet


def solve_network_v2(
    board: Mapping[Tuple[int, int], PlacedTile],
    spec: DesignSpec,
) -> SolveResult:
    """Solve Stokes hydraulic circuit on the 17x17 fine grid."""
    spec.validate()
    n_out = len(spec.outlet_cells)
    out_ports = outlet_port_nodes(spec)
    inlet_port = port_node(spec.inlet_cell[0], spec.inlet_cell[1], "N")

    # Build adjacency list: node -> list of (neighbor_node, conductance)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(TOTAL_NODES)]

    def add_resistor(u: int, v: int, r: float):
        if r <= 0 or not np.isfinite(r):
            raise ValueError(f"Invalid resistance {r} between {u} and {v}")
        g = 1.0 / float(r)
        adj[u].append((v, g))
        adj[v].append((u, g))

    # 1. Add edges for each placed tile
    for (r, c), placed in board.items():
        c_node = center_node(r, c)
        variant = placed.variant
        for s in variant.active_sides:
            p_node = port_node(r, c, s)
            r_val = variant.R_per_side[s]
            add_resistor(c_node, p_node, r_val)

    # 2. Add external inlet stub: INLET_VIRTUAL_NODE -> inlet_port
    r_stub = max(spec.R_stub, 1e-12)
    add_resistor(INLET_VIRTUAL_NODE, inlet_port, r_stub)

    # 3. Find connected component from inlet
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
    stranded = not any(connected_outlets)

    pressures = np.zeros(TOTAL_NODES, dtype=float)
    outlet_flows = np.zeros(n_out, dtype=float)

    if stranded:
        # Inlet cannot reach any outlet -> zero flow everywhere
        # Nominal placeholder pressure on the inlet side
        for node_idx in range(TOTAL_NODES):
            if visited[node_idx]:
                pressures[node_idx] = spec.inlet_flow * r_stub
        return SolveResult(
            ok=True,
            pressures=pressures,
            outlet_flows=outlet_flows,
            connected_outlets=connected_outlets,
            mass_residual=0.0,
            p_in=float(pressures[INLET_VIRTUAL_NODE]),
            stranded=True,
        )

    # 4. Assemble linear system for the connected nodes
    # For outlet ports, fluid discharges to ground (P=0) through an outlet stub of resistance r_stub.
    # KCL at outlet port node p: sum_nbrs g*(P_p - P_nbr) + P_p / r_stub = 0
    # Thus outlet ports are free nodes with an extra self-conductance 1/r_stub to ground.
    g_stub_out = 1.0 / r_stub
    free_nodes = [i for i in range(TOTAL_NODES) if visited[i]]
    node_to_idx = {node: idx for idx, node in enumerate(free_nodes)}
    n_free = len(free_nodes)

    A = np.zeros((n_free, n_free), dtype=float)
    b = np.zeros(n_free, dtype=float)

    for node in free_nodes:
        i = node_to_idx[node]
        for nbr, g in adj[node]:
            A[i, i] += g
            if nbr in node_to_idx:
                A[i, node_to_idx[nbr]] -= g
        # If this node is one of the connected outlet ports, add discharge conductance to ground (P=0)
        if node in out_ports:
            A[i, i] += g_stub_out

    # Current injected at the external inlet
    b[node_to_idx[INLET_VIRTUAL_NODE]] = spec.inlet_flow

    # 5. Solve for nodal pressures
    try:
        sol = np.linalg.solve(A, b)
        for node in free_nodes:
            pressures[node] = sol[node_to_idx[node]]
        ok = True
    except np.linalg.LinAlgError:
        ok = False

    # 6. Compute outlet flows leaving into ground
    for j, p in enumerate(out_ports):
        if connected_outlets[j]:
            outlet_flows[j] = pressures[p] * g_stub_out
        else:
            outlet_flows[j] = 0.0

    # 7. Mass conservation check on internal nodes (excluding inlet & ground discharge)
    mass_residual = 0.0
    for node in free_nodes:
        net_out = 0.0
        for nbr, g in adj[node]:
            net_out += g * (pressures[node] - pressures[nbr])
        if node in out_ports:
            net_out += pressures[node] * g_stub_out
        injected = spec.inlet_flow if node == INLET_VIRTUAL_NODE else 0.0
        mass_residual = max(mass_residual, abs(net_out - injected))

    return SolveResult(
        ok=ok,
        pressures=pressures,
        outlet_flows=outlet_flows,
        connected_outlets=connected_outlets,
        mass_residual=float(mass_residual),
        p_in=float(pressures[INLET_VIRTUAL_NODE]),
        stranded=False,
    )

