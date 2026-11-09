"""Topology, graph metrics, and design signatures for PhysicsAwareRL_v3.

Pure functions with no dependency on the environment.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Sequence, Set, Tuple

from .library_v2 import DIRECTION_DELTAS, OPPOSITE, PlacedTile


def topo_signature(board: Dict[Tuple[int, int], PlacedTile]) -> Tuple:
    """Canonical invariant signature of tile placements: (cell, sorted inlet sides, sorted outlet sides)."""
    return tuple(
        sorted(
            (
                cell,
                tuple(sorted(t.variant.inlet_sides)),
                tuple(sorted(t.variant.outlet_sides)),
            )
            for cell, t in board.items()
        )
    )


def topo_id(board: Dict[Tuple[int, int], PlacedTile]) -> str:
    """12-character SHA-1 hash of the topology signature (ignores variant resistances)."""
    return hashlib.sha1(repr(topo_signature(board)).encode("utf-8")).hexdigest()[:12]


def design_id(board: Dict[Tuple[int, int], PlacedTile]) -> str:
    """12-character SHA-1 hash of the concrete design: (cell, variant_id)."""
    sig = tuple(sorted((cell, t.variant.variant_id) for cell, t in board.items()))
    return hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()[:12]


def count_merges(board: Dict[Tuple[int, int], PlacedTile]) -> int:
    """Count number of tiles with two or more inlet sides."""
    return sum(1 for t in board.values() if len(t.variant.inlet_sides) >= 2)


def count_cycles(board: Dict[Tuple[int, int], PlacedTile]) -> int:
    """Cycle rank of the tile graph: E - V + C.

    V: number of tiles
    E: matched directed adjacent connections
    C: number of connected components
    """
    if not board:
        return 0

    parent = {c: c for c in board}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = 0
    for (r, c), t in board.items():
        for s in t.variant.outlet_sides:
            dr, dc = DIRECTION_DELTAS[s]
            nb = (r + dr, c + dc)
            if nb in board and OPPOSITE[s] in board[nb].variant.inlet_sides:
                edges += 1
                parent[find((r, c))] = find(nb)

    comps = len({find(c) for c in board})
    return edges - len(board) + comps


def jaccard(a: Sequence[Any] | Set[Any], b: Sequence[Any] | Set[Any]) -> float:
    """Jaccard similarity between two sets of occupied cells."""
    s_a, s_b = set(a), set(b)
    if not s_a and not s_b:
        return 1.0
    return len(s_a & s_b) / max(len(s_a | s_b), 1)
