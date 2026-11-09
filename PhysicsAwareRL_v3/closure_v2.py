"""Closed-topology enforcement and closure oracle for microfluidic RL (Stage 1)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from .config_v2 import DesignSpec
from .hydraulic_solver_v2 import port_node
from .library_v2 import ALL_SIDES, DIRECTION_DELTAS, Library, OPPOSITE, PlacedTile, TileVariant

Cell = Tuple[int, int]
Sides = Tuple[Tuple[str, ...], Tuple[str, ...]]  # (inlet_sides, outlet_sides), each sorted
TilesMap = Dict[Cell, Sides]


def sides_map(board: Mapping[Cell, PlacedTile]) -> TilesMap:
    """Extract normalized sides map from placed board."""
    return {
        cell: (tuple(sorted(placed.variant.inlet_sides)), tuple(sorted(placed.variant.outlet_sides)))
        for cell, placed in board.items()
    }


def board_signature(board: Mapping[Cell, PlacedTile]) -> frozenset:
    """Board signature used for environment-level mask cache."""
    return frozenset((cell, placed.variant.variant_id) for cell, placed in board.items())


def tiles_signature(tiles: TilesMap) -> frozenset:
    """Tiles signature used for oracle cache."""
    return frozenset((cell, I, O) for cell, (I, O) in tiles.items())


def allowed_combos(library: Library) -> Set[Sides]:
    """Normalized (inlet_sides, outlet_sides) keys present in the tile library."""
    return {
        (tuple(sorted(v.inlet_sides)), tuple(sorted(v.outlet_sides)))
        for v in library.variants
    }


@dataclass
class PortReport:
    demand: Dict[Cell, Tuple[str, ...]]   # empty cell -> sorted incoming sides (incl. virtual inlet)
    pending: List[Tuple[Cell, str]]       # (placed cell, outlet side) facing an empty cell
    open_inlets: List[Tuple[Cell, str]]
    blocked_outlets: List[Tuple[Cell, str]]
    missing_sinks: List[Cell]             # chip outlet cells not yet placed

    @property
    def closed(self) -> bool:
        return not (self.pending or self.open_inlets or self.blocked_outlets or self.missing_sinks)


def analyze_ports(tiles: TilesMap, spec: DesignSpec) -> PortReport:
    """Analyze all active sides in tiles against boundaries and neighboring cells."""
    open_inlets: List[Tuple[Cell, str]] = []
    pending: List[Tuple[Cell, str]] = []
    blocked_outlets: List[Tuple[Cell, str]] = []
    demand_dict: Dict[Cell, Set[str]] = {}
    outlet_set = {tuple(rc) for rc in spec.outlet_cells}

    # 1. Inlet checks
    for c, (I, O) in tiles.items():
        for s in I:
            if c == spec.inlet_cell and s == "N":
                continue
            dr, dc = DIRECTION_DELTAS[s]
            nb = (c[0] + dr, c[1] + dc)
            if nb in tiles and OPPOSITE[s] in tiles[nb][1]:
                continue
            open_inlets.append((c, s))

    # 2. Outlet checks
    for c, (I, O) in tiles.items():
        for s in O:
            if c in outlet_set and s == "S":
                continue
            dr, dc = DIRECTION_DELTAS[s]
            nb = (c[0] + dr, c[1] + dc)
            if not (0 <= nb[0] < spec.grid_size and 0 <= nb[1] < spec.grid_size):
                blocked_outlets.append((c, s))
            elif nb in tiles:
                if OPPOSITE[s] in tiles[nb][0]:
                    continue
                blocked_outlets.append((c, s))
            else:
                pending.append((c, s))
                demand_dict.setdefault(nb, set()).add(OPPOSITE[s])

    # 3. Virtual demand from external inlet if unplaced
    if spec.inlet_cell not in tiles:
        demand_dict.setdefault(spec.inlet_cell, set()).add("N")

    # 4. Missing sinks
    missing_sinks = [tuple(t) for t in spec.outlet_cells if tuple(t) not in tiles]

    # Convert demand sets to sorted tuples
    demand = {cell: tuple(sorted(sides)) for cell, sides in demand_dict.items()}

    return PortReport(
        demand=demand,
        pending=pending,
        open_inlets=open_inlets,
        blocked_outlets=blocked_outlets,
        missing_sinks=missing_sinks,
    )


def find_open_ports(
    board: Mapping[Cell, PlacedTile], spec: DesignSpec
) -> List[Tuple[Cell, str, str]]:
    """Returns all defects plus pending ports, with kind in {'open_inlet', 'pending', 'blocked'}."""
    tiles = sides_map(board)
    rep = analyze_ports(tiles, spec)
    out: List[Tuple[Cell, str, str]] = []
    for c, s in rep.open_inlets:
        out.append((c, s, "open_inlet"))
    for c, s in rep.pending:
        out.append((c, s, "pending"))
    for c, s in rep.blocked_outlets:
        out.append((c, s, "blocked"))
    return out


def check_port_degrees(board: Mapping[Cell, PlacedTile], spec: DesignSpec) -> List[int]:
    """Returns fine-grid port node IDs whose resistor-endpoint degree is not 2 (invariant I7)."""
    degree: Dict[int, int] = {}

    for (r, c), placed in board.items():
        v = placed.variant
        for s in v.active_sides:
            p = port_node(r, c, s)
            degree[p] = degree.get(p, 0) + 1

    # External virtual inlet stub
    if spec.inlet_cell in board:
        p_in = port_node(spec.inlet_cell[0], spec.inlet_cell[1], "N")
        degree[p_in] = degree.get(p_in, 0) + 1

    # External ground outlet stubs
    for r, c in spec.outlet_cells:
        if (r, c) in board:
            p_out = port_node(r, c, "S")
            degree[p_out] = degree.get(p_out, 0) + 1

    return sorted([p for p, deg in degree.items() if deg != 2])


def cell_context(
    tiles: TilesMap,
    spec: DesignSpec,
    cell: Cell,
    allowed: Set[Sides],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], List[Tuple[str, ...]]]:
    """Single source of truth for local placement legality.
    
    Returns (I, free_sides, options).
    """
    r, c = cell
    outlet_set = {tuple(rc) for rc in spec.outlet_cells}
    I_set: Set[str] = set()
    free_set: Set[str] = set()

    for s in ("N", "S", "W", "E"):
        dr, dc = DIRECTION_DELTAS[s]
        nr, nc = r + dr, c + dc
        in_bounds = (0 <= nr < spec.grid_size and 0 <= nc < spec.grid_size)

        if not in_bounds:
            if s == "N" and cell == spec.inlet_cell:
                I_set.add("N")
            elif s == "S" and cell in outlet_set:
                pass
            else:
                pass
        elif (nr, nc) in tiles:
            nb_O = tiles[(nr, nc)][1]
            if OPPOSITE[s] in nb_O:
                I_set.add(s)
            else:
                pass
        else:
            free_set.add(s)

    I_tuple = tuple(sorted(I_set))
    free_tuple = tuple(sorted(free_set))

    if not I_tuple:
        return I_tuple, free_tuple, []

    if cell in outlet_set:
        opt = ("S",)
        if (I_tuple, opt) in allowed:
            return I_tuple, free_tuple, [opt]
        return I_tuple, free_tuple, []

    options: List[Tuple[str, ...]] = []
    for k in range(1, len(free_tuple) + 1):
        for combo in combinations(free_tuple, k):
            o_tuple = tuple(sorted(combo))
            if (I_tuple, o_tuple) in allowed:
                options.append(o_tuple)

    return I_tuple, free_tuple, options


def ordered_options(
    tiles: TilesMap,
    spec: DesignSpec,
    X: Cell,
    allowed: Set[Sides],
    rep: PortReport,
) -> List[Tuple[str, ...]]:
    """Order candidate outlet options using heuristics for DFS search speed."""
    _, _, raw_options = cell_context(tiles, spec, X, allowed)
    if not raw_options:
        return []

    sinks_set = {tuple(rc) for rc in spec.outlet_cells}
    unclaimed = [s for s in rep.missing_sinks if s not in rep.demand]
    A = [d for d in rep.demand if d not in sinks_set]
    need_split = len(unclaimed) > len(A)
    targets = unclaimed if unclaimed else rep.missing_sinks

    def sort_key(O: Tuple[str, ...]):
        # Primary key: |O| descending if need_split, else ascending
        prim = -len(O) if need_split else len(O)
        # Secondary key: sum of Manhattan distance from neighbour cell to nearest target
        sec = 0
        for o in O:
            if o == "S" and X in sinks_set:
                sec += 0
            else:
                dr, dc = DIRECTION_DELTAS[o]
                nbr = (X[0] + dr, X[1] + dc)
                if targets:
                    sec += min(abs(nbr[0] - t[0]) + abs(nbr[1] - t[1]) for t in targets)
        return (prim, sec, O)

    return sorted(raw_options, key=sort_key)


class _Cap(Exception):
    pass


_MIN_TILES_GLOBAL_CACHE: Dict[Tuple, Optional[int]] = {}


class ClosureOracle:
    """Bounded DFS oracle to test if a candidate placement can be completed to a closed topology."""

    def __init__(self, spec: DesignSpec, allowed: Set[Sides], node_limit: int = 4000):
        self.spec = spec
        self.allowed = allowed
        self.node_limit = node_limit
        self._cache: Dict[frozenset, Optional[List[Tuple[Cell, Tuple[str, ...], Tuple[str, ...]]]]] = {}
        self.calls = 0
        self.cache_hits = 0
        self.nodes = 0
        self.cap_hits = 0

    def plan(
        self, tiles: TilesMap, budget: Optional[int] = None
    ) -> Optional[List[Tuple[Cell, Tuple[str, ...], Tuple[str, ...]]]]:
        """Return a valid completion plan closing the topology within budget (default max_tiles - len(tiles)), or None."""
        cur_budget = (self.spec.max_tiles - len(tiles)) if budget is None else budget
        sig = tiles_signature(tiles) if budget is None else (tiles_signature(tiles), budget)
        if sig in self._cache:
            self.cache_hits += 1
            return self._cache[sig]

        self.calls += 1
        nodes_in_call = 0
        grid_n = self.spec.grid_size

        def search(cur_tiles: TilesMap, cur_budget: int):
            nonlocal nodes_in_call
            nodes_in_call += 1
            self.nodes += 1
            if nodes_in_call > self.node_limit:
                raise _Cap()

            rep = analyze_ports(cur_tiles, self.spec)
            if rep.open_inlets or rep.blocked_outlets:
                return None
            if rep.closed:
                return []
            if cur_budget <= 0:
                return None

            # N1: demand empty but sinks missing
            if not rep.demand and rep.missing_sinks:
                return None
            # N2: all sinks placed but demand non-empty
            if not rep.missing_sinks and rep.demand:
                return None
            # N3: |set(demand) | set(missing_sinks)| > budget
            if len(set(rep.demand) | set(rep.missing_sinks)) > cur_budget:
                return None

            # N3b: Manhattan distance & bounding-box lower bound to reach all missing sinks
            if rep.demand and rep.missing_sinks:
                if len(rep.demand) >= len(rep.missing_sinks):
                    min_steps = sum(min(abs(d[0] - s[0]) + abs(d[1] - s[1]) for d in rep.demand) for s in rep.missing_sinks)
                    if min_steps + len(rep.missing_sinks) > cur_budget:
                        return None
                else:
                    all_pts = list(rep.demand.keys()) + rep.missing_sinks
                    r_span = max(p[0] for p in all_pts) - min(p[0] for p in all_pts)
                    c_span = max(p[1] for p in all_pts) - min(p[1] for p in all_pts)
                    if r_span + c_span + len(rep.missing_sinks) - 1 > cur_budget:
                        return None

            # N4: compute 4-connected components of EMPTY cells
            empty_cells = set(
                (r, c)
                for r in range(grid_n)
                for c in range(grid_n)
                if (r, c) not in cur_tiles
            )
            comp_id: Dict[Cell, int] = {}
            cid = 0
            for cell in empty_cells:
                if cell in comp_id:
                    continue
                cid += 1
                q = [cell]
                comp_id[cell] = cid
                while q:
                    curr = q.pop()
                    cr, cc = curr
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nbr = (cr + dr, cc + dc)
                        if nbr in empty_cells and nbr not in comp_id:
                            comp_id[nbr] = cid
                            q.append(nbr)

            sink_cids = {comp_id[s] for s in rep.missing_sinks if s in comp_id}
            demand_cids = {comp_id[d] for d in rep.demand if d in comp_id}

            for d in rep.demand:
                if d not in comp_id or comp_id[d] not in sink_cids:
                    return None
            for s in rep.missing_sinks:
                if s not in comp_id or comp_id[s] not in demand_cids:
                    return None

            # X = min(demand, key=(row, col))
            X = min(rep.demand.keys(), key=lambda c: (c[0], c[1]))
            I = rep.demand[X]

            for O in ordered_options(cur_tiles, self.spec, X, self.allowed, rep):
                next_tiles = dict(cur_tiles)
                next_tiles[X] = (I, O)
                sub_res = search(next_tiles, cur_budget - 1)
                if sub_res is not None:
                    return [(X, I, O)] + sub_res

            return None

        try:
            res = search(tiles, cur_budget)
        except _Cap:
            self.cap_hits += 1
            res = None

        self._cache[sig] = res
        return res

    def min_tiles_required(self, max_search: int = 30) -> Optional[int]:
        """Iterative deepening from empty board to find minimum tiles needed to close."""
        cache_k = (
            tuple(self.spec.inlet_cell),
            tuple(sorted(self.spec.outlet_cells)),
            self.spec.grid_size,
            frozenset(self.allowed),
        )
        if cache_k in _MIN_TILES_GLOBAL_CACHE:
            return _MIN_TILES_GLOBAL_CACHE[cache_k]

        start_k = 1 + len(self.spec.outlet_cells)
        for k in range(start_k, max_search + 1):
            nodes_in_call = 0

            def search_min(cur_tiles: TilesMap, cur_budget: int):
                nonlocal nodes_in_call
                nodes_in_call += 1
                if nodes_in_call > self.node_limit:
                    raise _Cap()

                rep = analyze_ports(cur_tiles, self.spec)
                if rep.open_inlets or rep.blocked_outlets:
                    return None
                if rep.closed:
                    return []
                if cur_budget <= 0:
                    return None
                if not rep.demand and rep.missing_sinks:
                    return None
                if not rep.missing_sinks and rep.demand:
                    return None
                # N3: |set(demand) | set(missing_sinks)| > budget
                if len(set(rep.demand) | set(rep.missing_sinks)) > cur_budget:
                    return None

                # N3b: Manhattan distance & bounding-box lower bound to reach all missing sinks
                if rep.demand and rep.missing_sinks:
                    if len(rep.demand) >= len(rep.missing_sinks):
                        min_steps = sum(min(abs(d[0] - s[0]) + abs(d[1] - s[1]) for d in rep.demand) for s in rep.missing_sinks)
                        if min_steps + len(rep.missing_sinks) > cur_budget:
                            return None
                    else:
                        all_pts = list(rep.demand.keys()) + rep.missing_sinks
                        r_span = max(p[0] for p in all_pts) - min(p[0] for p in all_pts)
                        c_span = max(p[1] for p in all_pts) - min(p[1] for p in all_pts)
                        if r_span + c_span + len(rep.missing_sinks) - 1 > cur_budget:
                            return None

                X = min(rep.demand.keys(), key=lambda c: (c[0], c[1]))
                I = rep.demand[X]

                for O in ordered_options(cur_tiles, self.spec, X, self.allowed, rep):
                    next_tiles = dict(cur_tiles)
                    next_tiles[X] = (I, O)
                    sub = search_min(next_tiles, cur_budget - 1)
                    if sub is not None:
                        return [(X, I, O)] + sub
                return None

            try:
                res = search_min({}, k)
                if res is not None:
                    _MIN_TILES_GLOBAL_CACHE[cache_k] = k
                    return k
            except _Cap:
                continue

        _MIN_TILES_GLOBAL_CACHE[cache_k] = None
        return None
