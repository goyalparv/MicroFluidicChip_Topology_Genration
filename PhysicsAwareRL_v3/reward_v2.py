"""Evaluation of candidate tile networks and reward calculation for Stage 1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

from .config_v2 import DesignSpec, RewardWeights
from .hydraulic_solver_v2 import SolveResult, solve_network_v2
from .library_v2 import DIRECTION_DELTAS, Library, PlacedTile

DENOM_FLOOR_FRACTION = 0.01


@dataclass
class Evaluation:
    solve: SolveResult
    rel_err: np.ndarray         # Signed (Q - Q*) / denom per outlet
    e_q: float                  # Mean clipped absolute relative error
    complexity: float           # Tiles placed / max_tiles
    violations: float           # Unconnected outlets + isolated tiles
    cost: float                 # lam_q * e_q + lam_c * complexity + lam_v * violations
    success: bool               # All outlets connected and |rel_err| <= tolerance
    reward: float               # -cost (+ success bonus)
    isolated_tiles: int = 0
    isolated_cost: float = 0.0
    conn_progress: float = 0.0  # Fraction of positive-target outlets connected [0, 1]
    balance_score: float = 0.0  # Flow split similarity to target [0, 1]
    # Step 1 diagnostics and Step 6 structural terms
    n_tiles: int = 0
    extra: int = 0
    loops: int = 0
    merges: int = 0
    p_ratio: float = 0.0
    struct_cost: float = 0.0
    struct_parts: Dict[str, float] = None
    robust_e_q: float = 0.0

    def __post_init__(self):
        if self.struct_parts is None:
            self.struct_parts = {}

    @property
    def cost_core(self) -> float:
        """Cost without the isolated/dead-end tip penalty (deferred until episode end)."""
        return self.cost - self.isolated_cost

    @property
    def score(self) -> float:
        """Selection score: reward minus structural penalty."""
        return float(self.reward - self.struct_cost)

    def metrics(self) -> Dict[str, float]:
        return {
            "E_Q": float(self.e_q),
            "complexity": float(self.complexity),
            "violations": float(self.violations),
            "cost": float(self.cost),
            "reward": float(self.reward),
            "success": bool(self.success),
            "isolated_tiles": int(self.isolated_tiles),
            "conn_progress": float(self.conn_progress),
            "balance_score": float(self.balance_score),
            "n_tiles": int(self.n_tiles),
            "extra": int(self.extra),
            "loops": int(self.loops),
            "merges": int(self.merges),
            "p_ratio": float(self.p_ratio),
            "struct_cost": float(self.struct_cost),
            "extra_n": float(self.struct_parts.get("extra_n", 0.0)),
            "press_n": float(self.struct_parts.get("press_n", 0.0)),
            "loops_n": float(self.struct_parts.get("loops_n", 0.0)),
            "score": float(self.score),
            "robust_e_q": float(self.robust_e_q),
        }


class Evaluator:
    def __init__(self, spec: DesignSpec, library: Library, weights: Optional[RewardWeights] = None):
        spec.validate()
        if not spec.target_flows:
            raise ValueError("DesignSpec.target_flows must be set to evaluate a design")
        self.spec = spec
        self.library = library
        self.weights = weights or RewardWeights()
        self.target = np.asarray(spec.target_flows, dtype=float)
        self.pos_outlets = self.target > 0
        floor = DENOM_FLOOR_FRACTION * spec.inlet_flow
        self.denom = np.maximum(self.target, floor)
        self.outlet_set = {tuple(rc) for rc in spec.outlet_cells}
        # Reference baseline (Step 1.2, defaults until set_reference is called)
        self.n_min: int = 17
        self.p_ref: float = 1.0

    def set_reference(self, n_min: int, p_ref: float) -> None:
        """Store reference minimum tile count and baseline inlet pressure."""
        self.n_min = int(n_min)
        self.p_ref = float(p_ref)

    def is_newly_placed_dead(
        self, board: Mapping[Tuple[int, int], PlacedTile], cell: Tuple[int, int]
    ) -> bool:
        """True if the tile just placed at `cell` has no reachable downstream path."""
        if cell not in board:
            return False
        placed = board[cell]
        r, c = cell
        for o_side in placed.variant.outlet_sides:
            dr, dc = DIRECTION_DELTAS[o_side]
            nr, nc = r + dr, c + dc
            if (nr, nc) in self.outlet_set:
                return False  # Points directly to a chip outlet
            if 0 <= nr < self.spec.grid_size and 0 <= nc < self.spec.grid_size:
                if (nr, nc) not in board:
                    return False  # Points to an empty cell that can be extended
                neighbor = board[(nr, nc)]
                from .library_v2 import OPPOSITE
                if OPPOSITE[o_side] in neighbor.variant.inlet_sides:
                    return False  # Connects to valid inlet of existing downstream neighbor
        return True

    def count_isolated_tiles(self, board: Mapping[Tuple[int, int], PlacedTile]) -> int:
        """Count placed tiles that carry no useful flow to any downstream tile or chip outlet."""
        isolated = 0
        for (r, c), placed in board.items():
            if (r, c) in self.outlet_set:
                continue
            # Check if any outlet side connects to an inlet of an adjacent placed tile
            has_downstream = False
            for o_side in placed.variant.outlet_sides:
                dr, dc = DIRECTION_DELTAS[o_side]
                nr, nc = r + dr, c + dc
                if (nr, nc) in board:
                    neighbor = board[(nr, nc)]
                    from .library_v2 import OPPOSITE
                    if OPPOSITE[o_side] in neighbor.variant.inlet_sides:
                        has_downstream = True
                        break
            if not has_downstream:
                isolated += 1
        return isolated

    def compute_robust_error(self, board: Mapping[Tuple[int, int], PlacedTile]) -> float:
        """Step 8b: Monte-Carlo evaluation of flow split under channel fabrication perturbations."""
        import dataclasses
        from .metrics_v2 import design_id
        did = design_id(dict(board))
        seed = int(did[:8], 16)
        rng = np.random.default_rng(seed)
        w = self.weights
        eps = getattr(w, "robust_eps", 0.10)
        samples = getattr(w, "robust_samples", 8)
        errors = []
        for _ in range(samples):
            pert_board = {}
            for cell, pt in board.items():
                new_r_map = {}
                for s, r_val in pt.variant.R_per_side.items():
                    u = rng.uniform(-eps, eps)
                    new_r_map[s] = float(max(1e-4, r_val * (1.0 + u)))
                pert_v = dataclasses.replace(pt.variant, R_per_side=new_r_map)
                pert_board[cell] = PlacedTile(cell=cell, variant=pert_v)
            sub_sr = solve_network_v2(pert_board, self.spec)
            if sub_sr.ok and not sub_sr.stranded:
                sub_rel = (sub_sr.outlet_flows - self.target) / self.denom
            else:
                sub_rel = np.full(len(self.target), w.error_clip)
            errors.append(float(np.mean(np.minimum(np.abs(sub_rel), w.error_clip))))
        return float(np.mean(errors)) if errors else 0.0

    def evaluate(self, board: Mapping[Tuple[int, int], PlacedTile]) -> Evaluation:
        w = self.weights
        sr = solve_network_v2(board, self.spec)

        if sr.ok and not sr.stranded:
            rel = (sr.outlet_flows - self.target) / self.denom
        else:
            rel = np.full(len(self.target), w.error_clip)

        e_q = float(np.mean(np.minimum(np.abs(rel), w.error_clip)))
        complexity = len(board) / float(self.spec.max_tiles)

        unconnected = int(np.sum((self.target > 0) & ~sr.connected_outlets)) if sr.ok else len(self.target)
        isolated = self.count_isolated_tiles(board)
        isolated_cost = float(w.dead_end_penalty * isolated)

        violations = float(unconnected + isolated)
        cost = w.lam_q * e_q + w.lam_c * complexity + w.lam_v * unconnected + isolated_cost

        success = bool(sr.ok and unconnected == 0 and np.all(np.abs(rel) <= w.tolerance))
        reward = -cost + (w.success_bonus if success else 0.0)

        # Connectivity progress: fraction of positive-target outlets connected to inlet
        n_pos = int(np.sum(self.pos_outlets))
        if sr.ok and not sr.stranded and n_pos > 0:
            conn_progress = float(np.sum(self.pos_outlets & sr.connected_outlets)) / float(n_pos)
        else:
            conn_progress = 0.0

        # Balance score: total variation similarity between actual and target flow split
        q_sum = float(np.sum(sr.outlet_flows)) if sr.ok else 0.0
        target_sum = float(np.sum(self.target))
        if sr.ok and not sr.stranded and q_sum > 1e-15 and target_sum > 0:
            beta = sr.outlet_flows / q_sum
            alpha = self.target / target_sum
            balance_score = float(1.0 - 0.5 * np.sum(np.abs(alpha - beta)))
            balance_score = max(0.0, min(1.0, balance_score))
        else:
            balance_score = 0.0

        # Step 1 Diagnostics & Step 6 Structural Cost Calculations
        from .metrics_v2 import count_cycles, count_merges
        n_tiles = len(board)
        extra = max(0, n_tiles - self.n_min)
        loops = count_cycles(dict(board))
        merges = count_merges(dict(board))
        p_ratio = float(sr.p_in / self.p_ref) if (sr.ok and not sr.stranded and self.p_ref > 0) else 0.0

        max_tiles = self.spec.max_tiles
        if max_tiles <= self.n_min:
            extra_n = 0.0
        else:
            extra_n = float(np.clip((n_tiles - self.n_min) / float(max_tiles - self.n_min), 0.0, 1.0))

        if not sr.ok or sr.stranded:
            press_n = 1.0
        elif p_ratio <= 1.0:
            press_n = 0.0
        else:
            p_clip = getattr(w, "p_clip", 20.0)
            denom_p = np.log(p_clip) if p_clip > 1.0 else 1.0
            press_n = float(np.clip(np.log(max(p_ratio, 1e-12)) / denom_p, 0.0, 1.0))

        loops_cap = getattr(w, "loops_cap", 8)
        loops_n = float(min(max(loops, 0), loops_cap) / float(loops_cap)) if loops_cap > 0 else 0.0

        lam_extra = getattr(w, "lam_extra_tiles", 0.0)
        lam_press = getattr(w, "lam_pressure", 0.0)
        lam_loops = getattr(w, "lam_loops", 0.0)
        lam_robust = getattr(w, "lam_robust", 0.0)

        robust_e_q = 0.0
        robust_cost = 0.0
        if lam_robust > 0.0 and len(board) > 0 and sr.ok and not sr.stranded:
            robust_e_q = self.compute_robust_error(board)
            robust_n = float(min(robust_e_q, 0.5) / 0.5)
            robust_cost = lam_robust * robust_n

        struct_cost = float(lam_extra * extra_n + lam_press * press_n + lam_loops * loops_n + robust_cost)
        struct_parts = {
            "extra_n": extra_n,
            "press_n": press_n,
            "loops_n": loops_n,
            "robust_n": (robust_cost / lam_robust) if lam_robust > 0 else 0.0,
        }

        return Evaluation(
            solve=sr,
            rel_err=rel,
            e_q=e_q,
            complexity=complexity,
            violations=violations,
            cost=cost,
            success=success,
            reward=reward,
            isolated_tiles=isolated,
            isolated_cost=isolated_cost,
            conn_progress=conn_progress,
            balance_score=balance_score,
            n_tiles=n_tiles,
            extra=extra,
            loops=loops,
            merges=merges,
            p_ratio=p_ratio,
            struct_cost=struct_cost,
            struct_parts=struct_parts,
            robust_e_q=robust_e_q,
        )

