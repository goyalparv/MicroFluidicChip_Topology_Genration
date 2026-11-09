"""Sequential tile-placement environment for Stage 1 microfluidic RL topology synthesis."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import logging

import numpy as np

from .closure import (
    ClosureOracle,
    Sides,
    TilesMap,
    allowed_combos,
    analyze_ports,
    board_signature,
    cell_context,
    find_open_ports,
    sides_map,
    tiles_signature,
)
from .config import DesignSpec, RewardWeights
from .hydraulic_solver import TOTAL_NODES
from .library import ALL_SIDES, DIRECTION_DELTAS, Library, OPPOSITE, PlacedTile, TileVariant
from .reward import Evaluation, Evaluator

CLIP_VAL = 3.0


FEATURE_DIM = 39


class TopologyEnv:
    def __init__(
        self,
        spec: DesignSpec,
        library: Library,
        weights: Optional[RewardWeights] = None,
        dense_reward: bool = True,
        enforce_closure: bool = True,
        closure_node_limit: int = 4000,
        strict_closure: bool = False,
        allow_large_struct: bool = False,
        tile_potential: str = "terminal",
    ):
        spec.validate()
        if not spec.target_flows:
            raise ValueError("spec.target_flows must be set")
        self.spec = spec
        self.library = library
        self.weights = weights or RewardWeights()
        self.allow_large_struct = allow_large_struct
        self.tile_potential = tile_potential
        self.weights.check_structural_budget(allow_large_struct=allow_large_struct)

        self.dense_reward = dense_reward
        self.enforce_closure = enforce_closure
        self.closure_node_limit = closure_node_limit
        self.strict_closure = strict_closure
        self.evaluator = Evaluator(spec, library, self.weights)

        self.N = spec.grid_size
        self.total_cells = self.N * self.N
        self.V = len(library)
        self.n_actions = self.total_cells * self.V + 1
        self.stop_action = self.total_cells * self.V

        # Build combo to variant indices mapping
        combo_map: Dict[Sides, List[int]] = {}
        for vi, v in enumerate(self.library.variants):
            key = (tuple(sorted(v.inlet_sides)), tuple(sorted(v.outlet_sides)))
            combo_map.setdefault(key, []).append(vi)
        self.combo_to_vidx: Dict[Sides, np.ndarray] = {
            k: np.array(v, dtype=int) for k, v in combo_map.items()
        }

        # Build closure oracle
        self.oracle = ClosureOracle(spec, allowed_combos(library), closure_node_limit)
        if self.enforce_closure:
            plan0 = self.oracle.plan({})
            if plan0 is None:
                min_req = self.oracle.min_tiles_required()
                raise ValueError(
                    f"Spec is infeasible within max_tiles={spec.max_tiles}. "
                    f"Minimum tiles required to close: {min_req}. Consider raising --max-tiles."
                )

        # Reference values (Step 1.2)
        self.n_min: int = 17
        self.p_ref: float = 1.0
        self.ref_board: Dict[Tuple[int, int], PlacedTile] = {}
        if self.enforce_closure:
            min_req = self.oracle.min_tiles_required()
            self.n_min = min_req if min_req is not None else 17
            ref_plan = self.oracle.plan({}, budget=self.n_min)
            if ref_plan is None:
                ref_plan = self.oracle.plan({})
            if ref_plan is not None:
                for (cell, I, O) in ref_plan:
                    vidxs = self.combo_to_vidx.get((I, O), np.array([], dtype=int))
                    best_v = None
                    best_key = None
                    for vi in vidxs:
                        v = self.library[vi]
                        key = (v.total_r, v.variant_id)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_v = v
                    if best_v is not None:
                        self.ref_board[cell] = PlacedTile(cell, best_v)
                from .hydraulic_solver import solve_network
                sr = solve_network(self.ref_board, spec)
                if not sr.ok or sr.stranded or not sr.connected_outlets.all():
                    raise ValueError(
                        f"Reference board solve failed: ok={sr.ok}, stranded={sr.stranded}, "
                        f"connected={sr.connected_outlets}"
                    )
                self.p_ref = float(sr.p_in)
                if not (np.isfinite(self.p_ref) and self.p_ref > 0):
                    raise ValueError(f"Invalid p_ref: {self.p_ref}")
                self.evaluator.set_reference(self.n_min, self.p_ref)

        self._mask_cache: Optional[Tuple[frozenset, np.ndarray]] = None
        self._closure_violation: bool = False
        self._dense_L_cache: Dict[Tuple, int] = {}

        self.inlet_cell = tuple(spec.inlet_cell)
        self.outlet_cells = [tuple(rc) for rc in spec.outlet_cells]
        self.outlet_set = set(self.outlet_cells)
        self.target = np.asarray(spec.target_flows, dtype=float)
        self.pos_outlets = self.target > 0
        self.max_r = max(self.library.max_r, 1e-6)

        # Distance normalization factor (max Manhattan distance on NxN grid)
        self.dist_scale = float(2 * (self.N - 1))

        self.board: Dict[Tuple[int, int], PlacedTile] = {}
        self.evaluation: Evaluation = self.evaluator.evaluate({})
        self.done = False
        self.steps = 0
        self.feature_dim = FEATURE_DIM
        self._prev_conn_prog = 0.0
        self._prev_balance = 0.0
        self.reset()

    def _min_tiles_to_close(self, tiles: TilesMap) -> int:
        sig = tiles_signature(tiles)
        if sig in self._dense_L_cache:
            return self._dense_L_cache[sig]
        rep = analyze_ports(tiles, self.spec)
        if rep.closed:
            self._dense_L_cache[sig] = 0
            return 0
        rem_budget = self.spec.max_tiles - len(tiles)
        lower_b = max(1, len(rep.pending))
        ans = None
        for b in range(lower_b, rem_budget + 1):
            p = self.oracle.plan(tiles, budget=b)
            if p is not None:
                ans = len(p)
                break
        if ans is None:
            p_full = self.oracle.plan(tiles)
            ans = len(p_full) if p_full is not None else rem_budget
        self._dense_L_cache[sig] = ans
        return ans

    def _phi_tile(self) -> float:
        if self.tile_potential != "dense":
            return 0.0
        if not self.enforce_closure:
            return 0.0
        tiles = sides_map(self.board)
        L = self._min_tiles_to_close(tiles)
        proj = len(self.board) + L
        if self.spec.max_tiles <= self.n_min:
            extra_ratio = 0.0
        else:
            extra_ratio = float(np.clip((proj - self.n_min) / (self.spec.max_tiles - self.n_min), 0.0, 1.0))
        return -float(self.weights.lam_extra_tiles * extra_ratio)

    def _phi(self) -> float:
        """Potential-based progress shaping: minus mean distance from placed tiles to unreached outlets."""
        sr = self.evaluation.solve
        if not sr.ok or sr.stranded or not self.pos_outlets.any():
            unreached_indices = [i for i, ok in enumerate(self.pos_outlets) if ok]
        else:
            unreached_indices = [i for i, ok in enumerate(self.pos_outlets & ~sr.connected_outlets) if ok]

        if not unreached_indices or not self.board:
            return 0.0

        dists = []
        placed_coords = np.array(list(self.board.keys()))
        for idx in unreached_indices:
            target_rc = np.array(self.outlet_cells[idx])
            min_d = np.min(np.sum(np.abs(placed_coords - target_rc), axis=1))
            dists.append(min_d / self.dist_scale)

        return -float(np.mean(dists))

    def reset(self) -> None:
        self.board = {}
        self.done = False
        self.steps = 0
        self.evaluation = self.evaluator.evaluate(self.board)
        self._prev_conn_prog = self.evaluation.conn_progress
        self._prev_balance = self.evaluation.balance_score
        self._mask_cache = None
        self._closure_violation = False
        if hasattr(self, "oracle") and len(self.oracle._cache) > 50000:
            self.oracle._cache.clear()
        if hasattr(self, "_dense_L_cache") and len(self._dense_L_cache) > 50000:
            self._dense_L_cache.clear()
        self._prev_phi_tile = self._phi_tile()

    def _legacy_valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        if self.done:
            return mask

        # 1. Empty board case: first tile MUST be placed at inlet_cell (0, 3)
        if not self.board:
            r_in, c_in = self.inlet_cell
            cell_idx = r_in * self.N + c_in
            # Must have inlet on North, cannot have outlet on North (top boundary)
            for vi, v in enumerate(self.library.variants):
                if "N" in v.inlet_sides and "N" not in v.outlet_sides:
                    mask[cell_idx * self.V + vi] = True
            return mask

        # 2. General growth step
        can_place_tiles = len(self.board) < self.spec.max_tiles
        any_tile_valid = False

        if can_place_tiles:
            # Find empty cells adjacent to placed tiles
            candidate_cells: Set[Tuple[int, int]] = set()
            for (r, c) in self.board:
                for side, (dr, dc) in DIRECTION_DELTAS.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.N and 0 <= nc < self.N and (nr, nc) not in self.board:
                        candidate_cells.add((nr, nc))

            for (r, c) in candidate_cells:
                cell_idx = r * self.N + c
                is_chip_outlet = (r, c) in self.outlet_set

                # Determine required inlet connections and blocked sides
                incoming_flow_sides: Set[str] = set()
                blocked_inlet_sides: Set[str] = set()
                blocked_outlet_sides: Set[str] = set()

                for side, (dr, dc) in DIRECTION_DELTAS.items():
                    nbr_r, nbr_c = r + dr, c + dc
                    if not (0 <= nbr_r < self.N and 0 <= nbr_c < self.N):
                        # Boundary edge
                        blocked_inlet_sides.add(side)
                        # Outlet allowed on boundary ONLY if South side on an outlet cell
                        if not (side == "S" and is_chip_outlet):
                            blocked_outlet_sides.add(side)
                    elif (nbr_r, nbr_c) in self.board:
                        # Neighbor is already placed
                        nbr = self.board[(nbr_r, nbr_c)]
                        opp = OPPOSITE[side]
                        if opp in nbr.variant.outlet_sides:
                            # Neighbor is exporting flow into this cell!
                            incoming_flow_sides.add(side)
                        else:
                            # Neighbor does NOT export flow into this cell
                            blocked_inlet_sides.add(side)
                        # No outlet can point back into an already-placed tile
                        blocked_outlet_sides.add(side)

                # If no neighbor exports flow into this cell, it cannot be connected yet
                if not incoming_flow_sides:
                    continue

                for vi, v in enumerate(self.library.variants):
                    # Check chip outlet restriction
                    if is_chip_outlet:
                        if v.outlet_sides != ("S",):
                            continue
                    else:
                        if "S" in v.outlet_sides and r == self.N - 1:
                            continue

                    # Check that all incoming flows are accepted
                    # (The tile must have inlets matching all incoming connections)
                    if not incoming_flow_sides.issubset(set(v.inlet_sides)):
                        continue

                    # Check blocked sides
                    if any(s in blocked_inlet_sides for s in v.inlet_sides):
                        continue
                    if any(s in blocked_outlet_sides for s in v.outlet_sides):
                        continue

                    mask[cell_idx * self.V + vi] = True
                    any_tile_valid = True

        # 3. STOP action validity
        # Valid if all outlet cells are placed AND connected to inlet, OR no valid moves left
        outlets_connected = (
            self.outlet_set.issubset(set(self.board.keys()))
            and self.evaluation.solve.ok
            and self.evaluation.solve.connected_outlets.all()
        )
        if len(self.board) > 0 and (outlets_connected or not any_tile_valid):
            mask[self.stop_action] = True

        return mask

    def valid_action_mask(self) -> np.ndarray:
        if not self.enforce_closure:
            return self._legacy_valid_action_mask()

        if self.done:
            return np.zeros(self.n_actions, dtype=bool)

        sig = board_signature(self.board)
        if self._mask_cache is not None and self._mask_cache[0] == sig:
            return self._mask_cache[1].copy()

        mask = np.zeros(self.n_actions, dtype=bool)
        tiles = sides_map(self.board)
        rep = analyze_ports(tiles, self.spec)

        # 4. If board is closed, enable only STOP
        if rep.closed:
            mask[self.stop_action] = True
            self._mask_cache = (sig, mask)
            return mask.copy()

        # 5. If len(board) < max_tiles, search valid tile actions
        if len(self.board) < self.spec.max_tiles:
            current_plan = self.oracle.plan(tiles)
            shortcut_candidate = current_plan[0] if current_plan else None

            for X in sorted(rep.demand.keys()):
                I, free, options = cell_context(tiles, self.spec, X, self.oracle.allowed)
                cell_idx = X[0] * self.N + X[1]
                for O in options:
                    if shortcut_candidate is not None and (X, I, O) == shortcut_candidate:
                        is_viable = True
                    else:
                        tiles2 = dict(tiles)
                        tiles2[X] = (I, O)
                        is_viable = (self.oracle.plan(tiles2) is not None)

                    if is_viable:
                        vidxs = self.combo_to_vidx.get((I, O))
                        if vidxs is not None:
                            mask[cell_idx * self.V + vidxs] = True

        # 7. Invariant violation check if no actions valid and not closed
        if not np.any(mask) and not rep.closed:
            self._closure_violation = True
            if self.strict_closure:
                raise RuntimeError(
                    f"Closure violation: no action is valid and board is not closed. "
                    f"Tiles placed ({len(self.board)}): {list(self.board.keys())}"
                )
            else:
                logging.getLogger(__name__).warning(
                    "Closure violation: no action valid and board not closed. Enabling STOP to prevent crash."
                )
                mask[self.stop_action] = True

        self._mask_cache = (sig, mask)
        return mask.copy()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode finished; call reset()")

        mask = self.valid_action_mask()
        if not (0 <= action < self.n_actions) or not mask[action]:
            raise ValueError(f"Action {action} is invalid or masked")

        prev_cost = self.evaluation.cost_core
        prev_phi = self._phi()
        prev_phi_tile = self._prev_phi_tile if hasattr(self, "_prev_phi_tile") else 0.0
        prev_conn = self._prev_conn_prog
        prev_balance = self._prev_balance
        self.steps += 1

        old_plan = None
        if self.enforce_closure and action != self.stop_action:
            old_tiles = sides_map(self.board)
            old_plan = self.oracle.plan(old_tiles)

        if action == self.stop_action:
            done = True
        else:
            cell_idx, vi = divmod(action, self.V)
            r, c = divmod(cell_idx, self.N)
            variant = self.library[vi]
            self.board[(r, c)] = PlacedTile((r, c), variant)
            self._mask_cache = None

            if self.enforce_closure:
                new_tiles = sides_map(self.board)
                new_sig = tiles_signature(new_tiles)
                placed_I = tuple(sorted(variant.inlet_sides))
                placed_O = tuple(sorted(variant.outlet_sides))
                if old_plan and len(old_plan) > 0 and old_plan[0] == ((r, c), placed_I, placed_O):
                    if new_sig not in self.oracle._cache:
                        self.oracle._cache[new_sig] = old_plan[1:]

            self.evaluation = self.evaluator.evaluate(self.board)
            done = len(self.board) >= self.spec.max_tiles

            if done and self.enforce_closure:
                if not self.is_closed():
                    self._closure_violation = True
                    if self.strict_closure:
                        raise RuntimeError(f"Episode reached max_tiles={self.spec.max_tiles} without closing.")

        self.done = done
        ev = self.evaluation

        if self.dense_reward:
            # 1. Telescoping core-cost decrease
            reward = (prev_cost - ev.cost_core)
            # 2. Connectivity progress reward
            conn_delta = ev.conn_progress - prev_conn
            reward += self.weights.lam_conn * conn_delta
            # 3. Resistance balance reward
            balance_delta = ev.balance_score - prev_balance
            reward += self.weights.lam_balance * balance_delta
            # 4. Dead branch penalty (only on tile placement)
            if action != self.stop_action:
                cell_idx, _ = divmod(action, self.V)
                r_placed, c_placed = divmod(cell_idx, self.N)
                if self.evaluator.is_newly_placed_dead(self.board, (r_placed, c_placed)):
                    reward -= self.weights.dead_branch_penalty
            # Potential progress shaping
            reward += self.weights.shaping * (self._phi() - prev_phi)
            # Dense tile potential shaping (Step 8a)
            if self.tile_potential == "dense":
                cur_phi_tile = self._phi_tile()
                reward += (cur_phi_tile - prev_phi_tile)
                self._prev_phi_tile = cur_phi_tile
            if done:
                reward -= ev.isolated_cost
        else:
            reward = -ev.cost if done else 0.0

        if done:
            if ev.success:
                reward += self.weights.success_bonus
            elif (ev.solve.ok and not ev.solve.stranded
                  and int(np.sum((self.target > 0) & ~ev.solve.connected_outlets)) == 0
                  and 0.05 < ev.e_q <= 0.15):
                reward += self.weights.partial_success_bonus

            # Step 6.4: Structural penalty (terminal-only, exactly once)
            if self.tile_potential == "dense":
                # In dense mode, extra_n part was already shaped step-by-step
                rem_struct = (
                    self.weights.lam_pressure * ev.struct_parts["press_n"]
                    + self.weights.lam_loops * ev.struct_parts["loops_n"]
                )
                if getattr(self.weights, "lam_robust", 0.0) > 0:
                    rem_struct += self.weights.lam_robust * ev.struct_parts.get("robust_n", 0.0)
                reward -= rem_struct
            else:
                reward -= ev.struct_cost

        reward -= self.weights.step_penalty

        self._prev_conn_prog = ev.conn_progress
        self._prev_balance = ev.balance_score

        if self.enforce_closure:
            rep = analyze_ports(sides_map(self.board), self.spec)
            info = {
                "evaluation": ev,
                "closed": bool(rep.closed),
                "pending": len(rep.pending),
                "closure_violation": bool(self._closure_violation),
            }
        else:
            info = {
                "evaluation": ev,
                "closed": False,
                "pending": 0,
                "closure_violation": False,
            }
        return float(reward), done, info

    def is_closed(self) -> bool:
        """Check if current board topology is completely closed."""
        return analyze_ports(sides_map(self.board), self.spec).closed

    def force_place(self, cell: Tuple[int, int], variant: TileVariant) -> None:
        """Force-place a tile on the board, used by curriculum pre-placement."""
        tiles = sides_map(self.board)
        allowed = self.oracle.allowed if self.enforce_closure else allowed_combos(self.library)
        I, free, options = cell_context(tiles, self.spec, cell, allowed)

        v_I = tuple(sorted(variant.inlet_sides))
        v_O = tuple(sorted(variant.outlet_sides))

        if v_I != I:
            raise ValueError(f"Cannot force place variant {variant.variant_id} at {cell}: inlets {v_I} != required {I}")
        if v_O not in options:
            raise ValueError(f"Cannot force place variant {variant.variant_id} at {cell}: outlets {v_O} not in legal options {options}")

        if self.enforce_closure:
            tiles2 = dict(tiles)
            tiles2[cell] = (v_I, v_O)
            if self.oracle.plan(tiles2) is None:
                raise ValueError(f"Placement of variant {variant.variant_id} at {cell} is not viable.")

        self.board[cell] = PlacedTile(cell, variant)
        self._mask_cache = None
        self.evaluation = self.evaluator.evaluate(self.board)
        self._prev_conn_prog = self.evaluation.conn_progress
        self._prev_balance = self.evaluation.balance_score
        self._prev_e_q = self.evaluation.e_q
        if len(self.board) >= self.spec.max_tiles:
            self.done = True

    def features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, valid_action_ids) for all unmasked actions."""
        mask = self.valid_action_mask()
        valid_ids = np.flatnonzero(mask)
        if len(valid_ids) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32), valid_ids

        ev = self.evaluation
        sr = ev.solve
        q_ref = max(self.spec.inlet_flow, 1e-12)

        # Global features
        n_out = len(self.outlet_cells)
        conn_feat = sr.connected_outlets.astype(float) if sr.ok else np.zeros(n_out)
        flow_fracs = (sr.outlet_flows / q_ref) if sr.ok else np.zeros(n_out)
        rel_errors = np.clip(ev.rel_err, -CLIP_VAL, CLIP_VAL) / CLIP_VAL

        # Per-branch resistance estimate (3)
        r_branch = np.zeros(n_out, dtype=np.float32)
        for k, (or_, oc) in enumerate(self.outlet_cells):
            tiles_on_branch = [
                placed for (r, c), placed in self.board.items()
                if (abs(r - 0) + abs(c - 3)) + (abs(r - or_) + abs(c - oc))
                   <= (abs(or_ - 0) + abs(oc - 3)) + 2
            ]
            if tiles_on_branch:
                total_r_branch = sum(t.variant.total_r for t in tiles_on_branch)
                max_r_branch = len(tiles_on_branch) * self.max_r
                r_branch[k] = float(total_r_branch / max_r_branch)
            else:
                r_branch[k] = 0.0

        # Resistance deviation per outlet (3)
        R_dev = np.zeros(n_out, dtype=np.float32)
        if sr.ok and not sr.stranded:
            for k in range(n_out):
                q_k = sr.outlet_flows[k]
                q_star_k = self.target[k]
                R_dev[k] = float(np.clip((q_star_k / (q_k + 1e-15)) - 1.0, -CLIP_VAL, CLIP_VAL) / CLIP_VAL)

        global_vec = np.concatenate([
            conn_feat,
            flow_fracs,
            rel_errors,
            [min(ev.e_q, CLIP_VAL) / CLIP_VAL],
            [len(self.board) / float(self.spec.max_tiles)],
            [self._phi()],
            r_branch,
            R_dev,
        ])

        rows = []
        for act in valid_ids:
            if act == self.stop_action:
                # Dedicated feature row for STOP action
                feat = np.zeros(self.feature_dim, dtype=np.float32)
                feat[:len(global_vec)] = global_vec
                feat[-1] = 1.0  # STOP flag
                rows.append(feat)
            else:
                cell_idx, vi = divmod(act, self.V)
                r, c = divmod(cell_idx, self.N)
                v = self.library[vi]

                # 1. Cell features (8)
                d_outlets = [
                    (abs(r - or_) + abs(c - oc)) / self.dist_scale
                    for or_, oc in self.outlet_cells
                ]
                is_inlet_adj = float(abs(r - self.inlet_cell[0]) + abs(c - self.inlet_cell[1]) == 1)
                nbr_count = sum(
                    1 for dr, dc in DIRECTION_DELTAS.values()
                    if 0 <= r + dr < self.N and 0 <= c + dc < self.N and (r + dr, c + dc) in self.board
                ) / 4.0
                is_outlet_cell = float((r, c) in self.outlet_set)

                cell_feats = [r / (self.N - 1), c / (self.N - 1)] + d_outlets + [
                    is_inlet_adj,
                    nbr_count,
                    is_outlet_cell,
                ]

                # 2. Variant features (7)
                r_n = v.R_per_side.get("N", 0.0) / self.max_r
                r_s = v.R_per_side.get("S", 0.0) / self.max_r
                r_e = v.R_per_side.get("E", 0.0) / self.max_r
                r_w = v.R_per_side.get("W", 0.0) / self.max_r
                var_feats = [
                    len(v.inlet_sides) / 3.0,
                    len(v.outlet_sides) / 3.0,
                    r_n,
                    r_s,
                    r_e,
                    r_w,
                    v.total_r / (4.0 * self.max_r),
                ]

                # 3. Progress toward each outlet (3)
                prog_feats = [
                    np.clip(1.0 - ((abs(r - or_) + abs(c - oc)) / self.dist_scale), 0.0, 1.0)
                    for or_, oc in self.outlet_cells
                ]

                # 4. Action physics alignment & diversity (2)
                k_nearest = int(np.argmin(d_outlets))
                R_def_k = R_dev[k_nearest]
                alignment = float((1.0 - v.total_r / self.max_r) if R_def_k > 0 else (v.total_r / self.max_r))
                n_same = sum(1 for placed in self.board.values() if placed.variant.variant_id == v.variant_id)
                diversity = float(1.0 / (1.0 + n_same))

                feat = np.concatenate([
                    global_vec,
                    cell_feats,
                    var_feats,
                    prog_feats,
                    [alignment, diversity],
                    [0.0],  # STOP flag = 0
                ]).astype(np.float32)
                rows.append(feat)

        return np.vstack(rows), valid_ids
