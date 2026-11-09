"""Design specification (Single inlet, 3 fixed outlets, no split/merge channels) and reward weights."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Sequence, Tuple

DEFAULT_INLET_FLOW = 1e-4  # m^3/s (or relative flow units)
TARGET_SUM_RTOL = 1e-3


def normalize_targets(values: Sequence[float], total: float) -> List[float]:
    """Scale ``values`` so they sum to ``total`` (mass conservation)."""
    s = float(sum(values))
    if s <= 0:
        raise ValueError("Target flows must have a positive sum")
    return [float(v) * total / s for v in values]


@dataclass
class DesignSpec:
    """User design specification for tile-based topology synthesis."""

    grid_size: int = 8
    inlet_cell: Tuple[int, int] = (0, 3)
    outlet_cells: List[Tuple[int, int]] = field(
        default_factory=lambda: [(7, 0), (7, 3), (7, 7)]
    )
    inlet_flow: float = DEFAULT_INLET_FLOW
    target_flows: List[float] = field(default_factory=list)
    R_stub: float = 1.0  # Terminal stub resistance for inlet and outlet ports
    max_tiles: int = 25  # Maximum number of tiles placed per design

    def cell_id(self, rc: Tuple[int, int]) -> int:
        return int(rc[0]) * self.grid_size + int(rc[1])

    def cell_rc(self, cid: int) -> Tuple[int, int]:
        return divmod(int(cid), self.grid_size)

    def validate(self) -> None:
        n = self.grid_size
        if n < 2:
            raise ValueError("grid_size must be >= 2")

        r_in, c_in = self.inlet_cell
        if r_in != 0 or not (0 <= c_in < n):
            raise ValueError(f"Inlet cell must be on row 0, got {self.inlet_cell}")

        if not (0 < self.inlet_flow < float("inf")):
            raise ValueError("inlet_flow must be positive and finite")

        if not self.outlet_cells:
            raise ValueError("At least one outlet cell is required")

        self.inlet_cell = tuple(self.inlet_cell)
        self.outlet_cells = [tuple(rc) for rc in self.outlet_cells]

        seen = {tuple(self.inlet_cell)}
        for rc in self.outlet_cells:
            r, c = int(rc[0]), int(rc[1])
            if r != n - 1 or not (0 <= c < n):
                raise ValueError(f"Outlet cell {rc} must be on bottom row ({n - 1})")
            if (r, c) in seen:
                raise ValueError(f"Cell {rc} is specified multiple times")
            seen.add((r, c))

        if self.R_stub < 0:
            raise ValueError("R_stub must be non-negative")

        if self.max_tiles < 1:
            raise ValueError("max_tiles must be >= 1")

        if self.target_flows:
            if len(self.target_flows) != len(self.outlet_cells):
                raise ValueError("target_flows length must match number of outlet_cells")
            if any(q < 0 for q in self.target_flows):
                raise ValueError("target_flows must be non-negative")
            total = float(sum(self.target_flows))
            if abs(total - self.inlet_flow) > TARGET_SUM_RTOL * self.inlet_flow:
                raise ValueError(
                    f"Target flows sum to {total:g}, but inlet_flow is {self.inlet_flow:g}. "
                    "Use normalize_targets() to scale them."
                )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["inlet_cell"] = list(self.inlet_cell)
        d["outlet_cells"] = [list(rc) for rc in self.outlet_cells]
        return d

    @staticmethod
    def from_dict(d: dict) -> "DesignSpec":
        d = dict(d)
        d["inlet_cell"] = tuple(d["inlet_cell"])
        d["outlet_cells"] = [tuple(rc) for rc in d["outlet_cells"]]
        return DesignSpec(**d)


@dataclass
class RewardWeights:
    """Hyperparameters for RL reward calculation."""

    lam_q: float = 1.0           # Flow error weight
    lam_c: float = 0.05          # Complexity penalty (tiles placed / max_tiles)
    lam_v: float = 0.5           # Penalty for unconnected outlets
    shaping: float = 20.0        # Potential shaping toward unconnected outlets
    success_bonus: float = 1.0   # Bonus when all outlets meet flow targets within tolerance
    dead_end_penalty: float = 0.2# Penalty per isolated / dead-end tile at episode end
    tolerance: float = 0.05      # 5% relative error per outlet for success bonus
    error_clip: float = 3.0      # Max clip for per-outlet relative error
    step_penalty: float = 0.0    # Optional per-step penalty
    lam_conn: float = 0.5        # Connectivity progress weight
    lam_balance: float = 0.3     # Resistance balance reward weight
    partial_success_bonus: float = 0.3  # Bonus for near-success
    dead_branch_penalty: float = 0.1    # Per-step dead branch penalty

    # New structural terms (fix_plan.md Section 6.1 & Step 8b)
    lam_extra_tiles: float = 0.0 # Penalty for tiles beyond n_min
    lam_pressure: float = 0.0    # Penalty for inlet pressure above reference design
    lam_loops: float = 0.0       # Penalty for cycles in tile graph
    p_clip: float = 20.0         # Cap on p_ratio before log
    loops_cap: int = 8           # Cap on loop count for normalization
    lam_robust: float = 0.0      # Robustness penalty (Step 8b)
    robust_eps: float = 0.10     # Perturbation epsilon
    robust_samples: int = 8      # Number of perturbed solves

    def check_structural_budget(self, allow_large_struct: bool = False) -> None:
        """Enforce: lam_extra_tiles + lam_pressure + lam_loops + lam_robust <= success_bonus - partial_success_bonus - 0.1."""
        limit = self.success_bonus - self.partial_success_bonus - 0.1
        total_struct = self.lam_extra_tiles + self.lam_pressure + self.lam_loops + self.lam_robust
        if total_struct > limit + 1e-9:
            msg = (
                f"Structural weights sum ({total_struct:.3f}) exceeds safety limit ({limit:.3f}) "
                f"[success_bonus ({self.success_bonus:g}) - partial_success_bonus ({self.partial_success_bonus:g}) - 0.1]."
            )
            if allow_large_struct:
                import warnings
                warnings.warn(msg + " Continuing because --allow-large-struct is set.")
            else:
                raise ValueError(msg + " Use --allow-large-struct to override.")


PRESETS = {
    "legacy": {"lam_extra_tiles": 0.0, "lam_pressure": 0.0, "lam_loops": 0.0},
    "compact": {"lam_extra_tiles": 0.30, "lam_pressure": 0.15, "lam_loops": 0.0},
    "low-pressure": {"lam_extra_tiles": 0.15, "lam_pressure": 0.30, "lam_loops": 0.0},
    "tree": {"lam_extra_tiles": 0.30, "lam_pressure": 0.15, "lam_loops": 0.10},
}


def apply_preset(weights: RewardWeights, preset_name: str) -> RewardWeights:
    """Apply a named objective preset to RewardWeights."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown objective preset: {preset_name}. Choices: {list(PRESETS.keys())}")
    for k, v in PRESETS[preset_name].items():
        setattr(weights, k, v)
    return weights


