"""Curriculum manager for progressive episode difficulty in microfluidic RL (Stage 1)."""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

from .library_v2 import DIRECTION_DELTAS, Library, PlacedTile, TileVariant

logger = logging.getLogger(__name__)


def _find_ns_variant(library: Library) -> Optional[TileVariant]:
    """Find variant with exact combo (('N',), ('S',)), minimal total_r, breaking ties by variant_id."""
    candidates = [
        v for v in library.variants
        if tuple(sorted(v.inlet_sides)) == ("N",) and tuple(sorted(v.outlet_sides)) == ("S",)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda v: (v.total_r, v.variant_id))
    return candidates[0]


class CurriculumManager:
    """Manages progressive curriculum stages for TopologyEnv training.

    Stage 0 (warmup): Pre-places inlet tile at (0,3) + downstream tile at (1,3).
    Stage 1 (easy): Pre-places inlet tile at (0,3) only.
    Stage 2 (normal): Full empty board (standard episode).
    Stage 3 (hard): Empty board with tighter flow error tolerance.
    """

    STAGES = ["warmup", "easy", "normal", "hard"]

    def __init__(
        self,
        total_episodes: int = 2000,
        window: int = 50,
        threshold: float = 0.4,
        q_gate: float = 0.15,
        min_stage_episodes: int = 100,
        max_stage_fraction: float = 0.25,
        final_normal_fraction: float = 0.4,
        mode: str = "sequential",
        use_hard_stage: bool = False,
        hard_tolerance_scale: float = 0.5,
        seed: int = 0,
    ):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if not (0.01 <= q_gate <= 0.5):
            raise ValueError(f"q_gate must be in [0.01, 0.5], got {q_gate}")
        if min_stage_episodes < window:
            raise ValueError(f"min_stage_episodes ({min_stage_episodes}) must be >= window ({window})")
        if max_stage_fraction * total_episodes < max(window, min_stage_episodes):
            raise ValueError(
                f"max_stage_fraction * total_episodes ({max_stage_fraction * total_episodes}) "
                f"must be >= max(window, min_stage_episodes) ({max(window, min_stage_episodes)})"
            )
        if 2.0 * max_stage_fraction > (1.0 - final_normal_fraction) + 1e-9:
            raise ValueError(
                f"2 * max_stage_fraction ({2.0 * max_stage_fraction}) must be <= "
                f"1 - final_normal_fraction ({1.0 - final_normal_fraction})"
            )
        if mode not in ("sequential", "mixed", "off"):
            raise ValueError(f"Unknown mode: {mode}, expected 'sequential', 'mixed', or 'off'")

        self.total_episodes = int(total_episodes)
        self.window = int(window)
        self.threshold = float(threshold)
        self.q_gate = float(q_gate)
        self.min_stage_episodes = int(min_stage_episodes)
        self.max_stage_fraction = float(max_stage_fraction)
        self.final_normal_fraction = float(final_normal_fraction)
        self.mode = mode
        self.use_hard_stage = bool(use_hard_stage)
        self.hard_tolerance_scale = float(hard_tolerance_scale)
        self.seed = int(seed)

        self.rng = np.random.default_rng(seed + 7)
        self.stage: int = 0
        self.last_stage: int = 3 if self.use_hard_stage else 2
        self.episodes_in_stage: int = 0
        self.seen: int = 0
        self.history: Deque[bool] = deque(maxlen=self.window)
        self.stage_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        self.used: int = 0
        self.transitions: List[Dict[str, Any]] = []
        self.preplace_fallbacks: int = 0
        self.base_tolerance: Optional[float] = None

    def setup_episode(self, env, library: Library) -> int:
        """Configure episode start state according to curriculum stage. Returns stage used."""
        env.reset()
        if self.base_tolerance is None:
            self.base_tolerance = float(env.weights.tolerance)
        else:
            env.weights.tolerance = self.base_tolerance

        if self.mode == "off":
            self.used = 2
            env.preplaced = ()
            env.steps = 0
            return self.used

        if self.mode == "mixed":
            u = float(self.seen) / max(float(self.total_episodes), 1.0)
            f = max(0.0, 1.0 - 2.0 * u)
            p_warmup = 0.5 * f
            p_easy = 0.3 * f
            p_normal = 1.0 - p_warmup - p_easy
            probs = np.array([p_warmup, p_easy, p_normal], dtype=float)
            probs = np.clip(probs, 0.0, 1.0)
            probs = probs / np.sum(probs)
            self.used = int(self.rng.choice([0, 1, 2], p=probs))
        else:
            self.used = self.stage

        if self.used == 3 and self.use_hard_stage:
            env.weights.tolerance = self.base_tolerance * self.hard_tolerance_scale

        if self.used == 0:
            ok = self._try_preplace(env, library, stage=0)
            if not ok:
                self.preplace_fallbacks += 1
                ok1 = self._try_preplace(env, library, stage=1)
                if not ok1:
                    self.preplace_fallbacks += 1
                    env.reset()
        elif self.used == 1:
            ok = self._try_preplace(env, library, stage=1)
            if not ok:
                self.preplace_fallbacks += 1
                env.reset()

        env.preplaced = tuple(sorted(env.board.keys()))
        env.steps = 0
        return self.used

    def _try_preplace(self, env, library: Library, stage: int) -> bool:
        try:
            v = _find_ns_variant(library)
            if v is None:
                return False
            r_in, c_in = env.inlet_cell
            env.force_place((r_in, c_in), v)
            if stage == 0:
                r_next, c_next = r_in + 1, c_in
                if not (0 <= r_next < env.N and 0 <= c_next < env.N):
                    return True
                if (r_next, c_next) in env.outlet_set:
                    return True
                env.force_place((r_next, c_next), v)
            return True
        except Exception as e:
            logger.warning(f"Pre-placement failed: {e}")
            return False

    def advance(self, info: Union[Dict[str, Any], bool, None] = None, *, success: Optional[bool] = None) -> Optional[str]:
        """Record episode outcome and advance curriculum stage if conditions are met."""
        if success is not None:
            info = {"success": bool(success), "closed": True, "e_q": 0.0 if success else 1.0}
        elif isinstance(info, bool):
            info = {"success": info, "closed": True, "e_q": 0.0 if info else 1.0}
        elif info is None:
            info = {"success": False, "closed": False, "e_q": 1.0}

        is_success = bool(info.get("success", False))
        is_closed = bool(info.get("closed", False))
        e_q = float(info.get("e_q", 1.0))
        good = is_success or (is_closed and e_q <= self.q_gate)

        self.history.append(good)
        self.seen += 1
        self.episodes_in_stage += 1
        self.stage_counts[self.used] = self.stage_counts.get(self.used, 0) + 1

        if self.mode != "sequential":
            return None

        if self.stage >= self.last_stage:
            return None

        reason = None
        target = self.stage

        # Rule 1: Final guarantee - jump straight to normal stage
        if self.stage < 2 and self.seen >= self.total_episodes * (1.0 - self.final_normal_fraction):
            target = 2
            reason = "final-guarantee"
        # Rule 2: Time fallback
        elif self.episodes_in_stage >= self.max_stage_fraction * self.total_episodes:
            target = self.stage + 1
            reason = "time"
        # Rule 3: Progress
        elif (
            self.episodes_in_stage >= max(self.window, self.min_stage_episodes)
            and len(self.history) == self.window
            and (sum(self.history) / float(self.window)) >= self.threshold
        ):
            target = self.stage + 1
            reason = "progress"

        if reason is not None:
            rate = float(sum(self.history)) / float(len(self.history)) if self.history else 0.0
            old_stage = self.stage
            self.stage = target
            self.history.clear()
            self.episodes_in_stage = 0
            trans = {
                "episode": self.seen,
                "from_stage": old_stage,
                "to_stage": self.stage,
                "reason": reason,
                "rate": rate,
                "window": self.window,
            }
            self.transitions.append(trans)
            print(
                f"[Curriculum] ep {self.seen}: stage {old_stage} -> {self.stage} "
                f"(reason={reason}, rate={rate:.2f}, window={self.window})",
                flush=True,
            )
            return reason

        return None

    def snapshot(self) -> Dict[str, Any]:
        """Return diagnostic snapshot of current curriculum state."""
        rate = float(sum(self.history)) / float(len(self.history)) if self.history else 0.0
        return {
            "stage": self.stage,
            "episodes_in_stage": self.episodes_in_stage,
            "rate": rate,
            "stage_counts": dict(self.stage_counts),
            "transitions": list(self.transitions),
        }
