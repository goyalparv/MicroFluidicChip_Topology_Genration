"""PhysicsAwareRL_v2: Tile-based microfluidic chip topology synthesis using COMSOL-measured libraries."""
from __future__ import annotations

from .config_v2 import DesignSpec, RewardWeights, normalize_targets
from .library_v2 import Library, PlacedTile, TileVariant
from .hydraulic_solver_v2 import SolveResult, solve_network_v2
from .reward_v2 import Evaluation, Evaluator
from .environment_v2 import TopologyEnv
from .agent_v2 import PolicyNet, ValueNet, Step, RolloutStep, rollout, rollout_ppo, compute_gae
from .curriculum_v2 import CurriculumManager
from .train_v2 import train_agent, train_agent_ppo
from .visualize_stage1 import render_board_ascii, render_chip_image

__all__ = [
    "DesignSpec",
    "RewardWeights",
    "normalize_targets",
    "Library",
    "PlacedTile",
    "TileVariant",
    "SolveResult",
    "solve_network_v2",
    "Evaluation",
    "Evaluator",
    "TopologyEnv",
    "PolicyNet",
    "ValueNet",
    "Step",
    "RolloutStep",
    "rollout",
    "rollout_ppo",
    "compute_gae",
    "CurriculumManager",
    "train_agent",
    "train_agent_ppo",
]
