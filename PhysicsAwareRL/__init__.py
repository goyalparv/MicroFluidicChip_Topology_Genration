"""PhysicsAwareRL: Tile-based microfluidic chip topology synthesis using COMSOL-measured libraries."""
from __future__ import annotations

from .config import DesignSpec, RewardWeights, normalize_targets
from .library import Library, PlacedTile, TileVariant
from .hydraulic_solver import SolveResult, solve_network, solve_network_v2
from .reward import Evaluation, Evaluator
from .environment import TopologyEnv
from .agent import PolicyNet, ValueNet, Step, RolloutStep, rollout, rollout_ppo, compute_gae
from .curriculum import CurriculumManager
from .train import train_agent, train_agent_ppo
from .visualize_stage1 import render_board_ascii, render_chip_image

__all__ = [
    "DesignSpec",
    "RewardWeights",
    "normalize_targets",
    "Library",
    "PlacedTile",
    "TileVariant",
    "SolveResult",
    "solve_network",
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
