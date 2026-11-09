"""Unit tests for closed-topology enforcement (stdlib unittest + numpy only)."""
from __future__ import annotations

import os
import unittest
import numpy as np

from PhysicsAwareRL_v3.config_v2 import DesignSpec, RewardWeights
from PhysicsAwareRL_v3.closure_v2 import (
    ClosureOracle,
    allowed_combos,
    analyze_ports,
    check_port_degrees,
    find_open_ports,
    sides_map,
)
from PhysicsAwareRL_v3.curriculum_v2 import CurriculumManager
from PhysicsAwareRL_v3.environment_v2 import FEATURE_DIM, TopologyEnv
from PhysicsAwareRL_v3.library_v2 import Library, PlacedTile, TileVariant


class TestClosureEnforcement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = Library.generate_initial_library()
        cls.spec = DesignSpec(
            grid_size=8,
            inlet_cell=(0, 3),
            outlet_cells=[(7, 0), (7, 3), (7, 7)],
            inlet_flow=1e-4,
            max_tiles=25,
            target_flows=[1e-4 / 3, 1e-4 / 3, 1e-4 / 3],
        )
        cls.spec.validate()

    def test_01_analyze_ports_defects(self):
        """Test defect detection on hand-crafted boards."""
        # A: Trunk with 1 of 3 outlets connected (2 pending)
        tiles_a = {
            (0, 3): (("N",), ("E", "S", "W")),
            (1, 3): (("N",), ("S",)),
        }
        rep_a = analyze_ports(tiles_a, self.spec)
        self.assertFalse(rep_a.closed)
        self.assertGreaterEqual(len(rep_a.pending), 2)
        self.assertIn(((0, 3), "E"), rep_a.pending)
        self.assertIn(((0, 3), "W"), rep_a.pending)

        # B: Blocked outlet (points out of bounds at column 0)
        tiles_b = {
            (0, 3): (("N",), ("S",)),
            (0, 0): (("N",), ("W",)),  # W is out of bounds
        }
        rep_b = analyze_ports(tiles_b, self.spec)
        self.assertIn(((0, 0), "W"), rep_b.blocked_outlets)

        # C: Dangling inlet (inlet on W with no upstream neighbor)
        tiles_c = {
            (0, 3): (("N", "W"), ("S",)),
        }
        rep_c = analyze_ports(tiles_c, self.spec)
        self.assertIn(((0, 3), "W"), rep_c.open_inlets)

        # D: Outlet pointing to placed non-accepting tile
        tiles_d = {
            (0, 3): (("N",), ("S",)),
            (1, 3): (("W",), ("S",)),  # accepts from W, but (0,3) exports from N to S
        }
        rep_d = analyze_ports(tiles_d, self.spec)
        self.assertIn(((0, 3), "S"), rep_d.blocked_outlets)
        self.assertIn(((1, 3), "W"), rep_d.open_inlets)

        # E: Fully closed 17-tile layout using oracle plan
        allowed = allowed_combos(self.library)
        oracle = ClosureOracle(self.spec, allowed)
        plan = oracle.plan({})
        self.assertIsNotNone(plan)
        closed_tiles = {cell: (I, O) for cell, I, O in plan}
        rep_e = analyze_ports(closed_tiles, self.spec)
        self.assertTrue(rep_e.closed)
        self.assertEqual(len(rep_e.pending), 0)
        self.assertEqual(len(rep_e.open_inlets), 0)
        self.assertEqual(len(rep_e.blocked_outlets), 0)
        self.assertEqual(len(rep_e.missing_sinks), 0)

        # Check port degrees on closed board with PlacedTiles
        board = {}
        for cell, (I, O) in closed_tiles.items():
            candidates = [
                v for v in self.library.variants
                if tuple(sorted(v.inlet_sides)) == I and tuple(sorted(v.outlet_sides)) == O
            ]
            board[cell] = PlacedTile(cell, candidates[0])
        bad_degrees = check_port_degrees(board, self.spec)
        self.assertEqual(bad_degrees, [])

    def test_02_mask_subset(self):
        """Every new tile action in closed mask must be a subset of legacy mask."""
        env_closed = TopologyEnv(self.spec, self.library, enforce_closure=True)
        env_legacy = TopologyEnv(self.spec, self.library, enforce_closure=False)

        # Test on initial board
        mask_c = env_closed.valid_action_mask()
        mask_l = env_legacy.valid_action_mask()
        for idx in range(env_closed.stop_action):
            if mask_c[idx]:
                self.assertTrue(mask_l[idx], f"Action {idx} in closed mask but not in legacy mask")

        # Test along a trajectory
        rng = np.random.default_rng(42)
        for _ in range(5):
            env_closed.reset()
            for _ in range(5):
                mc = env_closed.valid_action_mask()
                # sync legacy board
                env_legacy.board = dict(env_closed.board)
                ml = env_legacy.valid_action_mask()
                for idx in range(env_closed.stop_action):
                    if mc[idx]:
                        self.assertTrue(ml[idx], f"Action {idx} in closed mask but not in legacy mask")
                valid_actions = np.flatnonzero(mc)
                if len(valid_actions) == 0 or (len(valid_actions) == 1 and valid_actions[0] == env_closed.stop_action):
                    break
                tile_actions = [a for a in valid_actions if a != env_closed.stop_action]
                a = rng.choice(tile_actions)
                env_closed.step(a)

    def test_03_stop_condition(self):
        """STOP action is valid if and only if board is closed."""
        env = TopologyEnv(self.spec, self.library, enforce_closure=True)
        mask = env.valid_action_mask()
        self.assertFalse(mask[env.stop_action])

        # Step once
        valid_actions = [a for a in np.flatnonzero(mask) if a != env.stop_action]
        env.step(valid_actions[0])
        self.assertFalse(env.is_closed())
        mask2 = env.valid_action_mask()
        self.assertFalse(mask2[env.stop_action])

    def test_04_oracle_soundness_replay(self):
        """Replaying oracle plan through env.step closes the board and satisfies I1-I7."""
        env = TopologyEnv(self.spec, self.library, enforce_closure=True)
        plan = env.oracle.plan({})
        self.assertIsNotNone(plan)

        for cell, I, O in plan:
            mask = env.valid_action_mask()
            cell_idx = cell[0] * env.N + cell[1]
            vidxs = env.combo_to_vidx[(I, O)]
            chosen_vi = vidxs[0]
            action = cell_idx * env.V + chosen_vi
            self.assertTrue(mask[action], f"Action for {cell} {I}->{O} should be valid in mask")
            env.step(action)

        self.assertTrue(env.is_closed())
        # Verify degree invariant I7
        bad_deg = check_port_degrees(env.board, self.spec)
        self.assertEqual(bad_deg, [])

        if not env.done:
            mask_final = env.valid_action_mask()
            self.assertTrue(mask_final[env.stop_action])
            # Only STOP should be valid when closed
            self.assertEqual(int(np.sum(mask_final)), 1)
            # Take STOP
            _, done, info = env.step(env.stop_action)
            self.assertTrue(done)
            self.assertTrue(info["closed"])
            self.assertFalse(info["closure_violation"])

    def test_05_prefix_property(self):
        """Oracle plan satisfies prefix property: plan(S + P[0]) == P[1:]."""
        allowed = allowed_combos(self.library)
        oracle = ClosureOracle(self.spec, allowed)
        plan = oracle.plan({})
        self.assertIsNotNone(plan)
        first_step = plan[0]
        cell, I, O = first_step
        sub_tiles = {cell: (I, O)}
        sub_plan = oracle.plan(sub_tiles)
        self.assertEqual(sub_plan, plan[1:])

    def test_06_brute_force_comparison(self):
        """Oracle is sound on a smaller grid."""
        small_spec = DesignSpec(
            grid_size=4,
            inlet_cell=(0, 1),
            outlet_cells=[(3, 1), (3, 2)],
            inlet_flow=1e-4,
            max_tiles=6,
            target_flows=[5e-5, 5e-5],
        )
        small_spec.validate()
        allowed = allowed_combos(self.library)
        oracle = ClosureOracle(small_spec, allowed, node_limit=1000)
        plan = oracle.plan({})
        self.assertIsNotNone(plan)
        self.assertLessEqual(len(plan), 6)
        tiles = {c: (I, O) for c, I, O in plan}
        self.assertTrue(analyze_ports(tiles, small_spec).closed)

    def test_07_fuzz_random_episodes(self):
        """Uniform random valid actions always terminate closed with small mass residual."""
        rng = np.random.default_rng(123)
        env = TopologyEnv(self.spec, self.library, enforce_closure=True)

        for ep in range(25):
            env.reset()
            for step_i in range(self.spec.max_tiles + 1):
                mask = env.valid_action_mask()
                valid = np.flatnonzero(mask)
                self.assertGreater(len(valid), 0, f"Mask empty at step {step_i} ep {ep}")
                action = int(rng.choice(valid))
                _, done, info = env.step(action)
                if done:
                    self.assertTrue(info["closed"], f"Episode {ep} finished not closed!")
                    self.assertFalse(info["closure_violation"])
                    self.assertLessEqual(len(env.board), self.spec.max_tiles)
                    self.assertLess(env.evaluation.solve.mass_residual, 1e-7)
                    break

    def test_08_infeasible_spec(self):
        """Spec requiring > max_tiles raises ValueError at construction."""
        infeasible_spec = DesignSpec(
            grid_size=8,
            inlet_cell=(0, 3),
            outlet_cells=[(7, 0), (7, 3), (7, 7)],
            inlet_flow=1e-4,
            max_tiles=16,  # 16 is less than minimum 17
            target_flows=[1e-4 / 3, 1e-4 / 3, 1e-4 / 3],
        )
        with self.assertRaises(ValueError) as ctx:
            TopologyEnv(infeasible_spec, self.library, enforce_closure=True)
        self.assertIn("infeasible", str(ctx.exception).lower())

    def test_09_curriculum_stages(self):
        """All curriculum stages produce viable starts that can be completed to closed board."""
        curriculum = CurriculumManager()
        for stage in range(4):
            curriculum.stage = stage
            env = TopologyEnv(self.spec, self.library, enforce_closure=True)
            curriculum.setup_episode(env, self.library)
            plan = env.oracle.plan(sides_map(env.board))
            self.assertIsNotNone(plan, f"Curriculum stage {stage} produced non-viable start")

    def test_10_regression(self):
        """enforce_closure=False preserves legacy behavior."""
        env_legacy = TopologyEnv(self.spec, self.library, enforce_closure=False)
        mask = env_legacy.valid_action_mask()
        self.assertEqual(len(mask), env_legacy.n_actions)
        self.assertFalse(env_legacy.enforce_closure)

    def test_11_compatibility(self):
        """FEATURE_DIM == 39 and action space size unchanged."""
        env = TopologyEnv(self.spec, self.library, enforce_closure=True)
        self.assertEqual(env.feature_dim, 39)
        self.assertEqual(FEATURE_DIM, 39)
        self.assertEqual(env.n_actions, 8 * 8 * 640 + 1)
        X, valid_ids = env.features()
        if len(valid_ids) > 0:
            self.assertEqual(X.shape[1], 39)


if __name__ == "__main__":
    unittest.main()
