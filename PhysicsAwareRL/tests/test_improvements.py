"""Unit tests for the improved physics-aware RL implementation in PhysicsAwareRL_v3."""
from __future__ import annotations

import unittest
import numpy as np

from PhysicsAwareRL.config import DesignSpec, RewardWeights
from PhysicsAwareRL.library import Library, PlacedTile
from PhysicsAwareRL.reward import Evaluator
from PhysicsAwareRL.environment import TopologyEnv
from PhysicsAwareRL.agent import PolicyNet, ValueNet, rollout_ppo, compute_gae
from PhysicsAwareRL.curriculum import CurriculumManager
from PhysicsAwareRL.train import train_agent_ppo


class TestRLImprovements(unittest.TestCase):
    def setUp(self):
        self.spec = DesignSpec(
            outlet_cells=[(7, 0), (7, 3), (7, 7)],
            target_flows=[5e-5, 3e-5, 2e-5],
            inlet_flow=1e-4,
        )
        self.lib = Library.generate_initial_library()
        self.env = TopologyEnv(self.spec, self.lib)

    def test_feature_dim_is_39(self):
        self.assertEqual(self.env.feature_dim, 39)
        x, ids = self.env.features()
        self.assertEqual(x.shape[1], 39)
        self.assertEqual(len(x), len(ids))
        self.assertTrue(np.all(np.isfinite(x)))

    def test_evaluator_metrics_and_dead_branch(self):
        evaluator = self.env.evaluator
        ev_empty = evaluator.evaluate({})
        self.assertEqual(ev_empty.conn_progress, 0.0)
        self.assertEqual(ev_empty.balance_score, 0.0)

        # Place straight path from (0,3) to (7,3)
        v_ns = self.lib.by_combo[(("N",), ("S",))][0]
        board = {(r, 3): PlacedTile((r, 3), v_ns) for r in range(8)}
        ev_connected = evaluator.evaluate(board)
        # Outlet (7, 3) is connected out of 3 positive outlets
        self.assertAlmostEqual(ev_connected.conn_progress, 1.0 / 3.0, places=5)
        self.assertGreater(ev_connected.balance_score, 0.0)

        # Dead branch test:
        # A tile at (0, 0) with outlet pointing North (boundary)
        v_dead = None
        for v in self.lib.variants:
            if v.outlet_sides == ("N",):
                v_dead = v
                break
        if v_dead is not None:
            dead_board = {(0, 0): PlacedTile((0, 0), v_dead)}
            self.assertTrue(evaluator.is_newly_placed_dead(dead_board, (0, 0)))

        # A tile at (0, 3) with outlet pointing South (empty cell) is not dead
        self.assertFalse(evaluator.is_newly_placed_dead(board, (0, 3)))

    def test_valuenet_forward_and_adam(self):
        vnet = ValueNet(in_dim=18, hidden=32, seed=42)
        dummy_x = np.ones(18, dtype=float)
        val, h = vnet.forward(dummy_x)
        self.assertTrue(np.isfinite(val))
        self.assertEqual(h.shape, (32,))

        grads = vnet.zero_grads()
        vnet.accumulate_value(grads, dummy_x, h, g_v=1.0)
        self.assertTrue(np.all(np.isfinite(grads["W1"])))
        self.assertTrue(np.all(np.isfinite(grads["w2"])))
        vnet.adam_step(grads, lr=1e-3)

    def test_curriculum_stages(self):
        cm = CurriculumManager(
            total_episodes=40,
            window=5,
            threshold=0.4,
            min_stage_episodes=5,
            max_stage_fraction=0.25,
            final_normal_fraction=0.4,
        )
        self.assertEqual(cm.stage, 0)

        # Stage 0 setup
        cm.setup_episode(self.env, self.lib)
        self.assertEqual(len(self.env.board), 2)
        mask = self.env.valid_action_mask()
        self.assertTrue(mask.any())

        # Advance to Stage 1
        for _ in range(5):
            cm.advance(success=True)
        self.assertEqual(cm.stage, 1)
        cm.setup_episode(self.env, self.lib)
        self.assertEqual(len(self.env.board), 1)

        # Advance to Stage 2
        for _ in range(5):
            cm.advance(success=True)
        self.assertEqual(cm.stage, 2)
        cm.setup_episode(self.env, self.lib)
        self.assertEqual(len(self.env.board), 0)

    def test_ppo_rollout_and_gae(self):
        rng = np.random.default_rng(123)
        pnet = PolicyNet(self.env.feature_dim, hidden=32, seed=123)
        vnet = ValueNet(in_dim=18, hidden=16, seed=123)

        self.env.reset()
        traj, cost = rollout_ppo(self.env, pnet, vnet, rng, global_dim=18)
        self.assertGreater(len(traj), 0)
        for step in traj:
            self.assertEqual(step.global_x.shape[0], 18)
            self.assertEqual(step.action_x.shape[1], 39)
            self.assertTrue(np.isfinite(step.log_prob))
            self.assertTrue(np.isfinite(step.value))

        adv, returns = compute_gae(traj, vnet, global_dim=18)
        self.assertEqual(len(adv), len(traj))
        self.assertEqual(len(returns), len(traj))
        self.assertTrue(np.all(np.isfinite(adv)))
        self.assertTrue(np.all(np.isfinite(returns)))

    def test_train_agent_ppo_smoke(self):
        # Run 4 episodes with rollouts_per_update=2, K_epochs=2
        pnet, vnet, best_board, best_reward, history, meta = train_agent_ppo(
            self.env,
            episodes=4,
            rollouts_per_update=2,
            K_epochs=2,
            mini_batch_size=8,
            hidden=32,
            seed=42,
            global_dim=18,
            log_every=1,
            verbose=False,
            use_curriculum=True,
            cur_window=2,
            cur_min_stage_episodes=2,
            cur_max_stage_fraction=0.5,
            cur_final_normal_fraction=0.0,
            probe_every=0,
        )
        self.assertIsNotNone(pnet)
        self.assertIsNotNone(vnet)
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 2)
        self.assertTrue(np.isfinite(best_reward))


if __name__ == "__main__":
    unittest.main()

