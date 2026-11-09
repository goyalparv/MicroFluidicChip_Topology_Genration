"""Unit tests for the improved CurriculumManager and training probe mechanics."""
from __future__ import annotations

import unittest
import numpy as np

from PhysicsAwareRL.config import DesignSpec, RewardWeights
from PhysicsAwareRL.curriculum import CurriculumManager
from PhysicsAwareRL.environment import TopologyEnv
from PhysicsAwareRL.library import Library
from PhysicsAwareRL.closure import sides_map
from PhysicsAwareRL.train import train_agent_ppo


class TestCurriculumPlan(unittest.TestCase):
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

    def test_tc1_progress_advancement(self):
        """T-C1: Good rate 0.3 stays; rate 0.5 advances after min_stage_episodes & window."""
        cm = CurriculumManager(
            total_episodes=2000,
            window=50,
            threshold=0.4,
            min_stage_episodes=100,
            max_stage_fraction=0.25,
            final_normal_fraction=0.4,
        )
        # Stream at 0.3 (fail every 3 out of 10)
        for ep in range(120):
            good = (ep % 10 < 3)
            reason = cm.advance({"success": good, "closed": True, "e_q": 0.05 if good else 0.5})
            self.assertIsNone(reason)
            self.assertEqual(cm.stage, 0)

        # Now stream at 0.5 (good every other episode)
        advanced = False
        for ep in range(60):
            good = (ep % 2 == 0)
            reason = cm.advance({"success": good, "closed": True, "e_q": 0.05 if good else 0.5})
            if reason:
                self.assertEqual(reason, "progress")
                self.assertEqual(cm.stage, 1)
                advanced = True
                break
        self.assertTrue(advanced)

    def test_tc2_time_advancement(self):
        """T-C2: All-bad stream advances at max_stage_fraction * total_episodes, reason 'time'."""
        cm = CurriculumManager(
            total_episodes=1000,
            window=50,
            threshold=0.4,
            min_stage_episodes=100,
            max_stage_fraction=0.25,
            final_normal_fraction=0.4,
        )
        # 0.25 * 1000 = 250 episodes
        for ep in range(249):
            reason = cm.advance({"success": False, "closed": True, "e_q": 0.9})
            self.assertIsNone(reason)
            self.assertEqual(cm.stage, 0)

        reason = cm.advance({"success": False, "closed": True, "e_q": 0.9})
        self.assertEqual(reason, "time")
        self.assertEqual(cm.stage, 1)

    def test_tc3_final_guarantee(self):
        """T-C3: Jump to stage 2 at total * (1 - final_normal_fraction) with reason 'final-guarantee'."""
        cm = CurriculumManager(
            total_episodes=1000,
            window=50,
            threshold=0.4,
            min_stage_episodes=100,
            max_stage_fraction=0.3,
            final_normal_fraction=0.4,
        )
        # Advance to stage 1 at ep 300 by timeout (0.3 * 1000 = 300)
        for _ in range(299):
            cm.advance({"success": False, "closed": True, "e_q": 0.9})
        self.assertEqual(cm.stage, 0)
        reason = cm.advance({"success": False, "closed": True, "e_q": 0.9})
        self.assertEqual(reason, "time")
        self.assertEqual(cm.stage, 1)

        # At ep 600 = 1000 * (1 - 0.4), final guarantee triggers
        for _ in range(300, 599):
            reason = cm.advance({"success": False, "closed": True, "e_q": 0.9})
            self.assertIsNone(reason)
            self.assertEqual(cm.stage, 1)

        reason = cm.advance({"success": False, "closed": True, "e_q": 0.9})
        self.assertEqual(reason, "final-guarantee")
        self.assertEqual(cm.stage, 2)

    def test_tc4_invalid_parameters_raise(self):
        """T-C4: Invalid parameters raise ValueError."""
        # 2 * 0.4 = 0.8 > 1 - 0.4 = 0.6
        with self.assertRaises(ValueError):
            CurriculumManager(
                total_episodes=2000,
                max_stage_fraction=0.4,
                final_normal_fraction=0.4,
            )
        # min_stage_episodes < window
        with self.assertRaises(ValueError):
            CurriculumManager(
                total_episodes=2000,
                window=50,
                min_stage_episodes=40,
            )
        # q_gate out of bounds
        with self.assertRaises(ValueError):
            CurriculumManager(total_episodes=2000, q_gate=0.001)

    def test_tc5_mixed_mode(self):
        """T-C5: Mixed mode draws stages according to schedule and is deterministic."""
        cm1 = CurriculumManager(total_episodes=2000, mode="mixed", seed=42)
        cm2 = CurriculumManager(total_episodes=2000, mode="mixed", seed=42)
        env = TopologyEnv(self.spec, self.library, enforce_closure=True)

        stages1 = [cm1.setup_episode(env, self.library) for _ in range(100)]
        stages2 = [cm2.setup_episode(env, self.library) for _ in range(100)]
        self.assertEqual(stages1, stages2)

    def test_tc6_hard_stage_tolerance(self):
        """T-C6: Hard stage scales tolerance and subsequent episodes restore it."""
        cm = CurriculumManager(
            total_episodes=2000,
            use_hard_stage=True,
            hard_tolerance_scale=0.5,
        )
        env = TopologyEnv(self.spec, self.library, enforce_closure=True)
        base_tol = env.weights.tolerance

        # Set to stage 3
        cm.stage = 3
        cm.setup_episode(env, self.library)
        self.assertAlmostEqual(env.weights.tolerance, base_tol * 0.5)

        # Set to stage 2 (non-hard)
        cm.stage = 2
        cm.setup_episode(env, self.library)
        self.assertAlmostEqual(env.weights.tolerance, base_tol)

    def test_tc7_setup_episode_hygiene(self):
        """T-C7: Stages 0, 1, 2 set exact tiles, clear mask cache, and step counts exclude preplaced."""
        cm = CurriculumManager(total_episodes=2000)
        env = TopologyEnv(self.spec, self.library, enforce_closure=True)

        # Stage 0: 2 tiles
        cm.stage = 0
        used = cm.setup_episode(env, self.library)
        self.assertEqual(used, 0)
        self.assertEqual(len(env.board), 2)
        self.assertEqual(env.steps, 0)
        self.assertEqual(len(env.preplaced), 2)
        self.assertIn((0, 3), env.board)
        self.assertIn((1, 3), env.board)

        # Stage 1: 1 tile
        cm.stage = 1
        used = cm.setup_episode(env, self.library)
        self.assertEqual(used, 1)
        self.assertEqual(len(env.board), 1)
        self.assertEqual(env.steps, 0)
        self.assertEqual(len(env.preplaced), 1)
        self.assertIn((0, 3), env.board)

        # Stage 2: empty
        cm.stage = 2
        used = cm.setup_episode(env, self.library)
        self.assertEqual(used, 2)
        self.assertEqual(len(env.board), 0)
        self.assertEqual(env.steps, 0)
        self.assertEqual(len(env.preplaced), 0)

    def test_tc8_probe_read_only(self):
        """T-C8: Probes do not affect curriculum state or training buffer."""
        env = TopologyEnv(self.spec, self.library, enforce_closure=True)
        # Smoke run with probes
        pnet, vnet, board, rew, hist, meta = train_agent_ppo(
            env,
            episodes=4,
            rollouts_per_update=2,
            K_epochs=2,
            mini_batch_size=8,
            hidden=32,
            seed=42,
            probe_every=1,
            probe_episodes=2,
            verbose=False,
            cur_window=2,
            cur_min_stage_episodes=2,
            cur_max_stage_fraction=0.5,
            cur_final_normal_fraction=0.0,
        )
        self.assertIn("probe_success_rate", hist[0])
        self.assertIn("probe_mean_e_q", hist[0])

    def test_tc9_no_curriculum(self):
        """T-C9: curriculum_mode='off' starts every episode from empty board."""
        cm = CurriculumManager(total_episodes=2000, mode="off")
        env = TopologyEnv(self.spec, self.library, enforce_closure=True)
        st = cm.setup_episode(env, self.library)
        self.assertEqual(st, 2)
        self.assertEqual(len(env.board), 0)

    def test_tc10_preplaced_states_viable(self):
        """T-C10: All preplaced start states can be completed to a closed board."""
        cm = CurriculumManager(total_episodes=2000)
        env = TopologyEnv(self.spec, self.library, enforce_closure=True)
        for st in [0, 1, 2]:
            cm.stage = st
            cm.setup_episode(env, self.library)
            plan = env.oracle.plan(sides_map(env.board))
            self.assertIsNotNone(plan, f"Stage {st} preplaced state cannot be completed by oracle")


if __name__ == "__main__":
    unittest.main()
