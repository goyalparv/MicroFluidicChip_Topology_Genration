"""Unit tests for PhysicsAwareRL_v3 (stdlib unittest + numpy only)."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from PhysicsAwareRL_v3.config_v2 import DesignSpec, RewardWeights, normalize_targets
from PhysicsAwareRL_v3.environment_v2 import TopologyEnv
from PhysicsAwareRL_v3.hydraulic_solver_v2 import (
    center_node,
    outlet_port_nodes,
    port_node,
    solve_network_v2,
)
from PhysicsAwareRL_v3.library_v2 import (
    Library,
    PlacedTile,
    TileVariant,
    enumerate_side_combos,
)
from PhysicsAwareRL_v3.reward_v2 import Evaluator


class TestConfig(unittest.TestCase):
    def test_default_spec_validation(self):
        spec = DesignSpec(outlet_cells=[(7, 0), (7, 3), (7, 7)], target_flows=[5e-5, 3e-5, 2e-5])
        spec.validate()
        self.assertEqual(spec.inlet_cell, (0, 3))
        self.assertEqual(len(spec.outlet_cells), 3)

    def test_invalid_inlet_rejected(self):
        spec = DesignSpec(inlet_cell=(1, 3))
        with self.assertRaises(ValueError):
            spec.validate()

    def test_normalize_targets(self):
        scaled = normalize_targets([1.0, 2.0, 1.0], 1e-4)
        self.assertAlmostEqual(sum(scaled), 1e-4)
        self.assertAlmostEqual(scaled[0], 2.5e-5)
        self.assertAlmostEqual(scaled[1], 5.0e-5)


class TestLibrary(unittest.TestCase):
    def test_side_combos_count(self):
        combos = enumerate_side_combos()
        self.assertEqual(len(combos), 50)

    def test_initial_library_generation(self):
        lib = Library.generate_initial_library()
        self.assertEqual(len(lib), 640)
        self.assertEqual(len(lib.by_combo), 50)
        self.assertGreater(lib.max_r, lib.min_r)

    def test_csv_roundtrip(self):
        lib = Library.generate_initial_library(k_1in1out=2, k_1in2out=2, k_1in3out=2, k_2in1out=2, k_2in2out=2, k_3in1out=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_lib.csv")
            lib.to_csv(csv_path)
            loaded = Library.from_csv(csv_path)
            self.assertEqual(len(loaded), len(lib))
            v0 = lib[0]
            v0_loaded = loaded[0]
            self.assertEqual(v0.variant_id, v0_loaded.variant_id)
            self.assertEqual(v0.inlet_sides, v0_loaded.inlet_sides)
            self.assertEqual(v0.outlet_sides, v0_loaded.outlet_sides)
            for s in v0.R_per_side:
                self.assertAlmostEqual(v0.R_per_side[s], v0_loaded.R_per_side[s], places=4)


class TestHydraulicSolver(unittest.TestCase):
    def setUp(self):
        self.spec = DesignSpec(
            outlet_cells=[(7, 0), (7, 3), (7, 7)],
            target_flows=[5e-5, 3e-5, 2e-5],
            inlet_flow=1e-4,
        )
        self.lib = Library.generate_initial_library()

    def test_node_mappings(self):
        # Inlet cell (0, 3) North port -> (0, 7) -> node 7
        self.assertEqual(port_node(0, 3, "N"), 7)
        # Center of (2, 6) -> (5, 13) -> 5*17 + 13 = 98
        self.assertEqual(center_node(2, 6), 98)
        # Port nodes for outlets on bottom row
        out_ports = outlet_port_nodes(self.spec)
        self.assertEqual(out_ports, [273, 279, 287])

    def test_series_chain_conservation(self):
        # Straight vertical chain from (0,3) down to (7,3)
        v_ns = self.lib.by_combo[(("N",), ("S",))][0]
        board = {(r, 3): PlacedTile((r, 3), v_ns) for r in range(8)}
        spec = DesignSpec(outlet_cells=[(7, 3)], target_flows=[1e-4], inlet_flow=1e-4)
        sr = solve_network_v2(board, spec)

        self.assertTrue(sr.ok)
        self.assertFalse(sr.stranded)
        self.assertAlmostEqual(float(sr.outlet_flows[0]), 1e-4, places=9)
        self.assertLess(sr.mass_residual, 1e-12)

    def test_stranded_network(self):
        # Place only tile at (0,3) with outlet to West, not reaching row 7
        v_nw = self.lib.by_combo[(("N",), ("W",))][0]
        board = {(0, 3): PlacedTile((0, 3), v_nw)}
        sr = solve_network_v2(board, self.spec)

        self.assertTrue(sr.ok)
        self.assertTrue(sr.stranded)
        self.assertEqual(float(np.sum(sr.outlet_flows)), 0.0)
        self.assertFalse(sr.connected_outlets.any())

    def test_parallel_branch_split(self):
        # Flow splits into two branches
        v_split = self.lib.by_combo[(("N",), ("E", "S"))][0]
        # Custom symmetric split
        sym_split = TileVariant(
            variant_id="test_split",
            inlet_sides=("N",),
            outlet_sides=("E", "S"),
            R_per_side={"N": 1.0, "E": 2.0, "S": 2.0},
        )
        v_s = self.lib.by_combo[(("N",), ("S",))][0]
        v_w = self.lib.by_combo[(("W",), ("S",))][0]

        # Cell (6, 3) splits to South (7,3) and East (6,4) -> South (7,4)
        board = {
            (r, 3): PlacedTile((r, 3), v_s) for r in range(6)
        }
        board[(6, 3)] = PlacedTile((6, 3), sym_split)
        board[(7, 3)] = PlacedTile((7, 3), v_s)
        board[(6, 4)] = PlacedTile((6, 4), v_w)
        board[(7, 4)] = PlacedTile((7, 4), v_s)

        spec = DesignSpec(outlet_cells=[(7, 3), (7, 4)], target_flows=[5e-5, 5e-5], inlet_flow=1e-4)
        sr = solve_network_v2(board, spec)
        self.assertTrue(sr.ok)
        self.assertAlmostEqual(float(np.sum(sr.outlet_flows)), 1e-4, places=9)
        # Branch 1 total R = 2 + 1 + 1 + 1 = 5. Branch 2 total R = 2 + 1 + 1 + 1 + 1 + 1 = 7.
        # Flows divide inversely as resistances: 7/12 (58.33%) and 5/12 (41.67%).
        self.assertAlmostEqual(sr.outlet_flows[0], 1e-4 * (7.0 / 12.0), places=9)
        self.assertAlmostEqual(sr.outlet_flows[1], 1e-4 * (5.0 / 12.0), places=9)


class TestEnvironmentAndReward(unittest.TestCase):
    def setUp(self):
        self.spec = DesignSpec(
            outlet_cells=[(7, 0), (7, 3), (7, 7)],
            target_flows=[5e-5, 3e-5, 2e-5],
            inlet_flow=1e-4,
        )
        self.lib = Library.generate_initial_library()
        self.env = TopologyEnv(self.spec, self.lib)

    def test_initial_action_mask(self):
        mask = self.env.valid_action_mask()
        self.assertFalse(mask[self.env.stop_action])
        valid_indices = np.flatnonzero(mask)
        self.assertGreater(len(valid_indices), 0)

        # All initial valid actions must be in inlet_cell (0, 3)
        for act in valid_indices:
            cell_idx, vi = divmod(act, self.env.V)
            self.assertEqual(cell_idx, 0 * 8 + 3)
            self.assertIn("N", self.lib[vi].inlet_sides)

    def test_features_consistency(self):
        x, ids = self.env.features()
        self.assertEqual(len(x), len(ids))
        self.assertEqual(x.shape[1], self.env.feature_dim)
        self.assertTrue(np.all(np.isfinite(x)))

    def test_stop_action_masked_until_all_outlets_connected(self):
        mask = self.env.valid_action_mask()
        self.assertFalse(mask[self.env.stop_action])

    def test_evaluator_exact_match(self):
        v_ns = self.lib.by_combo[(("N",), ("S",))][0]
        board = {(r, 3): PlacedTile((r, 3), v_ns) for r in range(8)}
        spec = DesignSpec(outlet_cells=[(7, 3)], target_flows=[1e-4], inlet_flow=1e-4)
        ev = Evaluator(spec, self.lib, RewardWeights()).evaluate(board)
        self.assertAlmostEqual(ev.e_q, 0.0, places=6)
        self.assertTrue(ev.success)


if __name__ == "__main__":
    unittest.main()
