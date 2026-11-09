# PhysicsAwareRL: Tile-Based Microfluidic RL Topology Synthesis

This folder contains the complete, standalone implementation of **PhysicsAwareRL**, implementing all structural objective enhancements, learning-signal improvements, top-$K$ topology diversity archiving, and evaluation tools strictly according to `docs/fix_plan.md`.

---

## 1. Quick Start Commands

Run all commands from the repository root:

### 1.1 Run Stage 1 RL Training (Compact Objective with Batch Advantage Normalization)
```bash
python -m PhysicsAwareRL.run_stage1 --targets 0.5 0.3 0.2 --episodes 2000 --batch 8 --objective compact --out results/stage1
```

### 1.2 Objective Presets (`--objective`)
- `--objective compact` (Default): $\lambda_{\text{extra}} = 0.30$, $\lambda_{\text{pressure}} = 0.15$, $\lambda_{\text{loops}} = 0.0$. Penalizes excessive filler tiles beyond $n_{\text{min}}$ (17) and excess inlet pressure over baseline.
- `--objective low-pressure`: $\lambda_{\text{extra}} = 0.15$, $\lambda_{\text{pressure}} = 0.30$, $\lambda_{\text{loops}} = 0.0$. Prioritizes minimum hydraulic flow resistance and inlet pressure.
- `--objective tree`: $\lambda_{\text{extra}} = 0.30$, $\lambda_{\text{pressure}} = 0.15$, $\lambda_{\text{loops}} = 0.10$. Enforces cycle-free tree topologies.
- `--objective legacy`: $\lambda_{\text{extra}} = 0.0$, $\lambda_{\text{pressure}} = 0.0$, $\lambda_{\text{loops}} = 0.0$. Baseline objective without structural penalties.

### 1.3 Learning & Architecture Flags
- `--adv-norm batch` (Default): Batch-level advantage normalization across all rollout episodes in an update, preventing identical push from poor vs. good trajectories. Use `--adv-norm episode` for legacy mode.
- `--entropy 0.03 --entropy-final 0.003`: Linear annealing of exploration entropy coefficient from start to final update.
- `--archive-k 5` (Default 5): Keeps the top $K$ distinct topologies ranked deterministically by total ordering key `(-score, n_tiles, p_in, design_id)`.
- `--eval-episodes 200` (Default 200): Fixed post-training evaluation (200 stochastic episodes + 1 greedy episode) on dedicated RNG `seed + 1000`.
- `--tile-potential dense` (Optional): Dense iterative-deepening shaping potential for tile count.
- `--lam-robust 0.1 --robust-eps 0.10 --robust-samples 8` (Optional): Channel perturbation tolerance objective.

### 1.4 Run the Reference-Optimum Tool (Ground Truth for 17-Tile Skeleton)
```bash
python -m PhysicsAwareRL.tools.reference_optimum
```

### 1.5 Run Design Similarity / Diversity Comparison Tool
```bash
python -m PhysicsAwareRL.tools.compare_designs results/run1 results/run2
```

### 1.6 Run Regression Test Suite
```bash
python -m unittest PhysicsAwareRL.tests.test_fix_plan
```
```bash
python -m PhysicsAwareRL_v3.tools.reference_optimum --out results/reference_optimum.csv
```
This enumerates the 12 monotone staircase routes of the minimum 17-tile skeleton, finds the analytical best variant combination across the library, and automatically cross-checks against `solve_network_v2` to within $10^{-9}$ numerical tolerance.

### 1.5 Compare Synthesized Designs across Runs ($J_{\text{seed}}$ and $J_{\text{target}}$)
```bash
python -m PhysicsAwareRL_v3.tools.compare_designs results/run1 results/run2
```
Computes pairwise Jaccard similarities on occupied grid cells to distinguish seed luck ($J_{\text{seed}}$) from genuine target responsiveness ($J_{\text{target}}$).

### 1.6 Run Full Unit Test Suite (T1 to T13)
```bash
python -m unittest PhysicsAwareRL_v3.tests.test_fix_plan -v
```

### 1.7 Run Automated Batch Training Across 20 Target Ratios
```bash
python -m PhysicsAwareRL_v3.run_batch --episodes 2000 --batch 16 --out-dir results/batch_20_targets
```
Runs 20 distinct target ratio specifications sequentially from the ground up, automatically saving `chip_layout.png`, `training_curves.png`, `design.json`, `batch_summary.csv`, and `batch_overview.png`.

---

## 2. Directory Structure

```
PhysicsAwareRL_v3/
├── config_v2.py              # DesignSpec, RewardWeights with structural budgets & presets
├── closure_v2.py             # Closed-topology enforcement, port analyzer, bounded DFS closure oracle
├── metrics_v2.py             # Topo & design signatures, count_cycles, count_merges, jaccard similarity
├── library_v2.py             # 50 side-assignment combos, TileVariant, Library class (640 variants)
├── hydraulic_solver_v2.py    # 17x17 fine-grid Stokes circuit solver (290 nodes, exact KCL)
├── environment_v2.py         # TopologyEnv: sequential tile placement, n_min, p_ref, terminal struct_cost
├── reward_v2.py              # Evaluator & Evaluation: struct_parts, robust error, score property
├── agent_v2.py               # Physics-informed GAE-PPO (PolicyNet + ValueNet) & REINFORCE in pure NumPy
├── curriculum_v2.py          # Progressive staged curriculum manager
├── train_v2.py               # GAE-PPO with batch advantage normalization, entropy logging, Top-K archive
├── run_stage1_v2.py          # Main CLI entrypoint saving config.json, archive.json, design.json, eval_summary.json
├── visualize_stage1.py       # High-resolution chip layout generator and ASCII visualizer
│
├── tools/
│   ├── reference_optimum.py  # DP solver for 17-tile skeleton across 12 monotone routes with 1e-9 solver cross-check
│   └── compare_designs.py    # Multi-run design comparator computing J_seed and J_target
│
└── tests/
    ├── test_fix_plan.py      # Complete T1-T13 test suite covering all fix_plan.md requirements
    ├── test_closure_v2.py    # 11 unit tests for closed-topology invariants and oracle soundness
    ├── test_improvements_v2.py# Unit tests for GAE, PPO, ValueNet, curriculum, 39-dim features
    └── test_pipeline_v2.py   # Full pipeline unit test suite
```

---

## 3. Output Files Produced by Training Runs

Every run with `run_stage1_v2.py` outputs into `--out`:
1. `config.json`: Full CLI arguments, random seed, Git commit hash, timestamp, $n_{\text{min}}$, and $p_{\text{ref}}$.
2. `design.json`: The rank-1 synthesized design with `topo_id`, `design_id`, `score`, `eligible`, `n_min`, `p_ref`, full tile list, and metrics.
3. `archive.json`: Top-$K$ distinct topologies found during training/evaluation ranked by total ordering key.
4. `design_rank2.json` ... `design_rankK.json`: Full specifications of subsequent distinct top archive designs.
5. `eval_summary.json`: Detailed statistical summary over `--eval-episodes` (success rate, mean tiles, loop count, pressure ratio, distinct topologies, top-3 topology counts, and greedy rollout metrics).
6. `history.json`: Per-batch training diagnostics including `entropy_mean`, `entropy_norm_mean`, `maxprob_mean`, `mean_score`, `mean_struct_cost`, and structural parts (`extra_n`, `press_n`, `loops_n`).
7. `chip_layout.png`: High-resolution visual render of the rank-1 synthesized microfluidic chip.

---

## 4. Key Metrics Reference

- `score`: $E_{\text{score}} = \text{reward} - \text{struct\_cost}$. Main ranking objective.
- `n_min`: Minimum number of tiles required to close the chip specification (17 for default 8x8 1-in-3-out spec).
- `p_ref`: Inlet pressure of the minimum 17-tile reference design solved with baseline resistance variants.
- `extra`: Number of tiles placed beyond $n_{\text{min}}$ (`max(0, n_tiles - n_min)`).
- `loops`: Cycle rank ($E - V + C$) of the placed tile network graph.
- `merges`: Number of tiles where 2 or more incoming fluid flows meet.
- `p_ratio`: Realized inlet pressure divided by $p_{\text{ref}}$.
- `struct_cost`: $\lambda\_{\text{extra}} \cdot \text{extra\_n} + \lambda\_{\text{pressure}} \cdot \text{press\_n} + \lambda\_{\text{loops}} \cdot \text{loops\_n}$. Applied exactly once at episode termination.
- `topo_id`: 12-character SHA-1 hash of the graph topology (cells and port connection directions, invariant to resistance variant choices and dictionary order).
- `design_id`: 12-character SHA-1 hash of the exact physical chip design (including specific resistance variants).
