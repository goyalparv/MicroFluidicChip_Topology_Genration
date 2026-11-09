# Physics-Aware Reinforcement Learning for Microfluidic Topology Synthesis

An automated end-to-end computational framework that synthesizes discrete tile-based microfluidic chips to achieve precise target outlet flow ratios under Stokes flow. The framework combines a **verified nodal hydraulic circuit solver**, **finite-element COMSOL Multiphysics calibration**, and **physics-aware Proximal Policy Optimization (PPO)** with strict topological closure guarantees.

---

## 🔬 System Overview & Pipeline Architecture

The project is structured into three self-contained, interconnected stages:

```
+---------------------------------------------------------------------------------------+
|  PHASE 1: Solver Verification (solver_verification)                                   |
|  • 17x17 Nodal Circuit Resistor Network Solver (hydraulic_solver.py)                  |
|  • Automated Monte Carlo benchmarking against 2D Navier-Stokes FEM (COMSOL 6.x)       |
|  • Validates < 1.5% hydrodynamic relative error across arbitrary random topologies   |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|  PHASE 2: Tile Library Generation (PhysicsAwareRL/comsol)                              |
|  • Discrete tile geometries (25x25 rasters, 100 µm channel width) across 50 combos    |
|  • Serpentine meander paths with varying resistance ratios                            |
|  • Parametric COMSOL Java API automation (FlowJob.java & measure.py)                  |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|  PHASE 3: Physics-Aware RL Synthesis & Validation (PhysicsAwareRL)                    |
|  • PPO Agent with Breadth-First-Search topological closure oracle                     |
|  • Multi-objective structural presets: 'compact', 'tree', and 'low-pressure'         |
|  • 20-Target independent batch automation (run_batch.py)                              |
|  • Automated 4 standalone diagnostic learning curves                                  |
|  • Direct post-synthesis COMSOL validation (validate_design.py)                       |
+---------------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```text
Microfluidic-Physics-Aware-RL/
├── .gitignore                      # Prevents committing caches, .class, .mph, or heavy data
├── LICENSE                         # MIT Open-Source License
├── README.md                       # Master documentation
├── requirements.txt                # Python dependencies
│
├── docs/                           # Technical implementation & theory documents
│   ├── fix_plan.md                 # Detailed structural objective & diversity plan
│   └── Closed-Topology Enforcement Implementation Plan.md
│
├── solver_verification/            # PHASE 1: Standalone solver validation against COMSOL
│   ├── ChipProcessor.java          # COMSOL Java driver building 2D geometry & solving spf
│   ├── auto_pipeline.py            # Master Monte Carlo comparison across N random chips
│   ├── hydraulic_solver.py         # Standalone 17x17 nodal hydraulic circuit solver
│   ├── random_chip.py              # Generates connected 8x8 tile chip network topologies
│   ├── solver_wrapper.py           # Interface executing hydraulic circuit simulation
│   ├── README.md                   # Detailed solver verification guide
│   └── __init__.py
│
└── PhysicsAwareRL/                 # PHASES 2 & 3: Tile Library, RL Synthesis, Validation
    ├── comsol/                     # Phase 2: COMSOL FEM tile characterization
    │   ├── FlowJob.java            # COMSOL Java worker evaluating tile flow resistance
    │   ├── measure.py              # Automated tile raster measurement pipeline
    │   ├── runner.py               # Subprocess runner & flow problem abstractions
    │   └── __init__.py
    ├── tests/                      # Automated test suite (all 13 unittests)
    │   ├── test_fix_plan.py        # Core regression test suite (T1 - T13)
    │   ├── test_closure.py         # Closure oracle tests
    │   ├── test_curriculum_plan.py # Curriculum learning tests
    │   ├── test_improvements.py    # RL policy advantage normalization tests
    │   ├── test_pipeline.py        # Pipeline integration tests
    │   └── __init__.py
    ├── tools/                      # Analytical reference & comparison tools
    │   ├── reference_optimum.py    # Analytical DP solver for monotone flow routes
    │   ├── compare_designs.py      # Design & topology Jaccard distance calculator
    │   └── __init__.py
    ├── agent.py                    # Neural network policy & value networks
    ├── closure.py                  # BFS-based closure oracle guaranteeing leak-free chips
    ├── config.py                   # Hyperparameters, physical constants, & reward presets
    ├── curriculum.py               # Staged curriculum training manager
    ├── environment.py              # Microfluidic grid RL environment (8x8 tiles)
    ├── hydraulic_solver.py         # High-performance sparse nodal solver (solve_network)
    ├── library.py                  # Tile library loader & serpentine path generator
    ├── metrics.py                  # Topology hash (topo_id), design_id, cycles & merge count
    ├── reward.py                   # Relative flow error EQ + structural penalty evaluator
    ├── run_batch.py                # 20-target automated sequential batch runner
    ├── run_stage1.py               # Primary CLI entry point for Stage 1 RL synthesis
    ├── train.py                    # Physics-aware GAE-PPO training algorithm
    ├── validate_design.py          # Post-RL COMSOL validation (raster assembly & FEM solve)
    ├── visualize_stage1.py         # Standalone learning curves & chip layout visualizer
    ├── README.md
    └── __init__.py
```

---

## ⚙️ Installation & Prerequisites

### 1. Python Environment
Python 3.10 or 3.11 is recommended. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Java Development Kit (JDK) & COMSOL (For FEM Validation)
- **Java JDK**: JDK 11 or higher (`javac` and `java` available on PATH).
- **COMSOL Multiphysics 6.x**: Required for live FEM simulations. By default, the scripts reference the COMSOL Java API plugins:
  `C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins`
- **Mock/Dry-Run Support**: All pipelines include a `--backend mock` or `--validate-backend mock` option so that the entire codebase can be verified on machines without COMSOL.

---

## 🚀 Execution Guide

### Phase 1: Hydraulic Circuit Solver Verification
Verify that the fast 1D nodal circuit solver matches COMSOL Multiphysics 2D laminar Navier-Stokes simulations across random chip topologies:

```bash
# A. Quick dry-run test (no COMSOL required):
python -m solver_verification.auto_pipeline --backend mock --runs 5 --output_dir results/test_solver

# B. Real COMSOL Multiphysics verification (on COMSOL workstation):
python -m solver_verification.auto_pipeline --backend comsol --runs 10 --comsol_plugins "C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins" --output_dir verification_results
```

---

### Phase 2: Microchannel Tile Library Generation
Measure discrete tile geometries with serpentine channel paths to build the physical resistance lookup table (`library.csv`):

```bash
python -m PhysicsAwareRL.comsol.measure --comsol-plugins "C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins" --out-dir results/library
```
*(If no custom library is provided, the RL pipeline automatically synthesizes a calibrated unitless baseline library of 640 tile variants across 50 active port configurations).*

---

### Phase 3: Physics-Aware RL Topology Synthesis & Automated Validation

#### 1. Single Target Synthesis Run
Synthesize an $8 \times 8$ microfluidic chip for specific outlet flow ratios (e.g., $Q_1 : Q_2 : Q_3 = 0.50 : 0.30 : 0.20$):

```bash
python -m PhysicsAwareRL.run_stage1 \
    --targets 0.5 0.3 0.2 \
    --episodes 2000 \
    --batch 16 \
    --objective compact \
    --out results/run_50_30_20
```

**Key CLI Options**:
- `--objective`: Choose structural design preset (`compact`, `tree`, `low-pressure`, `legacy`).
- `--validate-backend`: Validation mode after RL completes (`comsol`, `mock`, `raster`, or `none`). Default is `comsol`.
- `--comsol-plugins`: Path to COMSOL plugins directory.
- `--no-validate`: Skip automated COMSOL validation (e.g. for rapid RL training only).

#### 2. Automated 20-Target Batch Run
Execute 20 independent target specifications sequentially from scratch. Results for each run are stored in independent, self-contained directories:

```bash
# Run all 20 targets sequentially:
python -m PhysicsAwareRL.run_batch --episodes 2000 --batch 16 --out-dir results/batch_20_targets

# Run specific target indices (e.g., Target 1, 2, and 3):
python -m PhysicsAwareRL.run_batch --indices 1 2 3 --episodes 2000 --batch 16 --out-dir results/batch_test
```

---

## 📊 Outputs & Artifacts

For each run, the pipeline writes a comprehensive set of diagnostic and design outputs directly into its destination directory:

| Output File | Format | Description |
| :--- | :--- | :--- |
| **`design.json`** | JSON | Full specification of the best synthesized chip layout (tiles, cells, variants, realized flows, port degrees). |
| **`chip_layout.png`** | PNG | High-resolution 2D visual schematic of the synthesized chip showing serpentine channels, inlet, and outlets. |
| **`rl_rewards.png`** | PNG | **Standalone Learning Curve 1**: Mean step reward, mean total score (reward − structural penalty), and best score. |
| **`rl_flow_error.png`** | PNG | **Standalone Learning Curve 2**: Hydraulic flow error $E_Q$ convergence relative to the 5% tolerance threshold. |
| **`rl_tile_count.png`** | PNG | **Standalone Learning Curve 3**: Topology compactness and placed tile count vs. the $n_{\text{min}} = 17$ minimal skeleton. |
| **`rl_policy_entropy.png`** | PNG | **Standalone Learning Curve 4**: Policy exploration health ($H_{\text{norm}}$) and action collapse monitoring. |
| **`archive.json`** | JSON | Top-$K$ distinct topologies discovered during training (sorted deterministically by score). |
| **`eval_summary.json`** | JSON | Post-training evaluation statistics over fixed test trials. |
| **`comsol_check/`** | Directory | **Automated COMSOL Validation**: Contains `comsol_validation.json`, `comsol_validation.csv`, `chip_mask.npy` ($200 \times 200$ raster), and `comsol_validation.png`. |

---

## 🧪 Automated Unit Testing

Run the full suite of regression unit tests to verify algorithmic correctness (closure oracle, curriculum manager, entropy scheduling, DP reference optimum, reward presets):

```bash
python -m unittest discover PhysicsAwareRL/tests
```
or run the main test plan directly:
```bash
python -m unittest PhysicsAwareRL.tests.test_fix_plan
```
