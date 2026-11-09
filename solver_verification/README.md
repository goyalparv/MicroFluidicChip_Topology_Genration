# Solver Verification Pipeline: Hydraulic Solver vs COMSOL Multiphysics

This directory provides the automated validation pipeline to verify the hydraulic circuit solver directly against finite-element COMSOL Multiphysics 2D laminar Navier-Stokes simulations.

---

## 1. Key Architectural & Physical Alignments

1. **Grid Architecture**:
   - $8 \times 8$ tile coarse grid (64 tiles).
   - Corresponding $17 \times 17$ fine-grid hydraulic circuit solver (289 nodes + 1 virtual inlet node = 290 total).
   - Fixed Inlet at cell `(0, 3)` on the **North** face.
   - Fixed 3 Outlets at cells `(7, 0)`, `(7, 3)`, and `(7, 7)` on the **South** face.
   - No middle split: outlet 2 is connected directly to cell `(7, 3)` South face.

2. **Input Flow Rate**:
   - Preserved consistently throughout the project at **$10^{-4}\,\text{m/s}$** (`0.0001 m/s` or $10^{-4}$ in flow units), matching `DEFAULT_INLET_FLOW` in `config.py` and the COMSOL boundary condition `U0in = 0.0001[m/s]`.

3. **Physical Resistance Scale**:
   - Microchannel width: $W = 100\,\mu\text{m}$.
   - Depth/thickness: $H = 100\,\mu\text{m}$.
   - Water properties: $\mu = 1.01 \times 10^{-3}\,\text{Pa}\cdot\text{s}$, $\rho = 998.2\,\text{kg/m}^3$.
   - Leg resistances: $R \sim 1.97 \times 10^{11} - 9.83 \times 10^{11}\,\text{Pa}\cdot\text{s}/\text{m}^3$ computed from:
     $$R = r_0 \cdot L, \quad r_0 = \frac{12 \mu}{W H^3 (1 - 0.63 H/W)} \approx 3.28 \times 10^8\,\frac{\text{Pa}\cdot\text{s}}{\text{m}^3 \cdot \mu\text{m}}$$
     This exactly matches the physical resistance magnitude of the tile library.

---

## 2. Directory Structure

```
solver_verification/
├── hydraulic_solver.py    # Standalone 17x17 circuit solver (zero external folder dependencies)
├── random_chip.py         # Generates connected 8x8 tile chip layouts connecting (0,3) to (7,0),(7,3),(7,7)
├── solver_wrapper.py      # Standalone runner invoking solve_network on the chip
├── ChipProcessor.java     # COMSOL 6.x Multiphysics driver building 2D geometry and solving Laminar Flow (spf)
├── auto_pipeline.py       # Master automation pipeline comparing Python vs COMSOL across N runs
└── README.md              # Documentation and execution instructions
```

> [!NOTE]
> This folder is **100% self-contained and standalone**. You can copy just the `solver_verification` folder to any workstation or COMSOL machine and run it directly without needing any other folders.

---

## 3. How to Run

From the repository root:

### A. On Your COMSOL Workstation (Real COMSOL Verification)
```bash
python -m solver_verification.auto_pipeline --backend comsol --runs 10 --comsol_plugins "C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins" --output_dir verification_results
```
* Compiles `ChipProcessor.java` once at startup.
* For each iteration:
  1. Generates a new random connected chip layout in `Run_i/chip_layout.txt`.
  2. Runs `hydraulic_solver.py` and captures outlet flows $[Q_1, Q_2, Q_3]$.
  3. Executes COMSOL via Java and extracts the true finite-element outlet velocities $[v_1, v_2, v_3]$.
  4. Calculates $L_1$ error, $L_\infty$ error, and relative percentage error.
* Aggregates all errors into `verification_results/aggregate_errors.csv` and generates `error_plot.png`.

### B. Quick Test Mode (Any Machine Without COMSOL)
```bash
python -m solver_verification.auto_pipeline --backend mock --runs 5 --output_dir verification_results
```
Runs the entire pipeline using synthetic finite-element Stokes flow calculations to verify script execution, CSV logging, and plot generation.


