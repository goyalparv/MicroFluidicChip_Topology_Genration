"""Validate an RL-synthesized design.json against COMSOL (or offline checks).

What it does
------------
1. Reads ``design.json`` written by ``run_stage1_v2`` (spec, tiles, per-leg R, realized flows).
2. Rebuilds the *exact* tile geometry of every placed tile from the library recipe
   (``get_tile_leg_paths`` -> 25x25 raster), i.e. the same shapes whose resistances were
   measured in COMSOL when the library was generated. Different tiles keep their different
   meander depths; nothing is assumed equal.
3. Stacks the tile rasters on the 8x8 grid into one (8*25) x (8*25) chip raster. Neighbouring
   tiles join pixel-to-pixel at the port pixels, so channels are continuous.
4. Checks the assembled raster (no stray contacts, tile joints only where legs meet).
5. Solves the chip with the selected backend and compares outlet flows with the targets and
   with the hydraulic solver.

Backends
--------
``comsol``  real COMSOL run through the existing ``ComsolJavaRunner`` / ``FlowJob.java``.
``raster``  offline pixel-network model (no COMSOL). Resistance per pixel step is uniform, so it
            is an independent check of the geometry assembly. It is NOT a physics reference.
``mock``    solver flows with a small deterministic perturbation, to dry-run the pipeline.
``none``    build + integrity checks only (same as ``--dry-run``).

Run (from the repository root)::

    python -m PhysicsAwareRL_v2.validate_design --design results/stage1_v2/design.json \\
        --backend comsol --comsol-plugins "C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins"

It also provides ``validate_rl_design(...)`` with the signature that ``auto_pipeline_v2 --design``
already calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # package layout (python -m PhysicsAwareRL_v2.validate_design)
    from .config_v2 import DesignSpec
    from .hydraulic_solver_v2 import solve_network_v2
    from .library_v2 import Library, PlacedTile, TileVariant
    from .comsol.measure_v2 import generate_tile_raster
    from .comsol.runner import ComsolJavaRunner, FlowProblem, FlowResult, Port
except ImportError:  # run from repo root as a script
    from PhysicsAwareRL_v2.config_v2 import DesignSpec
    from PhysicsAwareRL_v2.hydraulic_solver_v2 import solve_network_v2
    from PhysicsAwareRL_v2.library_v2 import Library, PlacedTile, TileVariant
    from PhysicsAwareRL_v2.comsol.measure_v2 import generate_tile_raster
    from PhysicsAwareRL_v2.comsol.runner import ComsolJavaRunner, FlowProblem, FlowResult, Port

try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

DEFAULT_COMSOL_PLUGINS = "C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins"
SIDE_LEN = 13                    # 13x13 grid-line tile -> 25x25 raster (same as measure_v2)
DENOM_FLOOR_FRACTION = 0.01      # same floor the Evaluator uses
ERROR_CLIP = 3.0                 # same clip the Evaluator uses
_VID_RE = re.compile(r"_v(\d+)$")
# Library resistances below this look like the unit-less baseline library, not COMSOL values.
BASELINE_R_THRESHOLD = 1e3


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_design(path: str) -> Tuple[DesignSpec, dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    spec = DesignSpec.from_dict(data["spec"])
    if not spec.target_flows:
        raise ValueError("design.json spec has no target_flows")
    spec.validate()
    if not data.get("tiles"):
        raise ValueError("design.json contains no tiles")
    return spec, data


def _sorted_tuple(x: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(x))


def combo_sizes(lib: Library) -> Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], int]:
    """Number of resistance variants per (inlet, outlet) combo (= k_total used at generation)."""
    sizes: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], int] = {}
    for v in lib.variants:
        key = (_sorted_tuple(v.inlet_sides), _sorted_tuple(v.outlet_sides))
        sizes[key] = sizes.get(key, 0) + 1
    return sizes


def variant_geometry_index(tile: dict, sizes: Optional[dict]) -> Tuple[int, int]:
    """Return (variant_idx, k_total) needed by ``get_tile_leg_paths`` for one tile entry.

    Preferred: fields ``variant_index`` and ``k_total`` stored in the tile entry.
    Fallback: parse the ``_vNN`` suffix of ``variant_id`` (IDs are 1-based) and count the
    variants of the same combo in the library CSV.
    """
    if "variant_index" in tile and "k_total" in tile:
        return int(tile["variant_index"]), int(tile["k_total"])
    m = _VID_RE.search(tile["variant_id"])
    if not m:
        raise ValueError(f"Cannot parse variant index from variant_id {tile['variant_id']!r}")
    vi = int(m.group(1)) - 1
    if sizes is None:
        raise ValueError(
            "Need the library CSV to find k_total (pass --library, or keep library.csv next to design.json)"
        )
    key = (_sorted_tuple(tile["inlet_sides"]), _sorted_tuple(tile["outlet_sides"]))
    k_total = sizes.get(key)
    if k_total is None or not (0 <= vi < k_total):
        raise ValueError(f"Variant {tile['variant_id']} (index {vi}) not found in library combo {key} (size {k_total})")
    return vi, k_total


def build_board(data: dict) -> Dict[Tuple[int, int], PlacedTile]:
    """Rebuild the board for the hydraulic solver from design.json (per-leg R as stored)."""
    board: Dict[Tuple[int, int], PlacedTile] = {}
    for t in data["tiles"]:
        cell = (int(t["cell"][0]), int(t["cell"][1]))
        variant = TileVariant(
            variant_id=t["variant_id"],
            inlet_sides=tuple(t["inlet_sides"]),
            outlet_sides=tuple(t["outlet_sides"]),
            R_per_side={k: float(v) for k, v in t["R_per_side"].items()},
        )
        board[cell] = PlacedTile(cell, variant)
    return board


# --------------------------------------------------------------------------------------
# Geometry assembly
# --------------------------------------------------------------------------------------
def tile_raster(tile: dict, vi: int, k_total: int) -> np.ndarray:
    """25x25 boolean raster of one tile, identical to the one used for library measurement.

    The side order of ``inlet_sides`` / ``outlet_sides`` is kept exactly as stored (the meander
    mapping depends on it).
    """
    mask = generate_tile_raster(
        tuple(tile["inlet_sides"]),
        tuple(tile["outlet_sides"]),
        side_len=SIDE_LEN,
        variant_idx=vi,
        k_total=k_total,
    )[0]
    return mask


def crosscheck_saved_raster(lib_dir: Optional[str], lib: Optional[Library], variant_id: str, mask: np.ndarray) -> str:
    """Compare the regenerated raster with the .npy saved at library-measurement time (if usable)."""
    if not lib_dir or lib is None or variant_id not in lib.by_id:
        return "skipped"
    rel = lib.by_id[variant_id].maze_file
    if not rel:
        return "skipped"
    path = os.path.join(lib_dir, rel)
    if not os.path.exists(path):
        return "skipped"
    saved = np.load(path)
    if saved.shape != mask.shape:  # e.g. the 13x13 visual dataset shares the same file names
        return "skipped"
    return "match" if np.array_equal(saved.astype(bool), mask) else "MISMATCH"


def assemble_chip_mask(spec: DesignSpec, tile_masks: Dict[Tuple[int, int], np.ndarray]) -> Tuple[np.ndarray, int]:
    """Stack tile rasters on the grid. Tile (r, c) occupies rows r*tp..(r+1)*tp-1, cols c*tp..(c+1)*tp-1."""
    n = spec.grid_size
    tp = next(iter(tile_masks.values())).shape[0]
    chip = np.zeros((n * tp, n * tp), dtype=bool)
    for (r, c), m in tile_masks.items():
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError(f"Tile cell {(r, c)} outside the {n}x{n} grid")
        if m.shape != (tp, tp):
            raise ValueError(f"Tile raster at {(r, c)} has shape {m.shape}, expected {(tp, tp)}")
        chip[r * tp:(r + 1) * tp, c * tp:(c + 1) * tp] |= m
    return chip, tp


def make_ports(spec: DesignSpec, tp: int) -> Tuple["Port", List["Port"]]:
    """Inlet on the N face of the inlet tile's top pixel, outlets on the S face of the bottom pixels."""
    mid = tp // 2
    r_in, c_in = spec.inlet_cell
    inlet = Port(r_in * tp, c_in * tp + mid, "N")
    outlets = [Port((r + 1) * tp - 1, c * tp + mid, "S") for r, c in spec.outlet_cells]
    return inlet, outlets


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """4-connected component labels of the fluid pixels (-1 for solid). Union-find, no scipy needed."""
    idx = -np.ones(mask.shape, dtype=np.int64)
    ys, xs = np.nonzero(mask)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    ra_, ca_ = np.nonzero(mask[:, :-1] & mask[:, 1:])
    for r, c in zip(ra_, ca_):
        union(int(idx[r, c]), int(idx[r, c + 1]))
    rd_, cd_ = np.nonzero(mask[:-1, :] & mask[1:, :])
    for r, c in zip(rd_, cd_):
        union(int(idx[r, c]), int(idx[r + 1, c]))
    roots: Dict[int, int] = {}
    labels = -np.ones(mask.shape, dtype=np.int64)
    for k in range(n):
        rt = find(k)
        labels[ys[k], xs[k]] = roots.setdefault(rt, len(roots))
    return labels, len(roots)


@dataclass
class IntegrityReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_fluid_pixels: int = 0
    n_components: int = 0
    n_cross_tile_links: int = 0
    n_expected_links: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def check_mask_integrity(
    chip: np.ndarray,
    tile_masks: Dict[Tuple[int, int], np.ndarray],
    tp: int,
    inlet: "Port",
    outlets: List["Port"],
    board: Dict[Tuple[int, int], PlacedTile],
) -> IntegrityReport:
    """Strict checks of the assembled raster. Errors mean the geometry is not what the solver assumes."""
    rep = IntegrityReport(ok=True)
    mid = tp // 2

    # 1. No pixel lost or merged
    expected_pixels = int(sum(int(m.sum()) for m in tile_masks.values()))
    rep.n_fluid_pixels = int(chip.sum())
    if rep.n_fluid_pixels != expected_pixels:
        rep.errors.append(f"fluid pixel count {rep.n_fluid_pixels} != sum of tile pixels {expected_pixels}")

    # 2. Ports are open pixels
    for name, p in [("inlet", inlet)] + [(f"outlet{j}", o) for j, o in enumerate(outlets)]:
        if not chip[p.row, p.col]:
            rep.errors.append(f"{name} pixel {(p.row, p.col)} is not fluid")

    # 3. Cross-tile contacts only at port pixels, and exactly where facing legs exist
    links = 0
    h, w = chip.shape
    for r in range(h):
        for c in range(w):
            if not chip[r, c]:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 >= h or c2 >= w or not chip[r2, c2]:
                    continue
                if (r // tp, c // tp) == (r2 // tp, c2 // tp):
                    continue
                links += 1
                if dr == 0:  # horizontal link between tiles
                    legal = (r % tp == mid) and (c % tp == tp - 1) and (c2 % tp == 0)
                else:        # vertical link between tiles
                    legal = (c % tp == mid) and (r % tp == tp - 1) and (r2 % tp == 0)
                if not legal:
                    rep.errors.append(f"unexpected contact between tiles at pixels {(r, c)} - {(r2, c2)}")
    rep.n_cross_tile_links = links

    expected = 0
    for (r, c), pt in board.items():
        act = pt.variant.active_sides
        if "E" in act and (r, c + 1) in board and "W" in board[(r, c + 1)].variant.active_sides:
            expected += 1
        if "S" in act and (r + 1, c) in board and "N" in board[(r + 1, c)].variant.active_sides:
            expected += 1
    rep.n_expected_links = expected
    if links != expected:
        rep.errors.append(f"tile-to-tile channel links {links} != expected facing-leg pairs {expected}")

    # 4. Connectivity
    labels, ncomp = label_components(chip)
    rep.n_components = ncomp
    if ncomp > 1:
        rep.warnings.append(f"{ncomp} separate fluid components (unreachable pieces carry no flow)")
    if chip[inlet.row, inlet.col]:
        lab_in = labels[inlet.row, inlet.col]
        for j, o in enumerate(outlets):
            if chip[o.row, o.col] and labels[o.row, o.col] != lab_in:
                rep.warnings.append(f"outlet {j} is not connected to the inlet in the raster")

    rep.ok = not rep.errors
    return rep


# --------------------------------------------------------------------------------------
# Backends (all expose .solve(FlowProblem) -> FlowResult)
# --------------------------------------------------------------------------------------
def raster_flow_fractions(mask: np.ndarray, inlet: "Port", outlets: List["Port"]) -> np.ndarray:
    """Outlet flow fractions of a uniform pixel-resistor network (1 unit per pixel step).

    Each outlet pixel is grounded through a half-step ghost resistor (R = 0.5), which puts the
    pressure reference on the port face just like COMSOL's boundary does.
    """
    labels, _ = label_components(mask)
    lab_in = labels[inlet.row, inlet.col]
    keep = labels == lab_in
    idx = -np.ones(mask.shape, dtype=np.int64)
    ys, xs = np.nonzero(keep)
    n = len(ys)
    idx[ys, xs] = np.arange(n)

    ra, ca = np.nonzero(keep[:, :-1] & keep[:, 1:])
    rd, cd = np.nonzero(keep[:-1, :] & keep[1:, :])
    a = np.concatenate([idx[ra, ca], idx[rd, cd]])
    b = np.concatenate([idx[ra, ca + 1], idx[rd + 1, cd]])

    ground = np.zeros(n)
    out_idx = []
    for o in outlets:
        if keep[o.row, o.col]:
            ground[idx[o.row, o.col]] += 2.0
            out_idx.append(int(idx[o.row, o.col]))
        else:
            out_idx.append(-1)

    rhs = np.zeros(n)
    rhs[idx[inlet.row, inlet.col]] = 1.0

    if _HAVE_SCIPY:
        A = _sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
        A = (A + A.T).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel()
        L = (_sp.diags(deg + ground) - A).tocsc()
        p = _spla.spsolve(L, rhs)
    else:
        if n > 5000:
            raise RuntimeError("Install scipy for the raster backend on large chips (dense solve would be too big)")
        A = np.zeros((n, n))
        A[a, b] = 1.0
        A[b, a] = 1.0
        L = np.diag(A.sum(axis=1) + ground) - A
        p = np.linalg.solve(L, rhs)

    flows = np.array([2.0 * p[i] if i >= 0 else 0.0 for i in out_idx])
    total = float(flows.sum())
    if total <= 0 or abs(total - 1.0) > 1e-6:
        raise RuntimeError(f"raster network does not conserve flow (sum of outlet flows = {total})")
    return flows / total


class RasterRunner:
    """Offline backend: uniform pixel-resistor network. Independent check of the geometry assembly."""

    name = "raster"

    def solve(self, prob: "FlowProblem") -> "FlowResult":
        frac = raster_flow_fractions(prob.mask, prob.inlet, prob.outlets)
        return FlowResult(
            p_in=float("nan"),
            q_in=prob.q_in,
            q_out=frac * prob.q_in,
            backend="raster",
            q_in_measured=prob.q_in,
        )


class MockRunner:
    """Dry-run backend: solver fractions with a small deterministic perturbation."""

    name = "mock"

    def __init__(self, solver_fractions: np.ndarray, r_total: float, perturb: float = 0.015):
        self.fr = np.asarray(solver_fractions, dtype=float)
        self.r_total = float(r_total)
        self.perturb = perturb

    def solve(self, prob: "FlowProblem") -> "FlowResult":
        k = np.arange(1, len(self.fr) + 1)
        fr = np.maximum(self.fr * (1.0 + self.perturb * np.sin(k)), 1e-12)
        fr = fr / fr.sum()
        return FlowResult(
            p_in=self.r_total * prob.q_in,
            q_in=prob.q_in,
            q_out=fr * prob.q_in,
            backend="mock",
            q_in_measured=prob.q_in,
        )


# --------------------------------------------------------------------------------------
# Comparison and reporting
# --------------------------------------------------------------------------------------
def compare_flows(spec: DesignSpec, solver_q: np.ndarray, backend_q_out: np.ndarray, tol: float) -> dict:
    qin = float(spec.inlet_flow)
    target = np.asarray(spec.target_flows, dtype=float)
    denom = np.maximum(target, DENOM_FLOOR_FRACTION * qin)

    solver_q = np.asarray(solver_q, dtype=float)
    s_sum = float(solver_q.sum())
    fr_s = solver_q / s_sum if s_sum > 0 else np.zeros_like(solver_q)

    qo = np.asarray(backend_q_out, dtype=float)
    o_sum = float(qo.sum())
    fr_c = qo / o_sum if o_sum > 0 else np.zeros_like(qo)
    q_c = fr_c * qin                      # scaled to the spec's inlet flow so units match the targets

    rel_c = (q_c - target) / denom
    rel_s = (solver_q - target) / denom
    vs_solver = (q_c - solver_q) / np.maximum(solver_q, DENOM_FLOOR_FRACTION * qin)

    return {
        "target_flows": target.tolist(),
        "solver_flows": solver_q.tolist(),
        "backend_flows": q_c.tolist(),
        "target_fractions": (target / qin).tolist(),
        "solver_fractions": fr_s.tolist(),
        "backend_fractions": fr_c.tolist(),
        "rel_err_backend_vs_target": rel_c.tolist(),
        "rel_err_solver_vs_target": rel_s.tolist(),
        "rel_diff_backend_vs_solver": vs_solver.tolist(),
        "e_q_backend": float(np.mean(np.minimum(np.abs(rel_c), ERROR_CLIP))),
        "e_q_solver": float(np.mean(np.minimum(np.abs(rel_s), ERROR_CLIP))),
        "max_abs_rel_backend": float(np.max(np.abs(rel_c))),
        "max_abs_rel_solver": float(np.max(np.abs(rel_s))),
        "success_backend": bool(np.all(np.abs(rel_c) <= tol)),
        "success_solver": bool(np.all(np.abs(rel_s) <= tol)),
        "tv_distance_vs_solver": float(0.5 * np.sum(np.abs(fr_c - fr_s))),
        "max_abs_rel_vs_solver": float(np.max(np.abs(vs_solver))),
        "tolerance": float(tol),
    }


def _save_outputs(out_dir: str, summary: dict, chip: np.ndarray, inlet, outlets, cmp_: Optional[dict], verbose: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "chip_mask.npy"), chip)
    with open(os.path.join(out_dir, "comsol_validation.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    if cmp_ is not None:
        with open(os.path.join(out_dir, "comsol_validation.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["outlet", "cell", "target_frac", "solver_frac", "backend_frac",
                        "rel_err_backend_vs_target", "rel_err_solver_vs_target", "rel_diff_backend_vs_solver"])
            for j, cell in enumerate(summary["outlet_cells"]):
                w.writerow([j + 1, cell, cmp_["target_fractions"][j], cmp_["solver_fractions"][j],
                            cmp_["backend_fractions"][j], cmp_["rel_err_backend_vs_target"][j],
                            cmp_["rel_err_solver_vs_target"][j], cmp_["rel_diff_backend_vs_solver"][j]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncols = 2 if cmp_ is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols + 2, 6.5), dpi=140)
        axes = np.atleast_1d(axes)
        ax = axes[0]
        ax.imshow(~chip, cmap="gray", interpolation="nearest")
        ax.plot(inlet.col, inlet.row, "v", color="#27ae60", markersize=9, label="inlet")
        for j, o in enumerate(outlets):
            ax.plot(o.col, o.row, "^", color="#c0392b", markersize=9, label="outlets" if j == 0 else None)
        ax.set_title(f"Assembled chip raster ({chip.shape[0]}x{chip.shape[1]} px)")
        ax.legend(loc="lower right", fontsize=7)
        ax.axis("off")
        if cmp_ is not None:
            ax = axes[1]
            x = np.arange(len(cmp_["target_fractions"]))
            ax.bar(x - 0.27, cmp_["target_fractions"], 0.27, label="target", color="#95a5a6")
            ax.bar(x, cmp_["solver_fractions"], 0.27, label="hydraulic solver", color="#1a73e8")
            ax.bar(x + 0.27, cmp_["backend_fractions"], 0.27, label=summary["backend"], color="#e67e22")
            ax.set_xticks(x)
            ax.set_xticklabels([f"OUT {j + 1}" for j in x])
            ax.set_ylabel("flow fraction of inlet")
            ax.set_title("Outlet flow fractions")
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "comsol_validation.png"), bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # plotting must never break validation
        if verbose:
            print(f"[validate] plot skipped: {exc}")


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------
def validate_rl_design(
    design_path: str,
    output_dir: str = "verification_results_v2",
    backend: str = "comsol",
    plugins_path: str = DEFAULT_COMSOL_PLUGINS,
    tol: float = 0.05,
    library_path: Optional[str] = None,
    library_dir: Optional[str] = None,
    pixel_um: float = 100.0,
    thickness_um: float = 100.0,
    u_in: float = 1e-4,
    work_dir: Optional[str] = None,
    mesh_hmax_frac: float = 1.0 / 3.0,
    mesh_hmin_frac: float = 1.0 / 16.0,
    timeout_s: int = 7200,
    dry_run: bool = False,
    allow_baseline_library: bool = False,
    save_mph: bool = False,
    verbose: bool = True,
) -> dict:
    """Validate ``design.json`` and return a summary dict (also written to ``output_dir``).

    ``pixel_um``, ``thickness_um``, ``u_in`` and the mesh fractions must equal the values used when
    the library was measured (``measure_v2`` defaults), otherwise the comparison mixes two models.
    """
    t0 = time.time()
    if dry_run:
        backend = "none"
    if backend not in ("comsol", "raster", "mock", "none"):
        raise ValueError(f"unknown backend {backend!r}")

    spec, data = load_design(design_path)
    board = build_board(data)
    design_dir = os.path.dirname(os.path.abspath(design_path))

    # Library (only needed to recover k_total when tile entries do not store it)
    lib: Optional[Library] = None
    lib_csv = library_path or os.path.join(design_dir, "library.csv")
    if os.path.exists(lib_csv):
        lib = Library.from_csv(lib_csv)
    sizes = combo_sizes(lib) if lib is not None else None

    max_r = max(max(pt.variant.R_per_side.values()) for pt in board.values())
    baseline_lib = max_r < BASELINE_R_THRESHOLD
    if baseline_lib:
        msg = (f"largest leg resistance is {max_r:.3g}: this looks like the unit-less baseline library, "
               "not COMSOL-measured values. A COMSOL comparison is only meaningful for a COMSOL library.")
        if backend == "comsol" and not allow_baseline_library:
            raise ValueError(msg + " Pass --allow-baseline-library to run anyway.")
        if verbose:
            print(f"[validate] WARNING: {msg}")

    # 1. Geometry
    tile_masks: Dict[Tuple[int, int], np.ndarray] = {}
    crosscheck: Dict[str, str] = {}
    for t in data["tiles"]:
        cell = (int(t["cell"][0]), int(t["cell"][1]))
        vi, k_total = variant_geometry_index(t, sizes)
        mask = tile_raster(t, vi, k_total)
        tile_masks[cell] = mask
        status = crosscheck_saved_raster(library_dir, lib, t["variant_id"], mask)
        crosscheck[t["variant_id"]] = status
        if status == "MISMATCH":
            raise ValueError(f"Regenerated raster of {t['variant_id']} differs from the saved library raster. "
                             "The library was generated with different settings.")
    chip, tp = assemble_chip_mask(spec, tile_masks)
    inlet, outlets = make_ports(spec, tp)
    integ = check_mask_integrity(chip, tile_masks, tp, inlet, outlets, board)

    # 2. Hydraulic solver on the reconstructed board
    sres = solve_network_v2(board, spec)
    if not sres.ok or sres.stranded:
        raise RuntimeError("Hydraulic solver failed or inlet is stranded for this design")
    solver_q = np.asarray(sres.outlet_flows, dtype=float)
    r_solver = float(sres.p_in / spec.inlet_flow)
    stored = data.get("realized_flows")
    consistency = None
    if stored is not None and len(stored) == len(solver_q):
        stored = np.asarray(stored, dtype=float)
        consistency = float(np.max(np.abs(stored - solver_q) / np.maximum(np.abs(stored), 1e-30)))

    summary: dict = {
        "design": os.path.abspath(design_path),
        "backend": backend,
        "n_tiles": len(board),
        "grid_size": spec.grid_size,
        "inlet_cell": list(spec.inlet_cell),
        "outlet_cells": [list(rc) for rc in spec.outlet_cells],
        "chip_raster_shape": list(chip.shape),
        "pixel_um": pixel_um,
        "thickness_um": thickness_um,
        "u_in": u_in,
        "library_csv": lib_csv if lib is not None else None,
        "baseline_library_warning": bool(baseline_lib),
        "raster_crosscheck": {k: v for k, v in crosscheck.items() if v != "skipped"},
        "integrity": integ.to_dict(),
        "solver_vs_design_json_max_rel_diff": consistency,
        "R_total_solver": r_solver,
    }

    if verbose:
        print(f"[validate] {len(board)} tiles -> chip raster {chip.shape[0]}x{chip.shape[1]} px "
              f"({int(chip.sum())} fluid pixels)")
        print(f"[validate] integrity: {'OK' if integ.ok else 'FAILED'}  "
              f"links {integ.n_cross_tile_links}/{integ.n_expected_links}  components {integ.n_components}")
        for e in integ.errors:
            print(f"[validate]   ERROR: {e}")
        for w in integ.warnings:
            print(f"[validate]   warning: {w}")
        if consistency is not None and consistency > 1e-6:
            print(f"[validate] WARNING: recomputed solver flows differ from design.json by {consistency:.2e} (relative)")

    if not integ.ok:
        summary["status"] = "integrity_failed"
        _save_outputs(output_dir, summary, chip, inlet, outlets, None, verbose)
        raise RuntimeError("Assembled chip failed the integrity check; see messages above. Nothing was sent to COMSOL.")

    if backend == "none":
        summary["status"] = "built_only"
        _save_outputs(output_dir, summary, chip, inlet, outlets, None, verbose)
        return summary

    # 3. Backend solve
    prob = FlowProblem(mask=chip, inlet=inlet, outlets=outlets, pixel_um=pixel_um,
                       thickness_um=thickness_um, u_in=u_in)
    prob.validate()
    if backend == "comsol":
        runner = ComsolJavaRunner(
            comsol_plugins=plugins_path,
            work_dir=work_dir or os.path.join(output_dir, "comsol_work"),
            timeout_s=timeout_s,
            mesh_hmax_frac=mesh_hmax_frac,
            mesh_hmin_frac=mesh_hmin_frac,
            save_mph=save_mph,
        )
        if verbose:
            print("[validate] running COMSOL (large chip: this can take several minutes)...")
    elif backend == "raster":
        runner = RasterRunner()
    else:
        runner = MockRunner(solver_q / solver_q.sum(), r_solver)

    t1 = time.time()
    res = runner.solve(prob)
    solve_s = time.time() - t1
    if len(res.q_out) != len(outlets):
        raise RuntimeError(f"backend returned {len(res.q_out)} outlet flows, expected {len(outlets)}")

    cmp_ = compare_flows(spec, solver_q, np.asarray(res.q_out, dtype=float), tol)
    q_in_meas = res.q_in_measured if res.q_in_measured else res.q_in
    mass_err = abs(float(np.sum(res.q_out)) - q_in_meas) / max(q_in_meas, 1e-30)
    r_backend = (res.p_in / q_in_meas) if (q_in_meas and np.isfinite(res.p_in)) else float("nan")

    summary.update({
        "status": "ok",
        "comparison": cmp_,
        "backend_mass_error": mass_err,
        "R_total_backend": r_backend,
        "R_total_ratio_backend_over_solver": (r_backend / r_solver) if (np.isfinite(r_backend) and r_solver > 0) else None,
        "backend_p_in": float(res.p_in),
        "backend_q_in_measured": float(q_in_meas),
        "backend_solve_seconds": solve_s,
        "total_seconds": time.time() - t0,
        "design_json_metrics": data.get("metrics"),
    })
    _save_outputs(output_dir, summary, chip, inlet, outlets, cmp_, verbose)

    if verbose:
        _print_table(spec, cmp_, summary)
    return summary


def _print_table(spec: DesignSpec, c: dict, s: dict) -> None:
    print("\n=========== OUTLET FLOW COMPARISON (fractions of inlet) ===========")
    print(f"{'outlet':<10}{'cell':<8}{'target':>9}{'solver':>9}{s['backend']:>9}{'err vs tgt':>12}{'diff vs solver':>16}")
    for j, rc in enumerate(spec.outlet_cells):
        print(f"OUT {j + 1:<6}{str(tuple(rc)):<8}{c['target_fractions'][j]:>9.4f}{c['solver_fractions'][j]:>9.4f}"
              f"{c['backend_fractions'][j]:>9.4f}{100 * c['rel_err_backend_vs_target'][j]:>11.2f}%"
              f"{100 * c['rel_diff_backend_vs_solver'][j]:>15.2f}%")
    tol = c["tolerance"]
    print(f"\n{s['backend']:>8}: E_Q = {c['e_q_backend']:.4f}   max |err| = {100 * c['max_abs_rel_backend']:.2f}%   "
          f"success@{100 * tol:.0f}% = {c['success_backend']}")
    print(f"  solver: E_Q = {c['e_q_solver']:.4f}   max |err| = {100 * c['max_abs_rel_solver']:.2f}%   "
          f"success@{100 * tol:.0f}% = {c['success_solver']}")
    print(f"  total-variation distance of flow split (backend vs solver) = {c['tv_distance_vs_solver']:.4f}")
    print(f"  backend mass error = {s['backend_mass_error']:.2e}")
    if s.get("R_total_ratio_backend_over_solver") is not None:
        print(f"  total resistance backend/solver = {s['R_total_ratio_backend_over_solver']:.3f}")
    print("====================================================================")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Validate an RL design.json against COMSOL (or offline checks)")
    ap.add_argument("--design", default="results/stage1_v2/design.json")
    ap.add_argument("--backend", choices=["comsol", "raster", "mock", "none"], default="comsol")
    ap.add_argument("--comsol-plugins", default=DEFAULT_COMSOL_PLUGINS)
    ap.add_argument("--out", default=None, help="output folder (default: <design folder>/comsol_check)")
    ap.add_argument("--work-dir", default=None, help="COMSOL work/cache folder (default: <out>/comsol_work)")
    ap.add_argument("--library", default=None, help="library.csv (default: library.csv next to design.json)")
    ap.add_argument("--library-dir", default=None,
                    help="folder of the COMSOL library (contains mazes/*.npy) for an optional raster cross-check")
    ap.add_argument("--pixel-um", type=float, default=100.0)
    ap.add_argument("--thickness-um", type=float, default=100.0)
    ap.add_argument("--u-in", type=float, default=1e-4)
    ap.add_argument("--mesh-hmax-frac", type=float, default=1.0 / 3.0)
    ap.add_argument("--mesh-hmin-frac", type=float, default=1.0 / 16.0)
    ap.add_argument("--timeout", type=int, default=7200, help="COMSOL timeout in seconds")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--dry-run", action="store_true", help="build and check the chip only")
    ap.add_argument("--allow-baseline-library", action="store_true")
    ap.add_argument("--save-mph", action="store_true")
    ap.add_argument("--fail-on-tol", action="store_true",
                    help="exit with code 3 if the backend result misses the tolerance")
    args = ap.parse_args(argv)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.design)), "comsol_check")
    summary = validate_rl_design(
        design_path=args.design, output_dir=out, backend=args.backend, plugins_path=args.comsol_plugins,
        tol=args.tol, library_path=args.library, library_dir=args.library_dir, pixel_um=args.pixel_um,
        thickness_um=args.thickness_um, u_in=args.u_in, work_dir=args.work_dir,
        mesh_hmax_frac=args.mesh_hmax_frac, mesh_hmin_frac=args.mesh_hmin_frac, timeout_s=args.timeout,
        dry_run=args.dry_run, allow_baseline_library=args.allow_baseline_library, save_mph=args.save_mph,
    )
    print(f"\n[validate] results saved to {os.path.abspath(out)}")
    if args.fail_on_tol and summary.get("comparison") and not summary["comparison"]["success_backend"]:
        raise SystemExit(3)
    return summary


if __name__ == "__main__":
    main()
