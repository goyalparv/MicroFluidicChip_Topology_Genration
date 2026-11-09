"""Flow runners for 2D Navier-Stokes / Stokes flow in raster pixel geometries."""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

VISCOSITY = 1.01e-3  # Pa s
DENSITY = 998.2      # kg/m^3


class Port(NamedTuple):
    row: int
    col: int
    side: str  # "N", "S", "W", "E"


def port_xy_um(port: Port, n_rows: int, pixel_um: float) -> Tuple[float, float]:
    """Coordinates in micrometres of the center of the port face."""
    w = float(pixel_um)
    y_mid = (n_rows - port.row - 0.5) * w
    x_mid = (port.col + 0.5) * w
    if port.side == "W":
        return port.col * w, y_mid
    if port.side == "E":
        return (port.col + 1) * w, y_mid
    if port.side == "N":
        return x_mid, (n_rows - port.row) * w
    if port.side == "S":
        return x_mid, (n_rows - port.row - 1) * w
    raise ValueError(f"Invalid port side {port.side!r}")


def rects_from_mask(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Decompose boolean pixel mask (True = fluid) into disjoint bounding rectangles."""
    h, w = mask.shape
    active = {}
    out = []
    for r in range(h + 1):
        runs = set()
        if r < h:
            row = mask[r]
            c = 0
            while c < w:
                if row[c]:
                    c0 = c
                    while c < w and row[c]:
                        c += 1
                    runs.add((c0, c - 1))
                else:
                    c += 1
        for key in [k for k in active if k not in runs]:
            r0 = active.pop(key)
            out.append((r0, key[0], r - r0, key[1] - key[0] + 1))
        for key in runs:
            if key not in active:
                active[key] = r
    out.sort()
    return out


@dataclass
class FlowProblem:
    mask: np.ndarray
    inlet: Port
    outlets: List[Port]
    pixel_um: float = 100.0
    thickness_um: float = 0.0
    u_in: float = 1e-4
    viscosity: float = VISCOSITY
    density: float = DENSITY

    @property
    def depth_m(self) -> float:
        return self.thickness_um * 1e-6 if self.thickness_um > 0 else 1.0

    @property
    def width_m(self) -> float:
        return self.pixel_um * 1e-6

    @property
    def q_in(self) -> float:
        return self.u_in * self.width_m * self.depth_m

    def validate(self) -> None:
        h, w = self.mask.shape
        for p in [self.inlet] + list(self.outlets):
            if not (0 <= p.row < h and 0 <= p.col < w and self.mask[p.row, p.col]):
                raise ValueError(f"Port {p} is not an open pixel in {h}x{w} mask")
        if not self.outlets:
            raise ValueError("At least one outlet port is required")


@dataclass
class FlowResult:
    p_in: float
    q_in: float
    q_out: np.ndarray
    thickness_applied: bool = True
    backend: str = ""
    q_in_measured: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def resistance(self) -> float:
        q = self.q_in_measured if self.q_in_measured else self.q_in
        return self.p_in / q

    @property
    def mass_error(self) -> float:
        q = self.q_in_measured if self.q_in_measured else self.q_in
        return abs(float(np.sum(self.q_out)) - q) / max(q, 1e-30)


def job_text(prob: FlowProblem, mesh_hmax_um: float, mesh_hmin_um: float, save_mph: bool) -> str:
    prob.validate()
    n_rows = prob.mask.shape[0]
    rects = rects_from_mask(prob.mask)
    lines = [
        f"pixel_um {prob.pixel_um!r}",
        f"thickness_um {prob.thickness_um!r}",
        f"viscosity {prob.viscosity!r}",
        f"density {prob.density!r}",
        f"u_in {prob.u_in!r}",
        f"mesh_hmax_um {mesh_hmax_um!r}",
        f"mesh_hmin_um {mesh_hmin_um!r}",
        f"save_mph {1 if save_mph else 0}",
        f"n_rects {len(rects)}",
    ]
    lines += [f"rect {r0} {c0} {nr} {nc}" for r0, c0, nr, nc in rects]
    lines.append(f"n_rows {n_rows}")
    for kind, p in [("inlet", prob.inlet)] + [("outlet", q) for q in prob.outlets]:
        x, y = port_xy_um(p, n_rows, prob.pixel_um)
        expr = {"W": "-u", "E": "u", "N": "v", "S": "-v"}[p.side]
        lines.append(f"{kind} {x!r} {y!r} {expr}")
    return "\n".join(lines) + "\n"


def parse_result(text: str, prob: FlowProblem) -> FlowResult:
    vals, outs = {}, {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "out_avg":
            outs[int(parts[1])] = float(parts[2])
        else:
            vals[parts[0]] = " ".join(parts[1:])
    if vals.get("status") != "ok":
        raise RuntimeError(f"COMSOL job failed: {vals.get('message', 'no message')}")
    w, d = prob.width_m, prob.depth_m
    q_out = np.array([outs[i] * w * d for i in range(len(prob.outlets))])
    q_in_meas = -float(vals["in_avg"]) * w * d
    return FlowResult(
        p_in=float(vals["p_in"]),
        q_in=prob.q_in,
        q_out=q_out,
        thickness_applied=vals.get("thickness_applied", "1") == "1",
        backend="comsol",
        q_in_measured=q_in_meas,
    )


class ComsolJavaRunner:
    """Executes FlowJob.java via COMSOL Java API with persistent job hashing/cache."""

    name = "comsol"

    def __init__(
        self,
        comsol_plugins: str = "C:/Program Files/COMSOL/COMSOL62/Multiphysics/plugins",
        work_dir: str = "results/comsol_work_v2",
        java: str = "java",
        javac: str = "javac",
        timeout_s: int = 3600,
        mesh_hmax_frac: float = 1 / 3,
        mesh_hmin_frac: float = 1 / 16,
        save_mph: bool = False,
    ):
        self.plugins = comsol_plugins
        self.work_dir = os.path.abspath(work_dir)
        self.java, self.javac = java, javac
        self.timeout_s = timeout_s
        self.mesh_hmax_frac, self.mesh_hmin_frac = mesh_hmax_frac, mesh_hmin_frac
        self.save_mph = save_mph
        self.bin_dir = os.path.join(self.work_dir, "bin")
        self._compiled = False

    def _classpath(self) -> str:
        sep = ";" if os.name == "nt" else ":"
        return f"{self.plugins}/*{sep}{self.bin_dir}"

    def compile(self) -> None:
        if self._compiled:
            return
        os.makedirs(self.bin_dir, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FlowJob.java")
        r = subprocess.run([self.javac, "-cp", f"{self.plugins}/*", "-d", self.bin_dir, src],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"javac compilation failed:\n{r.stdout}\n{r.stderr}")
        self._compiled = True

    def solve(self, prob: FlowProblem) -> FlowResult:
        text = job_text(prob, self.mesh_hmax_frac * prob.pixel_um, self.mesh_hmin_frac * prob.pixel_um, self.save_mph)
        key = hashlib.sha1(text.encode()).hexdigest()[:16]
        run_dir = os.path.join(self.work_dir, "jobs", key)
        res_path = os.path.join(run_dir, "result.txt")
        if not os.path.exists(res_path):
            self.compile()
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "job.txt"), "w", encoding="utf-8") as fh:
                fh.write(text)
            r = subprocess.run([self.java, "-cp", self._classpath(), "FlowJob", "job.txt", "result.txt"],
                               cwd=run_dir, capture_output=True, text=True, timeout=self.timeout_s)
            with open(os.path.join(run_dir, "stdout.txt"), "w", encoding="utf-8") as fh:
                fh.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
            if not os.path.exists(res_path):
                raise RuntimeError(f"COMSOL job produced no result (see {run_dir}):\n{r.stderr[-2000:]}")
        with open(res_path, "r", encoding="utf-8") as fh:
            return parse_result(fh.read(), prob)

