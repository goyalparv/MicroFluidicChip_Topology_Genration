"""Tile library representation and utilities for Stage 1.

Every tile variant defines:
  - inlet_sides: tuple of sides accepting flow (from "N", "S", "E", "W")
  - outlet_sides: tuple of sides exporting flow (disjoint from inlet_sides)
  - R_per_side: resistance of each active leg from/to the central junction
  - position_per_side: connection position along the 13-node face (fixed at 6 in Stage 1)
"""
from __future__ import annotations

import csv
import itertools
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

ALL_SIDES = ("N", "S", "E", "W")
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
DIRECTION_DELTAS = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}


@dataclass(frozen=True)
class TileVariant:
    variant_id: str
    inlet_sides: Tuple[str, ...]
    outlet_sides: Tuple[str, ...]
    R_per_side: Dict[str, float]
    position_per_side: Dict[str, int] = field(default_factory=dict)
    maze_file: str = ""

    def __post_init__(self):
        if not self.position_per_side:
            # Default connection position is 6 (center of 13-node border) for each active side
            active = set(self.inlet_sides) | set(self.outlet_sides)
            object.__setattr__(self, "position_per_side", {s: 6 for s in active})

    @property
    def active_sides(self) -> Set[str]:
        return set(self.inlet_sides) | set(self.outlet_sides)

    @property
    def total_r(self) -> float:
        return sum(self.R_per_side.values())


@dataclass
class PlacedTile:
    cell: Tuple[int, int]
    variant: TileVariant


def enumerate_side_combos() -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """Enumerate the 50 canonical (inlet_sides, outlet_sides) configurations.

    1in-1out: 4 * 3 = 12
    1in-2out: 4 * C(3,2) = 12
    1in-3out: 4 * 1 = 4
    2in-1out: C(4,2) * 2 = 12
    2in-2out: C(4,2) * C(2,2) = 6
    3in-1out: C(4,3) * 1 = 4
    Total = 50 combos.
    """
    combos = []
    # 1. 1in-1out
    for i_side in ALL_SIDES:
        rem = [s for s in ALL_SIDES if s != i_side]
        for o_side in rem:
            combos.append(((i_side,), (o_side,)))

    # 2. 1in-2out
    for i_side in ALL_SIDES:
        rem = [s for s in ALL_SIDES if s != i_side]
        for o_sides in itertools.combinations(rem, 2):
            combos.append(((i_side,), tuple(sorted(o_sides))))

    # 3. 1in-3out
    for i_side in ALL_SIDES:
        rem = tuple(sorted([s for s in ALL_SIDES if s != i_side]))
        combos.append(((i_side,), rem))

    # 4. 2in-1out
    for i_sides in itertools.combinations(ALL_SIDES, 2):
        rem = [s for s in ALL_SIDES if s not in i_sides]
        for o_side in rem:
            combos.append((tuple(sorted(i_sides)), (o_side,)))

    # 5. 2in-2out
    for i_sides in itertools.combinations(ALL_SIDES, 2):
        rem = tuple(sorted([s for s in ALL_SIDES if s not in i_sides]))
        combos.append((tuple(sorted(i_sides)), rem))

    # 6. 3in-1out
    for i_sides in itertools.combinations(ALL_SIDES, 3):
        rem = [s for s in ALL_SIDES if s not in i_sides]
        combos.append((tuple(sorted(i_sides)), (rem[0],)))

    return combos


class Library:
    """Manages the full collection of pre-simulated tile variants."""

    def __init__(self, variants: Sequence[TileVariant]):
        if not variants:
            raise ValueError("Library must contain at least one TileVariant")
        self.variants: List[TileVariant] = list(variants)
        self.by_id: Dict[str, TileVariant] = {v.variant_id: v for v in self.variants}
        self.by_combo: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], List[TileVariant]] = {}
        for v in self.variants:
            key = (v.inlet_sides, v.outlet_sides)
            self.by_combo.setdefault(key, []).append(v)

        # Precompute lookup for compatibility filtering
        self.variants_with_inlet: Dict[str, List[int]] = {s: [] for s in ALL_SIDES}
        self.variants_with_outlet: Dict[str, List[int]] = {s: [] for s in ALL_SIDES}
        for idx, v in enumerate(self.variants):
            for s in v.inlet_sides:
                self.variants_with_inlet[s].append(idx)
            for s in v.outlet_sides:
                self.variants_with_outlet[s].append(idx)

    def __len__(self) -> int:
        return len(self.variants)

    def __getitem__(self, idx: int) -> TileVariant:
        return self.variants[idx]

    @property
    def max_r(self) -> float:
        return max(max(v.R_per_side.values()) for v in self.variants)

    @property
    def min_r(self) -> float:
        return min(min(v.R_per_side.values()) for v in self.variants)

    def to_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["variant_id", "inlet_sides", "outlet_sides", "R_N", "R_S", "R_E", "R_W", "maze_file"])
            for v in self.variants:
                writer.writerow([
                    v.variant_id,
                    ";".join(v.inlet_sides),
                    ";".join(v.outlet_sides),
                    f"{v.R_per_side.get('N', ''):.6g}" if "N" in v.R_per_side else "",
                    f"{v.R_per_side.get('S', ''):.6g}" if "S" in v.R_per_side else "",
                    f"{v.R_per_side.get('E', ''):.6g}" if "E" in v.R_per_side else "",
                    f"{v.R_per_side.get('W', ''):.6g}" if "W" in v.R_per_side else "",
                    v.maze_file,
                ])

    @staticmethod
    def from_csv(path: str) -> "Library":
        variants = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                inlet = tuple(s.strip() for s in row["inlet_sides"].split(";") if s.strip())
                outlet = tuple(s.strip() for s in row["outlet_sides"].split(";") if s.strip())
                r_map = {}
                for s in ALL_SIDES:
                    val = row.get(f"R_{s}", "").strip()
                    if val:
                        r_map[s] = float(val)
                variants.append(TileVariant(
                    variant_id=row["variant_id"],
                    inlet_sides=inlet,
                    outlet_sides=outlet,
                    R_per_side=r_map,
                    maze_file=row.get("maze_file", "").strip()
                ))
        return Library(variants)

    @staticmethod
    def generate_initial_library(
        k_1in1out: int = 6,
        k_1in2out: int = 12,
        k_1in3out: int = 20,
        k_2in1out: int = 12,
        k_2in2out: int = 20,
        k_3in1out: int = 20,
        base_r: float = 1.0,
    ) -> "Library":
        """Generate a baseline library spanning the 50 side combos with varied resistance scales matching leg lengths."""
        combos = enumerate_side_combos()
        variants: List[TileVariant] = []

        for i_sides, o_sides in combos:
            n_in = len(i_sides)
            n_out = len(o_sides)
            tag = f"{n_in}in{n_out}out_{''.join(i_sides)}I_{''.join(o_sides)}O"

            if n_in == 1 and n_out == 1:
                k_count = k_1in1out
            elif n_in == 1 and n_out == 2:
                k_count = k_1in2out
            elif n_in == 1 and n_out == 3:
                k_count = k_1in3out
            elif n_in == 2 and n_out == 1:
                k_count = k_2in1out
            elif n_in == 2 and n_out == 2:
                k_count = k_2in2out
            else:
                k_count = k_3in1out

            for vi in range(k_count):
                vid = f"{tag}_v{vi + 1:02d}"
                paths = get_tile_leg_paths(i_sides, o_sides, variant_idx=vi, k_total=k_count)
                r_map = {}
                for s, p in paths.items():
                    leg_length = max(1, len(p) - 1)  # straight leg has length 6
                    # Hydraulic resistance proportional to leg length
                    r_map[s] = float(base_r * (leg_length / 6.0))

                variants.append(TileVariant(
                    variant_id=vid,
                    inlet_sides=i_sides,
                    outlet_sides=o_sides,
                    R_per_side=r_map,
                    maze_file=f"mazes/{vid}.npy"
                ))

        return Library(variants)

    def save_visual_dataset(self, out_dir: str) -> int:
        """Export 13x13 numpy arrays and high-resolution PNG visual images for every variant."""
        mazes_dir = os.path.join(out_dir, "mazes")
        os.makedirs(mazes_dir, exist_ok=True)
        count = 0

        # Helper to find variant index within combo
        combo_counts: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], int] = {}
        for v in self.variants:
            key = (v.inlet_sides, v.outlet_sides)
            combo_counts[key] = combo_counts.get(key, 0) + 1

        combo_seen: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], int] = {}
        for v in self.variants:
            key = (v.inlet_sides, v.outlet_sides)
            vi = combo_seen.get(key, 0)
            combo_seen[key] = vi + 1
            k_total = combo_counts[key]

            paths = get_tile_leg_paths(v.inlet_sides, v.outlet_sides, variant_idx=vi, k_total=k_total)
            grid = make_tile_13x13_grid(v.inlet_sides, v.outlet_sides, variant_idx=vi, k_total=k_total)

            npy_path = os.path.join(out_dir, v.maze_file) if v.maze_file else os.path.join(mazes_dir, f"{v.variant_id}.npy")
            png_path = os.path.splitext(npy_path)[0] + ".png"
            os.makedirs(os.path.dirname(os.path.abspath(npy_path)), exist_ok=True)
            np.save(npy_path, grid)
            render_maze_image(paths, v, png_path)
            count += 1

        print(f"Exported {count} 13x13 maze images and arrays to: {os.path.abspath(mazes_dir)}")
        return count


def make_line_coords(p1: Tuple[int, int], p2: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Generate discrete consecutive coordinates along a straight horizontal or vertical line."""
    r1, c1 = p1
    r2, c2 = p2
    pts = []
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        for c in range(c1, c2 + step, step):
            pts.append((r1, c))
    elif c1 == c2:
        step = 1 if r2 >= r1 else -1
        for r in range(r1, r2 + step, step):
            pts.append((r, c1))
    else:
        raise ValueError(f"Diagonal line not allowed between {p1} and {p2}")
    return pts


def make_leg_path_13x13(side: str, meander_level: int) -> List[Tuple[int, int]]:
    """Generate 100% continuous, non-intersecting path along 13x13 grid lines from (6, 6) to border port.

    Center junction is at (6, 6).
    Border ports (position 6) are at:
      N: (0, 6)
      S: (12, 6)
      W: (6, 0)
      E: (6, 12)

    Each leg meanders strictly inside its dedicated quadrant, guaranteeing zero collisions:
      N leg: rows 0..6, cols 6..6+w (Northeast)
      S leg: rows 6..12, cols 6-w..6 (Southwest)
      W leg: rows 6-w..6, cols 0..6 (Northwest)
      E leg: rows 6..6+w, cols 6..12 (Southeast)
    """
    if meander_level <= 0:
        if side == "N":
            return make_line_coords((6, 6), (0, 6))
        elif side == "S":
            return make_line_coords((6, 6), (12, 6))
        elif side == "W":
            return make_line_coords((6, 6), (6, 0))
        elif side == "E":
            return make_line_coords((6, 6), (6, 12))
        raise ValueError(f"Unknown side: {side}")

    w = min(max(meander_level, 1), 5)
    path: List[Tuple[int, int]] = []

    def add_segment(p1: Tuple[int, int], p2: Tuple[int, int]):
        seg = make_line_coords(p1, p2)
        if not path:
            path.extend(seg)
        else:
            path.extend(seg[1:])

    if side == "W":
        add_segment((6, 6), (6, 5))
        add_segment((6, 5), (6 - w, 5))
        add_segment((6 - w, 5), (6 - w, 4))
        add_segment((6 - w, 4), (6, 4))
        add_segment((6, 4), (6, 3))
        add_segment((6, 3), (6 - w, 3))
        add_segment((6 - w, 3), (6 - w, 2))
        add_segment((6 - w, 2), (6, 2))
        add_segment((6, 2), (6, 1))
        add_segment((6, 1), (6 - w, 1))
        add_segment((6 - w, 1), (6 - w, 0))
        add_segment((6 - w, 0), (6, 0))
    elif side == "E":
        add_segment((6, 6), (6, 7))
        add_segment((6, 7), (6 + w, 7))
        add_segment((6 + w, 7), (6 + w, 8))
        add_segment((6 + w, 8), (6, 8))
        add_segment((6, 8), (6, 9))
        add_segment((6, 9), (6 + w, 9))
        add_segment((6 + w, 9), (6 + w, 10))
        add_segment((6 + w, 10), (6, 10))
        add_segment((6, 10), (6, 11))
        add_segment((6, 11), (6 + w, 11))
        add_segment((6 + w, 11), (6 + w, 12))
        add_segment((6 + w, 12), (6, 12))
    elif side == "N":
        add_segment((6, 6), (5, 6))
        add_segment((5, 6), (5, 6 + w))
        add_segment((5, 6 + w), (4, 6 + w))
        add_segment((4, 6 + w), (4, 6))
        add_segment((4, 6), (3, 6))
        add_segment((3, 6), (3, 6 + w))
        add_segment((3, 6 + w), (2, 6 + w))
        add_segment((2, 6 + w), (2, 6))
        add_segment((2, 6), (1, 6))
        add_segment((1, 6), (1, 6 + w))
        add_segment((1, 6 + w), (0, 6 + w))
        add_segment((0, 6 + w), (0, 6))
    elif side == "S":
        add_segment((6, 6), (7, 6))
        add_segment((7, 6), (7, 6 - w))
        add_segment((7, 6 - w), (8, 6 - w))
        add_segment((8, 6 - w), (8, 6))
        add_segment((8, 6), (9, 6))
        add_segment((9, 6), (9, 6 - w))
        add_segment((9, 6 - w), (10, 6 - w))
        add_segment((10, 6 - w), (10, 6))
        add_segment((10, 6), (11, 6))
        add_segment((11, 6), (11, 6 - w))
        add_segment((11, 6 - w), (12, 6 - w))
        add_segment((12, 6 - w), (12, 6))

    return path


def get_variant_leg_meanders(
    i_sides: Tuple[str, ...],
    o_sides: Tuple[str, ...],
    vi: int,
    k_count: int,
) -> Dict[str, int]:
    """Map variant index vi to meander levels (0..5) for each active side."""
    n_in = len(i_sides)
    n_out = len(o_sides)
    meanders: Dict[str, int] = {}

    if n_in == 1 and n_out == 1:
        w = int(round(vi * 5.0 / max(1, k_count - 1)))
        meanders[i_sides[0]] = w
        meanders[o_sides[0]] = w
    elif n_in == 1 and n_out == 2:
        meanders[i_sides[0]] = vi % 4
        meanders[o_sides[0]] = (vi // 3) % 4
        meanders[o_sides[1]] = (vi + (vi // 2)) % 5
    elif n_in == 1 and n_out == 3:
        meanders[i_sides[0]] = vi % 4
        meanders[o_sides[0]] = (vi // 2) % 5
        meanders[o_sides[1]] = ((vi + 1) // 3) % 5
        meanders[o_sides[2]] = ((vi + 2) // 4) % 5
    elif n_in == 2 and n_out == 1:
        meanders[o_sides[0]] = vi % 4
        meanders[i_sides[0]] = (vi // 3) % 4
        meanders[i_sides[1]] = (vi + (vi // 2)) % 5
    elif n_in == 2 and n_out == 2:
        meanders[i_sides[0]] = vi % 4
        meanders[i_sides[1]] = (vi // 2) % 4
        meanders[o_sides[0]] = (vi // 3) % 5
        meanders[o_sides[1]] = (vi + 1) % 5
    else:  # 3in-1out
        meanders[o_sides[0]] = vi % 4
        meanders[i_sides[0]] = (vi // 2) % 5
        meanders[i_sides[1]] = ((vi + 1) // 3) % 5
        meanders[i_sides[2]] = ((vi + 2) // 4) % 5

    return meanders


def get_tile_leg_paths(
    i_sides: Tuple[str, ...],
    o_sides: Tuple[str, ...],
    variant_idx: int = 0,
    k_total: int = 6,
) -> Dict[str, List[Tuple[int, int]]]:
    """Return dictionary mapping side -> continuous 13x13 path from (6, 6) to border."""
    meanders = get_variant_leg_meanders(i_sides, o_sides, variant_idx, k_total)
    paths = {}
    for s in list(i_sides) + list(o_sides):
        paths[s] = make_leg_path_13x13(s, meanders[s])
    return paths


def make_tile_13x13_grid(
    i_sides: Tuple[str, ...],
    o_sides: Tuple[str, ...],
    variant_idx: int = 0,
    k_total: int = 6,
) -> np.ndarray:
    """Generate a (13, 13) integer matrix (1 = visited grid vertex, 0 = empty)."""
    grid = np.zeros((13, 13), dtype=int)
    paths = get_tile_leg_paths(i_sides, o_sides, variant_idx, k_total)
    for p in paths.values():
        for r, c in p:
            grid[r, c] = 1
    return grid


def render_maze_image(
    paths_dict: Dict[str, List[Tuple[int, int]]],
    variant: TileVariant,
    out_path: str,
    title_suffix: str = "",
) -> bool:
    """Render maze strictly by HIGHLIGHTING OVER 13x13 GRID LINES (no filled cell boxes)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)

    # 1. Draw 13x13 grid lines (r in 0..12, c in 0..12)
    for r in range(13):
        ax.plot([0, 12], [r, r], color="#d0d7de", linewidth=1.0, zorder=1)
    for c in range(13):
        ax.plot([c, c], [0, 12], color="#d0d7de", linewidth=1.0, zorder=1)

    # 2. Highlight fluid channels OVER the grid lines
    for side, path in paths_dict.items():
        xs = [pt[1] for pt in path]
        ys = [pt[0] for pt in path]
        ax.plot(xs, ys, color="#0d6efd", linewidth=5.0, solid_capstyle="round", zorder=3)

    # 3. Center junction marker J at (6, 6)
    ax.plot(6, 6, marker="o", markersize=14, color="#ffc107", markeredgecolor="#b28900", markeredgewidth=1.5, zorder=5)
    ax.text(6, 6, "J", ha="center", va="center", color="black", fontweight="bold", fontsize=8, zorder=6)

    # 4. Port markers along borders
    port_coords = {
        "N": (6, 0),
        "S": (6, 12),
        "W": (0, 6),
        "E": (12, 6),
    }

    for s in variant.inlet_sides:
        c, r = port_coords[s]
        ax.text(c, r, "IN", ha="center", va="center", color="white", fontweight="bold", fontsize=8,
                bbox=dict(boxstyle="square,pad=0.25", facecolor="#198754", edgecolor="none"), zorder=6)

    for s in variant.outlet_sides:
        c, r = port_coords[s]
        ax.text(c, r, "OUT", ha="center", va="center", color="white", fontweight="bold", fontsize=7.5,
                bbox=dict(boxstyle="square,pad=0.25", facecolor="#dc3545", edgecolor="none"), zorder=6)

    # Axis settings
    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xlim(-0.7, 12.7)
    ax.set_ylim(12.7, -0.7)  # row 0 at top (North)
    ax.set_aspect("equal")
    ax.tick_params(which="both", size=0, labelsize=8)

    r_strs = [f"{s}={v:.2e}" for s, v in sorted(variant.R_per_side.items())]
    r_info = ", ".join(r_strs) if r_strs else "R uncalibrated"
    title_str = f"{variant.variant_id} (13x13 Grid Lines)"
    if title_suffix:
        title_str += f" - {title_suffix}"

    ax.set_title(f"{title_str}\n{r_info}", fontsize=9.5, fontweight="bold", pad=10)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export 13x13 visual images for all library mazes")
    ap.add_argument("--out", type=str, default="results/library_visuals", help="Output directory")
    ap.add_argument("--library", type=str, default=None, help="Path to existing library.csv")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of images generated")
    args = ap.parse_args()

    lib = Library.from_csv(args.library) if args.library else Library.generate_initial_library()
    if args.limit:
        lib = Library(lib.variants[:args.limit])
    lib.save_visual_dataset(args.out)
