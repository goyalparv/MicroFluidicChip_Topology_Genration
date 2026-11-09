"""Visualization tool for Stage 1 microfluidic chip design.json outputs."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import DesignSpec
from .library import Library, get_tile_leg_paths


def render_board_ascii(spec_dict: dict, tiles: List[dict], open_ports: Optional[List[dict]] = None) -> str:
    """Render a clean ASCII diagram of the 8x8 chip layout."""
    n = spec_dict.get("grid_size", 8)
    inlet = tuple(spec_dict.get("inlet_cell", [0, 3]))
    outlets = {tuple(rc) for rc in spec_dict.get("outlet_cells", [])}
    board = {tuple(t["cell"]): t for t in tiles}
    defective_cells = {tuple(op["cell"]) for op in (open_ports or [])}

    lines = []
    lines.append("+" + "------+" * n)
    for r in range(n):
        row_str = "|"
        for c in range(n):
            mark = "*" if (r, c) in defective_cells else ""
            if (r, c) == inlet:
                if (r, c) in board:
                    t = board[(r, c)]
                    out_s = "".join(t["outlet_sides"])
                    sym = f"IN>{out_s}{mark}".center(6)
                else:
                    sym = f" IN {mark}".center(6)
            elif (r, c) in outlets:
                if (r, c) in board:
                    t = board[(r, c)]
                    in_s = "".join(t["inlet_sides"])
                    sym = f"{in_s}>OUT{mark}".center(6)
                else:
                    sym = f" OUT {mark}".center(6)
            elif (r, c) in board:
                t = board[(r, c)]
                in_s = "".join(t["inlet_sides"])
                out_s = "".join(t["outlet_sides"])
                sym = f"{in_s}>{out_s}{mark}"[:6].center(6)
            else:
                sym = " . ".center(6)
            row_str += sym + "|"
        lines.append(row_str)
        lines.append("+" + "------+" * n)
    return "\n".join(lines)


def render_chip_image(design_data: dict, out_png: str) -> None:
    """Render high-resolution 2D image of the complete 8x8 synthesized microfluidic chip."""
    spec = design_data.get("spec", {})
    tiles = design_data.get("tiles", [])
    metrics = design_data.get("metrics", {})
    realized_flows = design_data.get("realized_flows", [])

    grid_n = spec.get("grid_size", 8)
    inlet_cell = tuple(spec.get("inlet_cell", [0, 3]))
    outlet_cells = [tuple(rc) for rc in spec.get("outlet_cells", [])]
    target_flows = spec.get("target_flows", [])

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)

    # 1. Background grid
    scale = 12.0  # 12 units per tile cell
    for r in range(grid_n + 1):
        y = (grid_n - r) * scale
        ax.axhline(y, color="#cccccc", linestyle="--", linewidth=0.8)
    for c in range(grid_n + 1):
        x = c * scale
        ax.axvline(x, color="#cccccc", linestyle="--", linewidth=0.8)

    # 2. Draw placed tiles and serpentine paths
    for t in tiles:
        r, c = t["cell"]
        i_sides = tuple(t.get("inlet_sides", []))
        o_sides = tuple(t.get("outlet_sides", []))
        vid = t.get("variant_id", "")

        # Parse variant index and k_total
        if "variant_index" in t and "k_total" in t:
            vi = t["variant_index"]
            k_total = max(t["k_total"], 1)
        else:
            vi = 0
            if "_v" in vid:
                try:
                    vi = int(vid.split("_v")[-1])
                except ValueError:
                    vi = 0
            k_total = max(vi + 1, 12)

        # Dedicated leg paths in 13x13 space
        paths = get_tile_leg_paths(i_sides, o_sides, variant_idx=vi, k_total=k_total)

        # Origin of this cell in plot coordinates
        x0 = c * scale
        y0 = (grid_n - 1 - r) * scale

        # Center junction
        cx = x0 + 6.0
        cy = y0 + 6.0
        ax.plot(cx, cy, "s", markersize=8, color="#e67e22", zorder=4)

        # Plot each leg path
        for side, path_pts in paths.items():
            if len(path_pts) < 2:
                continue
            xs = [x0 + pt[1] for pt in path_pts]
            ys = [y0 + (12 - pt[0]) for pt in path_pts]
            ax.plot(xs, ys, color="#1a73e8", linewidth=3.0, solid_capstyle="round", zorder=3)

    # 3. Mark open ports with red crosses
    open_ports_list = list(design_data.get("closure", {}).get("open_ports", []))
    if not open_ports_list and tiles:
        from .closure import analyze_ports
        from .config import DesignSpec
        t_map = {
            tuple(t["cell"]): (tuple(sorted(t.get("inlet_sides", []))), tuple(sorted(t.get("outlet_sides", []))))
            for t in tiles
        }
        d_spec = DesignSpec(
            grid_size=grid_n,
            inlet_cell=inlet_cell,
            outlet_cells=outlet_cells,
            inlet_flow=spec.get("inlet_flow", 1e-4),
        )
        rep = analyze_ports(t_map, d_spec)
        open_ports_list = (
            [{"cell": list(c), "side": s, "kind": "open_inlet"} for c, s in rep.open_inlets] +
            [{"cell": list(c), "side": s, "kind": "pending"} for c, s in rep.pending] +
            [{"cell": list(c), "side": s, "kind": "blocked"} for c, s in rep.blocked_outlets]
        )

    for op in open_ports_list:
        c_r, c_c = op["cell"]
        side = op["side"]
        x0 = c_c * scale
        y0 = (grid_n - 1 - c_r) * scale
        if side == "N":
            px, py = x0 + 6.0, y0 + 12.0
        elif side == "S":
            px, py = x0 + 6.0, y0 + 0.0
        elif side == "W":
            px, py = x0 + 0.0, y0 + 6.0
        elif side == "E":
            px, py = x0 + 12.0, y0 + 6.0
        else:
            continue
        ax.plot(px, py, "x", markersize=10, markeredgewidth=2.5, color="#e74c3c", zorder=6)

    # 4. Highlight Inlet
    in_r, in_c = inlet_cell
    in_x = in_c * scale + 6.0
    in_y = (grid_n - in_r) * scale
    ax.plot(in_x, in_y, "v", markersize=14, color="#27ae60", zorder=5)
    ax.text(in_x, in_y + 2.0, "INLET\n$Q_{in}$", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#27ae60",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f8f5", edgecolor="#27ae60"))

    # 4. Highlight Outlets
    for j, (out_r, out_c) in enumerate(outlet_cells):
        out_x = out_c * scale + 6.0
        out_y = (grid_n - 1 - out_r) * scale
        ax.plot(out_x, out_y, "^", markersize=14, color="#c0392b", zorder=5)

        t_flow = target_flows[j] if j < len(target_flows) else 0.0
        r_flow = realized_flows[j] if j < len(realized_flows) else 0.0
        pct_err = abs(r_flow - t_flow) / max(t_flow, 1e-12) * 100.0 if t_flow > 0 else 0.0

        lbl = f"OUT {j+1}\nTarget: {t_flow:.2e}\nRealized: {r_flow:.2e}\nErr: {pct_err:.1f}%"
        ax.text(out_x, out_y - 2.0, lbl, ha="center", va="top",
                fontsize=8, fontweight="bold", color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fdebd0", edgecolor="#c0392b"))

    # Formatting
    e_q = metrics.get("E_Q", 0.0)
    success = metrics.get("success", False)
    tiles_count = len(tiles)

    title = f"Synthesized Microfluidic Chip (8x8 Grid)\nTiles: {tiles_count} | Flow Error $E_Q$: {e_q:.4f} | Success: {success}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=25)

    ax.set_xlim(-6, grid_n * scale + 6)
    ax.set_ylim(-16, grid_n * scale + 12)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"High-resolution chip layout saved to: {os.path.abspath(out_png)}")


def plot_training_curves(
    history: List[dict],
    out_dir_or_png: str,
    title_prefix: Optional[str] = None
) -> List[str]:
    """Generate separate standalone high-resolution learning curves showing RL training progress.

    Each metric family is saved as an individual PNG file:
    1. rl_rewards.png        - Mean step reward, mean total score, and best score
    2. rl_flow_error.png     - Mean hydraulic flow error E_Q and 5% success tolerance
    3. rl_tile_count.png     - Placed tile count vs n_min=17 skeleton and structural penalty
    4. rl_policy_entropy.png - Normalized entropy H_norm and max action probability
    """
    if not history:
        return []

    if out_dir_or_png.endswith(".png"):
        out_dir = os.path.dirname(os.path.abspath(out_dir_or_png))
    else:
        out_dir = os.path.abspath(out_dir_or_png)
    os.makedirs(out_dir, exist_ok=True)

    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]
    mean_reward = [h.get("mean_reward", 0.0) for h in history]
    mean_score = [h.get("mean_score", h.get("mean_reward", 0.0)) for h in history]
    best_reward = [h.get("best_reward", 0.0) for h in history]
    mean_eq = [h.get("mean_E_Q", 0.0) for h in history]
    mean_tiles = [h.get("mean_tiles", 0.0) for h in history]
    hnorm = [h.get("entropy_norm_mean", 0.0) for h in history]
    maxprob = [h.get("maxprob_mean", 0.0) for h in history]
    struct_cost = [h.get("mean_struct_cost", 0.0) for h in history]

    probe_eps = [h.get("episode") for h in history if "probe_mean_e_q" in h]
    probe_eq = [h.get("probe_mean_e_q") for h in history if "probe_mean_e_q" in h]

    pfx = f" ({title_prefix})" if title_prefix else ""
    saved_files = []

    # 1. Rewards & Scores Progression
    fig1, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    fig1.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#fafafa")
    ax1.grid(True, linestyle="--", alpha=0.6, color="#d0d0d0")
    ax1.plot(episodes, mean_reward, label="Mean Step Reward", color="#1976d2", alpha=0.75, lw=1.8)
    ax1.plot(episodes, mean_score, label="Mean Score (Rew - Struct)", color="#2e7d32", lw=2.2)
    ax1.plot(episodes, best_reward, label="Best Score Achieved", color="#e65100", lw=2.0, linestyle="-.")
    ax1.set_title(f"Reward & Score Progression{pfx}", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Episode", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Reward / Score", fontsize=11, fontweight="semibold")
    ax1.legend(loc="lower right", framealpha=0.95, fontsize=9.5, facecolor="#ffffff", edgecolor="#cccccc")
    plt.tight_layout()
    f1 = os.path.join(out_dir, "rl_rewards.png")
    fig1.savefig(f1, bbox_inches="tight")
    plt.close(fig1)
    saved_files.append(f1)

    # 2. Flow Error (E_Q) Convergence
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    fig2.patch.set_facecolor("#ffffff")
    ax2.set_facecolor("#fafafa")
    ax2.grid(True, linestyle="--", alpha=0.6, color="#d0d0d0")
    ax2.plot(episodes, mean_eq, label=r"Mean Flow Error $E_Q$", color="#d32f2f", lw=2.2)
    ax2.axhline(0.05, color="#2e7d32", linestyle="--", lw=1.8, label="Success Tolerance (5%)")
    if probe_eps and probe_eq:
        ax2.scatter(probe_eps, probe_eq, color="#7b1fa2", s=35, label=r"Probe Evaluation $E_Q$", zorder=4)
    ax2.set_title(rf"Hydraulic Flow Error $E_Q$ Convergence{pfx}", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Episode", fontsize=11, fontweight="semibold")
    ax2.set_ylabel(r"Relative Error $E_Q$", fontsize=11, fontweight="semibold")
    ax2.set_ylim(bottom=0.0)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9.5, facecolor="#ffffff", edgecolor="#cccccc")
    plt.tight_layout()
    f2 = os.path.join(out_dir, "rl_flow_error.png")
    fig2.savefig(f2, bbox_inches="tight")
    plt.close(fig2)
    saved_files.append(f2)

    # 3. Topology Compactness & Tile Usage
    fig3, ax3 = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    fig3.patch.set_facecolor("#ffffff")
    ax3.set_facecolor("#fafafa")
    ax3.grid(True, linestyle="--", alpha=0.6, color="#d0d0d0")
    line1 = ax3.plot(episodes, mean_tiles, label="Mean Placed Tiles", color="#0288d1", lw=2.2)
    line2 = [ax3.axhline(17, color="#f57c00", linestyle=":", lw=2.2, label=r"Min Skeleton ($n_{min}=17$)")]
    ax3_twin = ax3.twinx()
    line3 = ax3_twin.plot(episodes, struct_cost, label="Structural Cost Penalty", color="#7c4dff", lw=1.8, linestyle="--")
    ax3_twin.set_ylabel("Structural Penalty Cost", color="#7c4dff", fontsize=11, fontweight="semibold")
    ax3_twin.tick_params(axis="y", labelcolor="#7c4dff")
    ax3.set_title(f"Topology Compactness & Tile Usage{pfx}", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Episode", fontsize=11, fontweight="semibold")
    ax3.set_ylabel("Tile Count", fontsize=11, fontweight="semibold")
    all_lines = line1 + line2 + line3
    all_labels = [l.get_label() for l in all_lines]
    ax3.legend(all_lines, all_labels, loc="upper right", framealpha=0.95, fontsize=9.5, facecolor="#ffffff", edgecolor="#cccccc")
    plt.tight_layout()
    f3 = os.path.join(out_dir, "rl_tile_count.png")
    fig3.savefig(f3, bbox_inches="tight")
    plt.close(fig3)
    saved_files.append(f3)

    # 4. Policy Entropy & Exploration Health
    fig4, ax4 = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    fig4.patch.set_facecolor("#ffffff")
    ax4.set_facecolor("#fafafa")
    ax4.grid(True, linestyle="--", alpha=0.6, color="#d0d0d0")
    ax4.plot(episodes, hnorm, label=r"Normalized Entropy $H_{norm}$", color="#00796b", lw=2.2)
    ax4.plot(episodes, maxprob, label="Max Action Probability", color="#c2185b", lw=1.8, linestyle=":")
    ax4.axhline(0.10, color="#d32f2f", linestyle="--", lw=1.5, label="Collapse Alert Threshold (0.10)")
    ax4.set_title(rf"Policy Exploration & Entropy Health{pfx}", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Episode", fontsize=11, fontweight="semibold")
    ax4.set_ylabel("Normalized Index [0.0, 1.0]", fontsize=11, fontweight="semibold")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="upper right", framealpha=0.95, fontsize=9.5, facecolor="#ffffff", edgecolor="#cccccc")
    plt.tight_layout()
    f4 = os.path.join(out_dir, "rl_policy_entropy.png")
    fig4.savefig(f4, bbox_inches="tight")
    plt.close(fig4)
    saved_files.append(f4)

    for sf in saved_files:
        print(f"Standalone learning curve saved to: {sf}")
    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Visualize Stage 1 Synthesized Microfluidic Chip")
    parser.add_argument("--design", type=str, default="results/stage1_v2/design.json",
                        help="Path to design.json (default: results/stage1_v2/design.json)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output image path (default: <dir_of_design>/chip_layout.png)")
    args = parser.parse_args()

    if not os.path.exists(args.design):
        # Look in common fallback locations
        fallbacks = [
            "results/test_stage1_v2/design.json",
            "results/e2_R/design.json",
        ]
        found = None
        for fb in fallbacks:
            if os.path.exists(fb):
                found = fb
                break
        if found:
            print(f"Note: '{args.design}' not found. Using fallback: '{found}'")
            args.design = found
        else:
            raise SystemExit(f"Error: Could not find '{args.design}'. Please ensure design.json exists.")

    with open(args.design, "r", encoding="utf-8") as f:
        data = json.load(f)

    spec = data.get("spec", {})
    tiles = data.get("tiles", [])
    metrics = data.get("metrics", {})

    print("\n" + "=" * 50)
    print("       STAGE 1 SYNTHESIZED CHIP TOPOLOGY")
    print("=" * 50)
    open_ports = data.get("closure", {}).get("open_ports", [])
    print(render_board_ascii(spec, tiles, open_ports=open_ports))
    print("\nSummary Metrics:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v}")

    out_png = args.out or os.path.join(os.path.dirname(os.path.abspath(args.design)), "chip_layout.png")
    render_chip_image(data, out_png)


if __name__ == "__main__":
    main()

