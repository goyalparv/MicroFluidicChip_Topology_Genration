# Fixing Similar-Topology Outcomes: Step-by-Step Guide

PhysicsAwareRL\_v2, Stage 1. Handoff guide for the implementing engineer. The curriculum (warmup) fix is already done and is not touched here.

## 1. Purpose, assumptions, scope

**Problem.** Runs with different target ratios (equal split and 0.5/0.3/0.2) produced almost the same topology: 25 tiles (the full `max_tiles` budget), two straight trunk tiles, a full 3x5 block in rows 2 to 6 and columns 2 to 4, and the whole bottom row. Only the meander (resistance) variants differed.

**Assumptions (check them in Step 0).**

- The closure plan is implemented: the mask only allows actions that can still be closed, and the env has an oracle with `plan(...)`, `min_tiles_required()` and an `is_closed()` method. Names follow the closure plan; adapt them if your code differs.
- The curriculum fix is in place. This guide never edits `curriculum_v2.py`.
- The runtime fixes from the earlier document are optional. They make the experiments here faster, but nothing below depends on them.

**Must NOT change:** `agent_v2.py` (PolicyNet, ValueNet, rollouts, GAE), `FEATURE_DIM = 39`, the action space, `hydraulic_solver_v2.py`, `library_v2.py`, the closure rules, the existing dense reward terms (`cost`, `cost_core`, shaping, bonuses).

**Correction to the earlier objective document.** That document's presets (compact 0.5 + 0.2, low-pressure 0.2 + 0.5, tree 0.5 + 0.2 + 0.3) summed above its own safety limit. This guide replaces its structural-cost section, archive rules and reference-pressure procedure. Where the two documents differ, follow this one.

## 2. What 'fixed' can and cannot mean

Read this before implementing, because it decides what counts as success.

**Geometry forces most of the shape.** The inlet is at (0,3) and the outlets at (7,0), (7,3), (7,7). The minimum closed design has 17 tiles: six trunk tiles, a 1-in-3-out split at (6,3), a left arm of 4 tiles, a middle arm of 1 tile (the outlet tile) and a right arm of 5 tiles. The arms can take any monotone staircase route, which gives about 12 minimum topologies (3 left routes times 4 right routes; confirm the count with the oracle in Step 3).

**Targets change resistances, not shape.** In a tree, the flow of each branch is proportional to 1/R of that branch. With the baseline library one leg has R from 1 to 6, and a straight-through 1-in-1-out tile has path resistance 2 to 12. A 17-tile design can therefore meet moderate ratios by choosing meanders alone. Your earlier pre-closure 17-tile design reached E\_Q 0.0164 on the equal split.

**Consequence.** Once structure is no longer free, two different moderate targets will legitimately give the same occupied cells (up to route choice) and different resistance variants. Similar topology is a defect only when it comes from budget-filling, policy collapse or a selection artifact. Making designs more compact therefore makes them look more alike, not less.

**Outcomes this guide targets**

| ID | Outcome |
| --- | --- |
| O1 | No filler: tile count follows need (about `n_min` when the targets are feasible at the minimum skeleton, more only when physics demands it). |
| O2 | The reported design is the best by an explicit score among distinct successful designs, and an archive shows the alternatives. |
| O3 | Results are stable across seeds (success rate, tile count). |
| O4 | Target response is demonstrable: flow fractions and resistance choices follow the targets, and topology changes when the targets cannot be met at the minimum skeleton (extreme targets). |
| O5 | Design intent is explicit: the objective weights are chosen deliberately, with an optional robustness objective as an experiment. |

No criterion in this guide requires different occupied cells for different moderate targets.

## 3. Diagnosis recap

| # | Cause (ranked) | Evidence | Fixed in |
| --- | --- | --- | --- |
| 1 | Structure is nearly free in the reward: 17 to 25 tiles costs about 0.016 against a success bonus of 1.0, and extra tiles give extra tuning knobs. | Both images use 25 of 25 tiles; best reward equals 1.0 - 0.05 - E\_Q. | Step 6 |
| 2 | Best-of-N selection: the saved design is one lucky stochastic episode ranked by a structure-blind reward. | Mean E\_Q 0.12 at episode 2000 while best is 0.947. | Step 2 |
| 3 | Same seed in both runs (default `--seed 0`). | Run commands. | Step 4, Step 7 |
| 4 | Per-episode advantage normalization removes each episode's mean, so good and bad episodes push the policy equally. Training is short (250 updates). | `train_v2.py`. | Step 5 |
| 5 | Possible policy collapse (entropy never logged). | Runtime step-change. | Step 5 |
| 6 | Geometry forces most of the backbone. | Section 2. | Accepted, measured in Step 3 |
| 7 | Heavy potential shaping (weight 20) dominates until outlets are connected. | `config_v2.py`. | Not changed (see section 9) |

## 4. Metric definitions (used everywhere below)

| Name | Definition |
| --- | --- |
| `occ(board)` | frozenset of occupied cells |
| `topo_signature(board)` | sorted tuple of `(cell, sorted inlet sides, sorted outlet sides)`, variants ignored |
| `topo_id(board)` | first 12 hex characters of SHA-1 of `repr(topo_signature)` |
| `design_id(board)` | same, but built from `(cell, variant_id)` pairs |
| `n_min` | minimum number of tiles for a closed design (oracle, iterative deepening) |
| `extra` | `max(0, n_tiles - n_min)` |
| `loops` | cycle rank of the tile graph: `E - V + C` (V tiles, E matched adjacent connections, C connected components) |
| `merges` | number of tiles with two or more inlet sides |
| `p_ratio` | `p_in / p_ref`, where `p_ref` is the inlet pressure of the reference design (Step 1) |
| `jaccard(A, B)` | \` |
| `J_seed` | mean Jaccard between final designs of different seeds, same targets |
| `J_target` | mean Jaccard between final designs of different targets, same seed |

## Step 0. Prepare and protect the baseline

0.1 Tag the current commit. Copy the existing `results/` folders of the two runs. Record the Python version, the numpy version and the git hash in a text file.

0.2 Confirm the assumptions with asserts in a scratch script:

- `env.oracle.min_tiles_required() == 17` for the default spec;
- `env.is_closed()` exists and is False on an empty board;
- `FEATURE_DIM == 39` and `env.n_actions == 64 * len(library) + 1`;
- the existing unit tests pass.

0.3 Add every new behaviour behind a flag whose default equals the legacy behaviour:

- `--objective legacy` (all new structural weights 0);
- `--adv-norm episode` (the current per-episode normalization);
- no entropy schedule unless `--entropy-final` is given. The defaults are flipped only in Step 9, after validation.

0.4 Make every run traceable. At the start of `run_stage1_v2.main`, write `config.json` into the output folder with all command-line arguments, the final `RewardWeights`, the library path (or `generated`), the seed, the git hash and the start time.

**Gate:** all checks pass and a legacy run with the same seed reproduces the previous `history.json`.

## Step 1. Metrics module and reference values

### 1.1 New file `metrics_v2.py`

Pure functions with no dependency on the env.

```python
import hashlib
from .library_v2 import DIRECTION_DELTAS, OPPOSITE

def topo_signature(board):
    return tuple(sorted((cell, tuple(sorted(t.variant.inlet_sides)), tuple(sorted(t.variant.outlet_sides)))
                        for cell, t in board.items()))

def topo_id(board):
    return hashlib.sha1(repr(topo_signature(board)).encode()).hexdigest()[:12]

def design_id(board):
    sig = tuple(sorted((cell, t.variant.variant_id) for cell, t in board.items()))
    return hashlib.sha1(repr(sig).encode()).hexdigest()[:12]

def count_merges(board):
    return sum(1 for t in board.values() if len(t.variant.inlet_sides) >= 2)

def count_cycles(board):
    parent = {c: c for c in board}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = 0
    for (r, c), t in board.items():
        for s in t.variant.outlet_sides:
            dr, dc = DIRECTION_DELTAS[s]
            nb = (r + dr, c + dc)
            if nb in board and OPPOSITE[s] in board[nb].variant.inlet_sides:
                edges += 1
                parent[find((r, c))] = find(nb)
    comps = len({find(c) for c in board})
    return edges - len(board) + comps

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)
```

Expected values to unit-test: a straight chain gives `loops = 0`; a split into two branches that merge again (a diamond of 4 tiles) gives 1; a tree with a 1-in-3-out split gives 0; an empty board gives 0.

### 1.2 Reference values in `environment_v2.py` (no reward effect yet)

In `__init__`, right after the feasibility check:

1. `self.n_min = self.oracle.min_tiles_required()`. If your oracle lacks it, implement iterative deepening: for budget b from `len(needed cells)` up to `max_tiles`, call `plan({}, budget=b)` and return the first b that returns a plan.
2. `plan = self.oracle.plan({}, budget=self.n_min)`. The oracle's `plan` must accept an optional `budget` argument; add it (default `spec.max_tiles - len(tiles)`). This guarantees a plan with exactly `n_min` tiles.
3. For each `(cell, I, O)` in the plan, choose from the library the variant with that combo and the smallest `total_r`, breaking ties by `variant_id`. Build `ref_board`.
4. `sr = solve_network_v2(ref_board, spec)`. Require `sr.ok`, not `sr.stranded`, and all outlets connected; otherwise raise `ValueError`. Set `self.p_ref = sr.p_in` and check it is finite and positive.
5. `self.evaluator.set_reference(self.n_min, self.p_ref)`.
6. Recompute these whenever the library or `max_tiles` changes (they are computed per env instance, so this is automatic).

### 1.3 `reward_v2.py`

- `Evaluator.set_reference(n_min, p_ref)` stores both. Until it is called, the new fields below stay 0.
- Append these fields to `Evaluation` with defaults, after the existing defaults, so existing constructors still work: `n_tiles: int = 0`, `extra: int = 0`, `loops: int = 0`, `merges: int = 0`, `p_ratio: float = 0.0`.
- In `evaluate()`, after the solve, fill them: `n_tiles = len(board)`, `extra = max(0, n_tiles - n_min)`, `loops = count_cycles(board)`, `merges = count_merges(board)`, `p_ratio = sr.p_in / p_ref` (0 if the solve failed or is stranded).
- Add the five keys to `metrics()`.
- These fields are diagnostics only. They must not enter `cost`, `cost_core` or `reward`.

### 1.4 Tests

`count_cycles` and `count_merges` on hand-built boards; `topo_id` ignores variants and is stable under dictionary order; `ref_board` is closed, has `n_min` tiles and a positive finite `p_ref`; `evaluate()` output for `cost` and `reward` is unchanged on 200 random boards.

**Gate:** the evaluator's `cost`, `cost_core` and `reward` are bit-identical to before on a fixed set of boards.

## Step 2. Archive, selection and fixed evaluation (no reward change)

This step fixes the best-of-N problem and gives every later step an honest measurement.

### 2.1 Eligibility and score

- A final board is **eligible** if it is closed and `ev.success` is true.
- `score = ev.reward - ev.struct_cost`. Add `struct_cost: float = 0.0` to `Evaluation` now (it stays 0 until Step 6) and a property `score`.
- Ranking key, lowest first: `(-score, n_tiles, p_in, design_id)`. This is a total order, so ties are deterministic.

### 2.2 Archive (`train_v2.py`)

- Keep an archive of at most K designs (`--archive-k`, default 5), at most one per `topo_id`.
- On every finished training episode that is eligible: if its `topo_id` is already present, replace that entry only if the new key is better; otherwise add it if there is space; otherwise replace the worst entry if the new key is better.
- Each entry stores: the tile list (as in `design.json`), the metrics (`ev.metrics()`), `score`, `topo_id`, `design_id`, episode index, seed.
- `best_board` and `best_reward` are replaced by the first entry of the archive. If the archive is empty at the end, fall back to the board with the highest `score` and mark it `eligible: false`.
- This applies to PPO and REINFORCE. REINFORCE has no stages, so no stage filter exists there.

### 2.3 Outputs (`run_stage1_v2.py`)

- `archive.json`: the sorted list of entries.
- `design.json`: rank 1, with the existing schema plus `topo_id`, `design_id`, `score`, `eligible`, `n_min`, `p_ref` and the new metrics.
- `design_rank2.json` and so on for the other entries. The chip image is rendered for rank 1 only.

### 2.4 Fixed evaluation after training

Run `--eval-episodes` (default 200) stochastic episodes plus 1 greedy episode, all from `env.reset()` (empty board), using a dedicated generator `np.random.default_rng(seed + 1000)` so the training stream is untouched. Use the trained policy without updating it. Per episode record `success`, `n_tiles`, `e_q`, `extra`, `loops`, `p_ratio`, `topo_id`, `design_id`. Write `eval_summary.json` with: `success_rate`, `mean_tiles`, `std_tiles`, `frac_at_budget` (share of episodes with `n_tiles == max_tiles`), `mean_extra`, `mean_loops`, `mean_p_ratio`, `n_distinct_topo`, the three most frequent `topo_id` values with their counts, and the greedy episode's metrics. Eligible evaluation designs may update the archive.

### 2.5 Tests

Archive replacement rules (same `topo_id`, better and worse; full archive); deterministic ordering with ties; evaluation does not change the training generator (main history identical with evaluation on or off); `eval_summary.json` has every key.

**Gate:** a legacy run produces `archive.json` and `eval_summary.json`, and `design.json` is eligible or marked ineligible correctly.

## Step 3. Reference-optimum tool (ground truth for the minimum skeleton)

This tool answers one question exactly: for a given target vector, how accurate can a minimum-size design be? It tells you whether extra tiles are ever needed, and it gives the RL result something to be compared with.

New file `tools/reference_optimum.py`. Supported geometry: one inlet, one outlet in the inlet column, one outlet to its left and one to its right on the bottom row (the default spec). For other specs print 'unsupported spec' and exit.

### 3.1 Model

The skeleton is a tree, so the split tile's center node has one common pressure and the three arms are independent resistances. With `R_i` the resistance from the split center to ground through arm `i`:

`Q_i = Q_in * (1 / R_i) / sum_j (1 / R_j)`

`R_i` = (the split tile's leg toward arm `i`) + (sum over the arm's tiles of that tile's `R_in + R_out`) + `R_stub`. The last term is the outlet stub to ground, as in the solver. The trunk resistance does not affect the ratio.

### 3.2 Enumeration

1. Enumerate the monotone staircase routes of the left and right arms with the same tile counts as the minimum design (4 and 5). The middle arm is the outlet tile alone. Print the number of routes; expect 12 in total.
2. For each route, the per-tile combo is fixed (turn or straight). Collect, from the library, the set of path resistances `total_r` for each combo and build the set of achievable arm sums with a small dynamic program. Round sums to 4 significant digits to keep the sets small.
3. For every variant of the split tile (the library combo with inlet N and outlets W, E, S) take its three leg resistances.
4. For every split variant and every pair of achievable left and right sums (and the middle sums), compute the flow fractions, the relative errors `(Q - Q*) / max(Q*, 0.01 * Q_in)` and `e_q` exactly as `Evaluator` does. Use numpy outer products. Keep the best.
5. Output: best `e_q`, per-outlet relative errors, whether `success` holds at `weights.tolerance`, the chosen variant ids, and a flag `needs_extra_tiles = not success`.

### 3.3 Cross-check (mandatory)

Build the best design as a real board (`PlacedTile` objects with the chosen variants, trunk tiles with the minimum-resistance straight variant), run `solve_network_v2`, and assert that the realized flow fractions match the tool's fractions to within 1e-9. If they do not, the model in 3.1 is wrong; fix the tool before using any of its numbers.

### 3.4 Run it

Run for the targets of interest: equal split, 0.5/0.3/0.2, 0.2/0.3/0.5, 0.8/0.1/0.1, 0.1/0.8/0.1, 0.1/0.1/0.8. Save `reference_optimum.csv`. Targets with `needs_extra_tiles = True` are the ones for which a compact objective must still allow extra tiles.

**Gate:** the cross-check passes, and the table exists.

## Step 4. Phase A: diagnose before changing the reward

The causes in section 3 are inferences. These runs test them with the legacy objective. Use `--episodes 5000 --batch 16`, targets equal and 0.5/0.3/0.2, and the metrics from Steps 1 and 2.

| Run | Settings | Purpose |
| --- | --- | --- |
| A1 | `max_tiles` 25, seeds 0 to 4, both targets (10 runs) | Baseline: tile usage, success rate, `J_seed`, `J_target`, archive diversity |
| A2 | `max_tiles` 17, seeds 0 to 2, both targets | Is a minimum design reachable at all, and with which success rate? |
| A3 | `max_tiles` 21, seeds 0 to 2, both targets | Does tile usage follow the budget? |

Compute `J_seed` and `J_target` with a small script `tools/compare_designs.py` that reads the rank-1 `design.json` of every run.

### Decision rules

| Hypothesis | Confirmed when | Meaning |
| --- | --- | --- |
| H-free: structure is free | In A1 and A3, `frac_at_budget` is at least 0.8 in most runs, and A2 reaches a success rate above 0 with 17 tiles | The budget is spent because nothing discourages it. Step 6 is needed. |
| H-seed: sameness is luck | `J_target` (same seed) is clearly larger than `J_seed` (same targets) | Runs resemble each other because of the shared seed. Always compare different seeds (Step 7). |
| H-geometry: shape is forced | `J_seed` and `J_target` are both above 0.9 at the minimum budget (A2) | Similar topology at the minimum is expected, not a defect. |
| H-select: selection hides diversity | `archive.json` holds three or more distinct `topo_id` values with success | Rank-by-score will reveal them. |
| H-collapse: policy collapse | After Step 5 adds entropy logging: normalized entropy below 0.1 early in training | Apply the entropy schedule (Step 5). |

Write the results into `results/phaseA/summary.md` with the numbers. If H-free is not confirmed, stop and report before continuing, because Step 6 would then address the wrong cause.

## Step 5. Learning-signal fixes

These make the policy learn from the difference between good and bad episodes and keep exploration alive. They change `train_v2.py` and `run_stage1_v2.py` only.

### 5.1 Batch-level advantage normalization (`--adv-norm batch`)

Current behaviour: inside the rollout loop, each episode's GAE advantages are normalized to mean 0 and std 1 on their own. This removes the episode's mean, so a poor episode and a good one push the policy equally hard.

New behaviour (when `--adv-norm batch`):

1. In the rollout loop, keep for every episode `(traj, raw_adv, returns)`; do not normalize yet.
2. After all `rollouts_per_update` episodes: `all_adv = np.concatenate([raw_adv ...])`, `mu = all_adv.mean()`, `sd = all_adv.std()`.
3. If `len(all_adv) >= 2` and `sd > 1e-8`: `adv_norm = (raw_adv - mu) / (sd + 1e-8)`; otherwise `adv_norm = raw_adv - mu`.
4. Build the PPO buffer from the episodes in the same order as before, using the batch-normalized values. The returns (value targets) stay unnormalized.
5. Nothing here consumes random numbers, so with the same seed the rollouts are identical; only the update differs.

Keep `--adv-norm episode` as the exact legacy path.

### 5.2 Entropy schedule

Flags: `--entropy` (start value, default 0.01 as now) and `--entropy-final` (default None, meaning a constant coefficient). With `n_updates` updates and zero-based update index `u`:

`entropy_coef(u) = e_final + (e_start - e_final) * (1 - u / max(n_updates - 1, 1))`

Pass `entropy_coef(u)` into both the policy-gradient accumulation and the entropy gradient of that update. Suggested experiment values: start 0.03, final 0.003.

### 5.3 Entropy logging

After each update's rollouts, from the stored `action_p` of every step (no change to `agent_v2.py`):

- `H = -sum(p * log(p + 1e-12))`;
- `H_norm = H / log(m)` for steps with `m > 1` valid actions (skip steps with `m == 1`);
- `maxprob = max(p)`. Add the means as `entropy_mean`, `entropy_norm_mean`, `maxprob_mean` to `history`, and print `entropy_norm_mean` on each log line. Print a warning when `entropy_norm_mean < 0.10` on five consecutive log lines.

### 5.4 Training length and seeds

- Use at least 5000 episodes and `--batch 16` for every comparison run. No code change.
- The seed must reach: `PolicyNet(seed)`, `ValueNet(seed + 1)`, the training generator, the evaluation generator (`seed + 1000`) and any other stream you added. Never compare runs that share a seed unless the comparison is explicitly paired.

### 5.5 Tests

- Batch normalization: the normalized batch has mean 0 and std 1; the difference between two episodes' mean advantages equals the raw difference divided by the batch std.
- The legacy flags (`--adv-norm episode`, no `--entropy-final`) reproduce the previous `history.json` for the same seed.
- The entropy schedule returns `e_start` at `u = 0` and `e_final` at the last update, and is monotone.
- `H_norm` is 1 for a uniform distribution and 0 for a one-hot distribution.

### 5.6 Gate

Run seeds 0 to 2, both targets, 5000 episodes, with `--adv-norm batch --entropy 0.03 --entropy-final 0.003`, and compare with the A1 runs on the fixed evaluation (paired by seed). Accept if the mean evaluation success rate is at least the legacy value minus 0.05 and no collapse warning appears. If the success rate drops, rerun with `--adv-norm batch` only (no schedule) and keep whichever variant passes. Record the result.

## Step 6. Structural objective terms

This step changes what is optimized. Do it only after Step 4 confirms H-free.

### 6.1 New weights (`config_v2.py`, `RewardWeights`)

| Field | Default | Meaning |
| --- | --- | --- |
| `lam_extra_tiles` | 0.0 | Penalty for tiles beyond `n_min` |
| `lam_pressure` | 0.0 | Penalty for inlet pressure above the reference design |
| `lam_loops` | 0.0 | Penalty for cycles in the tile graph |
| `p_clip` | 20.0 | Cap on `p_ratio` before the log |
| `loops_cap` | 8 | Cap on loop count for normalization |

**Weight budget rule (corrected).** A near-success failing design can earn `partial_success_bonus` (0.3) at no structural cost. A successful design with every structural term at its cap pays the sum of the weights. Success must always win by a margin of at least 0.1, so:

`lam_extra_tiles + lam_pressure + lam_loops <= success_bonus - partial_success_bonus - 0.1`

With the defaults this limit is 0.6. Implement `RewardWeights.check_structural_budget()`, call it at env construction, and raise `ValueError` when the sum exceeds the limit. Add `--allow-large-struct` to bypass it with a printed warning (needed only if `success_bonus` is raised).

### 6.2 Presets (`--objective`)

| Preset | `lam_extra_tiles` | `lam_pressure` | `lam_loops` | Sum |
| --- | --- | --- | --- | --- |
| `legacy` | 0 | 0 | 0 | 0 |
| `compact` | 0.30 | 0.15 | 0 | 0.45 |
| `low-pressure` | 0.15 | 0.30 | 0 | 0.45 |
| `tree` | 0.30 | 0.15 | 0.10 | 0.55 |

The three weights also get individual flags that override the preset. Recommended starting point: `compact`.

### 6.3 Formulas

All use the quantities added in Step 1 (`n_tiles`, `loops`, `p_ratio`) and are evaluated on the final board.

- `extra_n = 0` if `max_tiles <= n_min`, else `clip((n_tiles - n_min) / (max_tiles - n_min), 0, 1)`
- `press_n = 1` if the solve failed or is stranded; else `0` if `p_ratio <= 1`; else `clip(log(p_ratio) / log(p_clip), 0, 1)`
- `loops_n = min(max(loops, 0), loops_cap) / loops_cap`
- `struct_cost = lam_extra_tiles * extra_n + lam_pressure * press_n + lam_loops * loops_n`

In `Evaluator.evaluate`, store `struct_cost` and a dict `struct_parts` (`extra_n`, `press_n`, `loops_n`) on the `Evaluation`. The components are cheap, so compute them on every call; they carry meaning only for closed boards. Add `struct_cost` and the three parts to `metrics()`. The `score` property is `reward - struct_cost`.

### 6.4 Environment integration (`environment_v2.py`, `step()`)

- When `done` is true, whether by STOP or by reaching `max_tiles`: `reward -= ev.struct_cost`. Place it after the success and partial-success bonuses. Apply it exactly once.
- Apply the same subtraction in the non-dense branch (`reward = -ev.cost if done else 0.0`).
- Do not add `struct_cost` to `cost` or `cost_core`. The telescoping dense reward must stay exactly as it is.
- Do not apply anything on non-terminal steps. A terminal-only penalty is deliberate: partial boards have low pressure and few loops, so a dense version would distort credit assignment (see Step 8a for the dense alternative).
- With all three weights at 0 nothing changes.

### 6.5 Training and archive

The archive and `best_board` already rank by `score` (Step 2). Add `mean_score` and the mean of each struct part to `history`.

### 6.6 Tuning guide

| Symptom after a run (fixed evaluation) | Action |
| --- | --- |
| `mean_extra` barely falls versus legacy | First check that the weights reached the env (`config.json`). Then try Step 8a (dense tile potential). Do not raise the sum above 0.6. |
| Success rate falls more than 0.10 below legacy | Compare with `reference_optimum.csv`. If the targets need extra tiles, halve `lam_extra_tiles`. If not, the policy needs more training: more episodes, or the Step 5 schedule. |
| `p_ratio` stays high | Raise `lam_pressure` toward 0.30 while lowering `lam_extra_tiles` to keep the sum within 0.6. |
| Loops persist and are unwanted | Use `tree`. |
| Designs look identical across moderate targets | Expected (section 2). Check resistance variants and flow fractions instead of occupancy. |

### 6.7 Tests

- Hand-computed `extra_n`, `press_n`, `loops_n` and `struct_cost` for known boards; `press_n = 1` for a stranded solve; `extra_n = 0` when `max_tiles == n_min`.
- With all weights 0, rewards, `history` and `design.json` match the previous version for the same seed.
- For the same action sequence, per-step rewards under `legacy` and `compact` are identical on every non-terminal step, and differ on the final step by exactly `struct_cost`.
- The penalty is applied once for an episode ending by STOP and once for one ending by `max_tiles`.
- `check_structural_budget` raises for a sum above the limit and passes at the limit.
- `cost` and `cost_core` are unchanged by the weights.

### 6.8 Gate

Short runs (seeds 0 to 2, both targets, 5000 episodes) with `--objective compact` and the Step 5 flags show `mean_extra` clearly below the legacy value and a success rate within 0.10 of legacy. If not, use the tuning guide before continuing.

## Step 7. Experiments and tuning

Run after Steps 5 and 6 pass their gates. Every run uses 5000 episodes, `--batch 16`, a distinct seed per repetition, and the fixed evaluation of Step 2.

### 7.1 Matrix

| Phase | Settings | Purpose |
| --- | --- | --- |
| B | Legacy objective plus Step 5 flags; seeds 0 to 4; both targets | Effect of the learning-signal fixes alone |
| C1 | `compact` plus Step 5 flags; seeds 0 to 4; both targets | Main candidate |
| C2 | `low-pressure`, same seeds and targets | Pressure effect |
| C3 | `tree`, same seeds and targets | Loop effect |
| D | `compact`; targets 0.8/0.1/0.1, 0.1/0.8/0.1, 0.1/0.1/0.8; seeds 0 to 2 | Target response: compare with `reference_optimum.csv` |
| E (optional) | Only `--adv-norm batch`; only the entropy schedule; only `compact` | Attribute improvements to one change |

### 7.2 Report for every phase (mean and standard deviation over seeds)

| Column | Source |
| --- | --- |
| `success_rate` | `eval_summary.json` |
| `mean_tiles`, `frac_at_budget`, `mean_extra` | `eval_summary.json` |
| `mean_loops`, `mean_p_ratio` | `eval_summary.json` |
| `n_distinct_topo` and the top `topo_id` counts | `eval_summary.json` |
| `J_seed`, `J_target` | `tools/compare_designs.py` on the rank-1 designs |
| `gap` | best archive `e_q` minus the reference-optimum `e_q` for the same targets (moderate targets only) |
| Entropy curve | `history.json` |

### 7.3 Interpretation rules

- A shrinking `mean_extra` with a stable success rate means the filler is gone (O1).
- Rising `J_target` under `compact` is expected for moderate targets. Judge target response in phase D, not by occupancy at moderate targets.
- In phase D, for each target compare the tool's `needs_extra_tiles` with the observed `mean_extra`: targets that the tool says are infeasible at the minimum skeleton must produce designs with `extra > 0` and success; feasible targets should give `extra` close to 0.
- Compare `compact` with `legacy` only on paired seeds.
- If the success rate falls under every preset, return to Step 5 (training signal) before changing the weights again.

## Step 8. Optional upgrades (only if Step 7 shows a need)

### 8a. Dense tile potential

Use it when `mean_extra` does not fall enough under `compact` even though the weights are right (a credit-assignment problem: the penalty arrives only at the end of about 23 steps).

- Flag `--tile-potential dense` (default `terminal`).
- For a state `s`, `proj(s) = len(board) + L(s)`, where `L(s)` is the exact minimum number of tiles needed to close `s`. Compute it with iterative deepening on the oracle, from a lower bound up to the remaining budget, and cache it by the state key.
- `phi_t(s) = -lam_extra_tiles * clip((proj(s) - n_min) / (max_tiles - n_min), 0, 1)`.
- Add `phi_t(s_next) - phi_t(s)` to the reward of every step. This is the same undiscounted difference convention the existing shaping uses.
- When dense mode is on, set the `extra_n` part of the terminal `struct_cost` to 0, so the extra-tile cost is not counted twice. Pressure and loop terms stay terminal.
- At a terminal state `L = 0`, so `proj = n_tiles` and the shaping sum equals `-lam_extra_tiles * extra_n` minus a constant. Pre-placed trunk tiles lie on every minimum design, so the start potential is 0.
- If the search returns UNKNOWN (node limit), use the length of the oracle's plan as a fallback and count it. Require zero such cases in the acceptance run.
- Test: on random episodes, the sum of the shaping terms equals `phi_t(s_T) - phi_t(s_0)` to 1e-9.

### 8b. Robustness objective (an experiment, not a default)

A real reason to prefer redundancy or particular topologies is tolerance to fabrication error. This term scores how well the flow split survives random errors in the channel resistances.

- Flags `--lam-robust` (default 0), `--robust-eps` (default 0.10), `--robust-samples` (default 8).
- At a terminal state, build `robust-samples` perturbed copies of the board. For every tile and every active leg, multiply the leg resistance by `1 + u`, with `u` uniform in `[-eps, eps]`. Create the perturbed variant with `dataclasses.replace(variant, R_per_side=new_map)`; the solver is not modified.
- Seed the generator with the first 8 hex digits of `design_id(board)` converted to an integer, so the score is a deterministic function of the board.
- For each copy run `solve_network_v2` and compute `e_q` with the evaluator's own denominators and clipping. `robust_e_q` is the mean over the copies.
- `robust_n = min(robust_e_q, 0.5) / 0.5`; the cost is `lam_robust * robust_n`, added to `struct_cost` (terminal only). The weight budget rule of Step 6 then covers four weights.
- Cost: 8 extra solves per episode, a few milliseconds.
- Add `robust_e_q` to `metrics()` when the term is active.
- No result is promised. Run it as a separate experiment and report whether the preferred topologies differ from the `compact` ones.

## Step 9. Finalize

9.1 Flip the defaults only after the acceptance criteria below are met on your data: `--adv-norm batch`, `--objective compact` (or the preset that won), `--eval-episodes 200`, `--archive-k 5`. Keep `--entropy-final` unset unless the Step 5 gate showed a benefit. Keep every legacy flag working.

9.2 Update the README: new files (`metrics_v2.py`, `tools/reference_optimum.py`, `tools/compare_designs.py`), the new flags, the new output files (`config.json`, `archive.json`, `design_rank*.json`, `eval_summary.json`) and the meaning of the metrics.

9.3 Re-run the full unit-test suite and the legacy-equivalence tests. Tag the commit.

## 10. Test summary

| ID | Test | Step |
| --- | --- | --- |
| T1 | `count_cycles`, `count_merges`, `topo_id`, `design_id` on hand-built boards | 1 |
| T2 | `ref_board` closed with `n_min` tiles; `p_ref` finite and positive | 1 |
| T3 | `cost`, `cost_core` and `reward` unchanged after adding diagnostics | 1 |
| T4 | Archive rules and deterministic ordering | 2 |
| T5 | Evaluation does not change the training random stream | 2 |
| T6 | Reference-optimum tool matches `solve_network_v2` to 1e-9 | 3 |
| T7 | Batch advantage normalization properties | 5 |
| T8 | Entropy schedule and `H_norm` values | 5 |
| T9 | Legacy flags reproduce the old `history.json` | 5, 6 |
| T10 | Struct formulas, once-only terminal application, non-terminal rewards identical | 6 |
| T11 | `check_structural_budget` | 6 |
| T12 | Dense potential telescoping identity (if 8a is built) | 8a |
| T13 | Robust score is deterministic for a given board (if 8b is built) | 8b |

## 11. Acceptance criteria

| ID | Criterion |
| --- | --- |
| AC1 | Phase A is documented with numbers, and the verdict on H-free, H-seed, H-geometry, H-select and H-collapse is stated. |
| AC2 | With `compact` (phase C1), the median `n_tiles` over seeds and targets is at most `n_min + 2` for targets that the reference tool marks feasible at the minimum skeleton, and the success rate is within 0.05 of the legacy value. |
| AC3 | With `tree` (phase C3), `loops = 0` in at least 80% of evaluation designs. |
| AC4 | With `low-pressure` (phase C2), the median `p_ratio` is at least 20% lower than legacy. If it is not, report the number and the reason. |
| AC5 | For moderate targets, the best archive design reaches an `e_q` within 0.02 of the reference optimum, or meets the tolerance. |
| AC6 | The standard deviation across seeds of the success rate is at most 0.15. |
| AC7 | In phase D, every target marked infeasible at the minimum skeleton yields successful designs with `extra > 0`, and feasible targets stay near `extra = 0`. |
| AC8 | Every run folder contains `config.json`, `archive.json`, `eval_summary.json`; legacy flags still reproduce the old behaviour. |

No criterion requires different occupied cells for different moderate targets.

## 12. Mistakes to avoid

1. Do not add `struct_cost` to `cost` or `cost_core`; it would change the telescoping dense reward.
2. Do not apply the structural penalty twice (terminal plus dense potential). In dense mode the terminal `extra_n` part is zero.
3. Apply the terminal penalty for both ways an episode can end (STOP and `max_tiles`), exactly once.
4. Do not compare runs that share a seed unless the comparison is explicitly paired.
5. Do not judge quality by the best of N training episodes. Use the fixed evaluation.
6. Do not change `FEATURE_DIM`, the action space or `agent_v2.py`.
7. Recompute `n_min` and `p_ref` for every library and spec; never reuse them across libraries.
8. Build the reference plan with budget exactly `n_min`; a plan from the full budget may be longer and would inflate `p_ref`.
9. Do not interpret `press_n`, `loops_n` or `extra_n` on partial boards.
10. Keep the weight sum within the budget rule; check it at construction.
11. Use sorted tuples in every signature, so hashes are stable.
12. Keep separate random streams for training, evaluation and any perturbation, so adding diagnostics never changes training.
13. Compute `gap` only for targets that the reference tool supports, and cross-check the tool against the real solver first.
14. Flip defaults last, after the acceptance criteria are met.
15. Do not expect different occupied cells for different moderate targets as a success signal.

## 13. Deliberately not changed

- The shaping weight (20) and the mismatch between the undiscounted shaping difference and gamma 0.99. It biases mildly toward connecting outlets sooner and does not explain the similar topologies.
- `lam_c` (0.05 times tiles over `max_tiles`) and the success tolerance of 5%.
- The `is_newly_placed_dead` behaviour on outlet tiles (a spurious -0.1 per correct outlet tile, a constant offset for successful designs). It is a separate item; apply it only if you approve the reward change.

## 14. Files touched

| File | Changes |
| --- | --- |
| `metrics_v2.py` | New: signatures, `count_cycles`, `count_merges`, `jaccard` |
| `environment_v2.py` | `n_min`, `p_ref`, `ref_board`; terminal `struct_cost`; dense potential (8a) |
| `reward_v2.py` | `set_reference`, new `Evaluation` fields, `struct_parts`, `score`, robust error (8b) |
| `config_v2.py` | New `RewardWeights` fields, `check_structural_budget` |
| `train_v2.py` | Archive, batch advantage normalization, entropy schedule and logging, fixed evaluation |
| `run_stage1_v2.py` | New flags, `config.json`, `archive.json`, `design_rank*.json`, `eval_summary.json` |
| `tools/reference_optimum.py`, `tools/compare_designs.py` | New |
| `tests/` | T1 to T13 |

Not changed: `agent_v2.py`, `curriculum_v2.py`, `hydraulic_solver_v2.py`, `library_v2.py`, the closure rules, `FEATURE_DIM`, the action space.

## 15. Open decisions

- **Which preset.** `compact` is the recommended start; the weights are starting values, not tuned.
- **Is occupancy diversity actually wanted?** If yes for moderate targets, the objective must reward something other than compactness (for example robustness, Step 8b), otherwise sameness is the correct answer.
- **Archive size and evaluation length.** K = 5 and 200 episodes are defaults; raise them if distinct designs are rare.
- **Dense potential.** Build 8a only if Step 7 shows the terminal penalty is too slow to take effect.
