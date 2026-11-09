# Closed-Topology Enforcement: Implementation Plan

PhysicsAwareRL\_v2, Stage 1. Handoff document for the implementing engineer.

## 1. Goal and scope

**Problem.** The synthesized chip contains dangling legs (open ends): tile legs that lead nowhere. The solver treats them as sealed dead ends carrying zero flow, and the reward cannot see them, so the agent adds them freely.

**Requirement.** No open ends in any episode's final topology.

**What 'complete' means.** Tiles are placed one at a time, so mid-episode a tile always has outlet legs facing empty cells (the growth frontier). That cannot be avoided without changing the action space. The guarantee delivered by this plan is:

1. At every step there is never an open inlet and never a blocked or illegal outlet.
2. Every pending (frontier) port is provably closable within the remaining `max_tiles` budget.
3. The final topology of every episode, ending by STOP or by reaching `max_tiles`, has zero open ports, and every chip outlet is placed and connected.

**Must NOT change:**

- `agent_v2.py` (PolicyNet, ValueNet, PPO rollout, GAE)
- `FEATURE_DIM = 39` and the feature layout
- action space size (`64*640 + 1`)
- reward formulas and `RewardWeights`
- `hydraulic_solver_v2.py` and `library_v2.py`

**May change (minimally):** `environment_v2.py` (mask, new helpers), `curriculum_v2.py`, small guards in `train_v2.py`, `run_stage1_v2.py`, `visualize_stage1.py`. One new file: `closure_v2.py`. One new test file: `tests/test_closure_v2.py`.

## 2. Root cause (current code)

The example output has 17 tiles, which is exactly the minimum for the default spec (trunk down column 3, one split, arms along rows 6 and 7). Every open end is an outlet leg facing an empty cell. Four code paths allow this:

1. **STOP gating.** `valid_action_mask` enables STOP once all chip outlets are placed and connected. It never checks other legs.
2. **Free extra legs.** A dead-end leg carries zero flow, so `e_q`, `conn_progress`, `balance_score` and shaping cannot see it. Multi-outlet variants give more resistance variety at no cost.
3. **Inlets are only a subset check.** `incoming_flow_sides.issubset(v.inlet_sides)` lets a 2-in or 3-in tile be placed with an inlet facing an empty cell. This can never be repaired later, because no outlet may point into a placed tile. The first-tile rule has the same hole (it only requires `'N' in inlet_sides`).
4. **Termination.** `done = len(board) >= max_tiles` ends the episode regardless of pending ports.

The soft penalties do not help. `count_isolated_tiles` only counts leaf tiles with no downstream neighbour at all. `is_newly_placed_dead` returns False whenever any outlet points at an empty cell, which is always true for a fresh tile.

## 3. Definitions

For each placed tile at cell `c` and each active side `s`, let `n` be the neighbour across `s`.

**Inlet side**

- OK (chip inlet): `c == inlet_cell` and `s == 'N'`.
- OK (matched): `n` is placed and `OPPOSITE[s]` is in `n`'s outlet sides.
- Otherwise: **OPEN\_INLET**. Must never exist at any step.

**Outlet side**

- OK (chip outlet): `c` in `outlet_set` and `s == 'S'`.
- OK (matched): `n` placed and `OPPOSITE[s]` in `n`'s inlet sides.
- **PENDING**: `n` in bounds and empty. Legal mid-episode only.
- **BLOCKED**: `n` out of bounds, or placed but not accepting. Must never exist.

**Closed topology**: no OPEN\_INLET, no PENDING, no BLOCKED, and all chip outlet cells are placed.

**Facts about the existing rules that this plan relies on**

- A new tile can never export into a placed tile, so the flow graph is always a DAG.
- Once a chip outlet cell is placed, nothing can join it later (it was placed with its exact inlet set).
- Hence pending ports can only terminate at unplaced chip outlet cells, directly or through merges.

**Hard invariants** (checked in tests and at the end of every episode)

| ID | Invariant |
| --- | --- |
| I1 | Every tile side is a chip port, or matched in direction (outlet to inlet). |
| I2 | Boundary-facing sides are only the inlet tile's N and the outlet tiles' S. |
| I3 | No pending, blocked or open sides. |
| I4 | All tiles are reachable from the inlet. |
| I5 | All chip outlet cells are occupied and connected. |
| I6 | `len(board) <= max_tiles`. |
| I7 | Geometric, solver-independent: every fine-grid port node has resistor-endpoint degree exactly 2. Each tile side adds 1 at `port_node(r,c,s)`; the inlet port gets +1 (virtual stub); each outlet cell's S port gets +1 (ground stub). |

## 4. Mechanism: four pieces

**M1. Exact-inlet rule.** A tile's `inlet_sides` must equal the set of sides whose neighbour exports into the cell. For the inlet cell the implicit incoming set is `{'N'}`. This permanently kills open inlets. Merge tiles (2-in, 3-in) can then only be placed where two or three exporters already point at the cell.

**M2. STOP only when closed.** Remove the `or not any_tile_valid` branch. If the board is closed, no tile action exists (a tile needs an exporting neighbour facing an empty cell), so the mask is exactly `{STOP}`.

**M3. Viability filter (the core).** A tile action is allowed only if, after placing it, a valid completion exists that closes everything within `max_tiles`. The test is a deterministic, bounded DFS (the 'closure oracle') that replays the real placement rules, so every plan it returns is valid by construction. It is sound but possibly conservative.

**M4. Budget reserve.** Part of M3: the oracle's plan length must be at most `max_tiles - len(board_after)`. At `len == max_tiles` the board is then necessarily closed.

**Alternatives rejected**

- Penalty-only: soft, open ends still appear.
- Post-hoc auto-closing at STOP: the agent loses control of the closing tiles' resistances, hurting target matching.
- Macro-actions (place a whole path): an architecture change.

## 5. New module `closure_v2.py`

All functions are pure functions of `(board, spec, library)`. Nothing is cached on the env itself, so curriculum writes cannot desync it.

### 5.1 Data helpers

```python
Sides = Tuple[Tuple[str, ...], Tuple[str, ...]]   # (inlet_sides, outlet_sides), each sorted
TilesMap = Dict[Tuple[int, int], Sides]

def sides_map(board) -> TilesMap:
    # {cell: (tuple(sorted(v.inlet_sides)), tuple(sorted(v.outlet_sides)))}

def board_signature(board) -> frozenset:
    # frozenset((cell, variant_id)); used for the env-level mask cache

def tiles_signature(tiles: TilesMap) -> frozenset:
    # frozenset((cell, I, O)); used for the oracle cache

def allowed_combos(library) -> Set[Sides]:
    # normalized keys from library.variants
```

Rules: always normalize with `tuple(sorted(...))` on both inlets and outlets. Do NOT assume all 50 combos exist; a COMSOL-derived CSV may lack some, so use `allowed_combos`.

### 5.2 `analyze_ports(tiles, spec) -> PortReport`

```python
@dataclass
class PortReport:
    demand: Dict[Cell, Tuple[str, ...]]   # empty cell -> sorted incoming sides (incl. virtual inlet)
    pending: List[Tuple[Cell, str]]       # (placed cell, outlet side) facing an empty cell
    open_inlets: List[Tuple[Cell, str]]
    blocked_outlets: List[Tuple[Cell, str]]
    missing_sinks: List[Cell]             # chip outlet cells not yet placed
    @property
    def closed(self) -> bool:             # not (pending or open_inlets or blocked_outlets or missing_sinks)
```

Algorithm:

1. For each tile `(c, (I, O))` and each `s` in `I`: if `c == inlet_cell and s == 'N'`, OK. Else if the neighbour is placed and `OPPOSITE[s]` is in its `O`, OK. Else append to `open_inlets`.
2. For each `s` in `O`: if `c in outlet_set and s == 'S'`, OK. Else if the neighbour is out of bounds, append to `blocked_outlets`. Else if the neighbour is placed: require `OPPOSITE[s]` in its `I`, else append to `blocked_outlets`. Else (neighbour empty) append to `pending` and add `OPPOSITE[s]` to `demand[neighbour]`.
3. If `inlet_cell` is not in `tiles`, set `demand[inlet_cell] = ('N',)` (virtual demand from the external inlet).
4. `missing_sinks = [t for t in outlet_cells if t not in tiles]`.

`find_open_ports(board, spec) -> List[(cell, side, kind)]` is a thin wrapper returning all defects plus pending ports, with `kind` in `{'open_inlet', 'pending', 'blocked'}`. It is used by tests, the final validator and the visualizer.

### 5.3 `check_port_degrees(board, spec) -> List[int]` (invariant I7)

Returns the fine-grid port node ids whose degree is not 2. Use `port_node` from `hydraulic_solver_v2`. Do not run the solver.

### 5.4 `cell_context(tiles, spec, cell)`

Single source of truth for local placement legality, shared by the mask, the oracle and the curriculum pre-placement. Returns `(I, free_sides, options)`.

For each side `s` with neighbour `nb`:

- **Out of bounds**
  - `s == 'N'` and `cell == inlet_cell`: add `'N'` to `I`.
  - `s == 'S'` and `cell in outlet_set`: nothing here; handled below.
  - Otherwise: blocked, no leg allowed on this side.
- **Neighbour placed**: if it exports toward `cell` (`OPPOSITE[s]` in its `O`), add `s` to `I`. Otherwise the side is blocked. Never allow an outlet toward a placed tile.
- **Neighbour empty**: add `s` to `free`.

Then:

- If `cell in outlet_set`: the only option is `O = ('S',)`. If `I` is empty the cell is not a candidate.
- Otherwise: options are all non-empty subsets of `free` (sorted tuples), filtered by `(tuple(sorted(I)), O) in allowed_combos`. If `I` is empty the cell is not a candidate.

Note: a non-outlet bottom-row cell has `S` out of bounds, so `S` is correctly blocked; this replaces the current special-case check.

### 5.5 `ClosureOracle`

```python
class ClosureOracle:
    def __init__(self, spec, allowed, node_limit=4000): ...
    def plan(self, tiles: TilesMap) -> Optional[List[Tuple[Cell, Tuple, Tuple]]]
    def min_tiles_required(self) -> Optional[int]   # iterative deepening from the empty board
    stats: calls, cache_hits, nodes, cap_hits
```

Budget for a call is `spec.max_tiles - len(tiles)`.

**Cache.** `self._cache: Dict[frozenset, Optional[plan]]` keyed by `tiles_signature`. Clear it in `env.reset()` when it exceeds about 50,000 entries.

**Search** (recursive, one node counter per top-level call):

```
search(tiles, budget):
    nodes += 1; if nodes > node_limit: raise _Cap     # caught at top: return None, cap_hits += 1
    rep = analyze_ports(tiles)
    if rep.open_inlets or rep.blocked_outlets: return None   # illegal state
    if rep.closed: return []
    if budget <= 0: return None
    # cheap necessary conditions (return None on failure):
    N1. demand empty but sinks missing        -> None   (nothing can reach them)
    N2. all sinks placed but demand non-empty -> None   (flow has nowhere to go)
    N3. |set(demand) | set(missing_sinks)| > budget -> None
    N4. compute 4-connected components of EMPTY cells;
        every demanded cell's component must contain >= 1 missing sink;
        every missing sink's component must contain >= 1 demanded cell
    X = min(demand, key=(row, col))      # fixed order: lower rows first, so sinks fill last and merges stay possible
    I = demand[X]
    for O in ordered_options(X, ...):
        res = search(tiles + {X: (I, O)}, budget - 1)
        if res is not None: return [(X, I, O)] + res
    return None
```

`ordered_options` takes the option list from `cell_context(X)` and sorts it with these heuristics. They affect speed only, never correctness:

1. `unclaimed` = missing sinks not in `demand`; `A` = demanded cells that are not sinks; `need_split = len(unclaimed) > len(A)`.
2. Primary key: `|O|` descending if `need_split`, else ascending.
3. Secondary key: sum over `o` in `O` of the Manhattan distance from the neighbour cell to the nearest target (nearest `unclaimed` sink if any, else nearest missing sink).
4. Tertiary key: the tuple `O` itself, for a total order.

**Determinism rules (mandatory)**

- No randomness anywhere.
- Iterate sets only through `sorted(...)`.
- Ordering is a pure function of the state.
- The node limit applies per top-level call.
- Hitting the cap means 'not viable' (sound, possibly conservative) and is counted in `cap_hits`.

**Prefix property.** If `plan(S) = P`, then `plan(S + P[0])` should equal `P[1:]`. It is enforced by seeding the cache in `step()` (section 6.4) and verified by a unit test.

## 6. `environment_v2.py` changes

### 6.1 Constructor

New keyword arguments:

```python
enforce_closure: bool = True,
closure_node_limit: int = 4000,
strict_closure: bool = False,
```

- `enforce_closure=False` must reproduce the current behaviour bit-for-bit. Keep the old mask as `_legacy_valid_action_mask()` and branch on the flag. This is the regression baseline and the ablation switch.
- Build `self.combo_to_vidx: Dict[Sides, np.ndarray]` by normalizing `(sorted inlet, sorted outlet)` for each library variant.
- Build `self.oracle = ClosureOracle(spec, allowed_combos(library), closure_node_limit)`.
- **Feasibility check:** if `oracle.plan({})` is `None`, raise `ValueError` that states the infeasibility and the value of `min_tiles_required()` (the default spec needs 17 tiles), and suggests raising `--max-tiles`.
- Add `self._mask_cache = (signature, mask)` and `self._closure_violation = False`.

### 6.2 New `valid_action_mask()` (when `enforce_closure`)

1. If `self.done`, return all False.
2. Compute `sig = board_signature(board)`. If it matches `_mask_cache`, return a copy.
3. `tiles = sides_map(board)`, `rep = analyze_ports(tiles, spec)`.
4. If `rep.closed`: enable only STOP, cache, return.
5. If `len(board) >= max_tiles`: skip to step 7.
6. For each `X` in `sorted(rep.demand)` (this covers the inlet cell on an empty board): get `(I, free, options)` from `cell_context`. For each `O` in `options`, build `tiles2 = tiles + {X: (I, O)}`. If `oracle.plan(tiles2)` is not `None`, set `mask[cell_idx*V + vi] = True` for every `vi` in `combo_to_vidx[(I, O)]`.
7. If no tile action is enabled and the board is not closed, this is an invariant violation. Set `_closure_violation = True`. If `strict_closure`, raise `RuntimeError` with a board dump. Otherwise log once per episode and enable STOP so training does not crash; the guards in section 8 exclude such episodes from `best_board`.
8. STOP is never enabled unless the board is closed (step 4).

**Speed notes.** First accept `plan(tiles)[0]` without a new DFS when it equals the candidate option. Rely on the oracle's necessary conditions N1 to N4 to reject most bad options before any DFS. Replace the current Python loop over all 640 variants per candidate cell with `combo_to_vidx` lookups; this alone offsets part of the oracle cost.

### 6.3 First-tile behaviour

It falls out of the generic path: `X = inlet_cell`, `I = ('N',)`, options are subsets of the in-bounds free sides excluding N. This also fixes the extra-inlet hole and out-of-bounds E/W outlets for inlets in edge columns.

### 6.4 `step()`

- Reward logic stays unchanged.
- After placing a tile, look up `plan = oracle.plan(sides_map(board))`. It must not be `None`; it is normally already cached from mask time.
- If the action matches the previous state's `plan[0]` and the new state is not cached, seed `cache[new_sig] = old_plan[1:]` (prefix property).
- When `done` via `max_tiles`, assert the board is closed (raise if `strict_closure`, otherwise set `_closure_violation`).
- Add to `info`: `{'closed': bool, 'pending': int, 'closure_violation': bool}`.

### 6.5 New methods and cleanup

- `is_closed() -> bool`.
- `force_place(cell, variant)`, used by the curriculum. It validates through `cell_context` (exact inlets, outlets only on free sides) and the oracle, puts the tile on the board, re-evaluates, refreshes `_prev_conn_prog` and `_prev_balance`, and invalidates the mask cache. It raises on invalid or non-viable placements.
- Remove the duplicated unreachable `return np.vstack(rows), valid_ids` at the end of `features()`.

`features()` stays exactly as is (39-dim) and benefits from the mask cache.

## 7. `curriculum_v2.py`

- Replace direct `env.board[...] = PlacedTile(...)` writes with `env.force_place(...)`.
- `_pre_place_inlet`: choose the variant whose combo is exactly `(('N',), ('S',))`, picking minimal `total_r` and breaking ties by `variant_id`. The current 'first variant with N in inlet\_sides...' could pick a 2-in tile with a dangling inlet if the library order changes.
- `_pre_place_downstream`: the cell is `(r_in + 1, c_in)`, with the same exact combo `(('N',), ('S',))`. Skip if that cell is an outlet cell or out of bounds.
- After pre-placement the state must be viable; `force_place` enforces this.
- Leave stage 3 ('hard') untouched.

## 8. `train_v2.py`, `run_stage1_v2.py`, `visualize_stage1.py`

**`train_v2.py`**

- Only consider a board for `best_board` / `best_reward` if `env.is_closed()`, in PPO, REINFORCE and the final greedy evaluation.
- Append to each `history` dict, without removing existing keys: `closed_rate`, `mean_tiles`, `oracle_cap_hits`, `oracle_calls`.

**`run_stage1_v2.py`**

- Add flags `--allow-open-ends` (sets `enforce_closure=False`), `--closure-node-limit`, `--strict-closure`.
- After training, call `find_open_ports(best_board, spec)` and `check_port_degrees`. If either is non-empty while closure is enforced, print a loud error and exit with code 2.
- Add `'closure': {'closed': bool, 'open_ports': [...]}` to `design.json`.
- Write `variant_index` and `k_total` (index within the combo and combo size) into each tile entry of `design.json`.

**`visualize_stage1.py`**

- Draw a red cross at every open port. Edge midpoints in plot coordinates: `x0+6±6` for E/W and `y0+6±6` for N/S, with N at the top.
- Mark tiles with open ports in the ASCII view.
- Use `variant_index` and `k_total` from `design.json` when present. Currently the viewer parses the index from the ID suffix (off by one, since IDs use `vi+1`) and hard-codes `k_total=max(vi+1, 12)`, so it can draw meanders that differ from the ones whose resistances were solved.

## 9. Tests (`tests/test_closure_v2.py`)

1. **`analyze_ports` unit tests** on hand-built boards: a trunk with only 1 of 3 outlets; a dangling outlet; a dangling inlet; an outlet pointing at a placed non-accepting tile; a fully closed 17-tile layout (zero defects, passes I7).
2. **Mask-subset test.** On random reachable states, every new tile action must also be in the legacy mask.
3. **STOP test.** STOP is invalid whenever any pending port exists.
4. **Oracle soundness (replay).** For random states, apply `plan(S)` through `env.step`, choosing any variant of each `(I, O)`. Every step must be in the mask and the final board must be closed with I1 to I7 holding.
5. **Prefix property.** `plan(S + P[0]) == P[1:]` on random states.
6. **Brute-force comparison.** On a 4x4 grid with 2 outlets, run an exhaustive order-free search. Oracle-viable must be a subset of truly viable (100% soundness); report the completeness ratio.
7. **Fuzz.** At least 5,000 episodes of uniform-random valid actions across several specs (different targets, `max_tiles` in {17, 20, 25}, inlet in an edge column, 2 vs 3 outlets, adjacent outlets). Assert every episode ends closed, the mask is never empty before STOP, `len(board) <= max_tiles`, and solver `mass_residual` is small.
8. **Infeasible spec.** `max_tiles=16` on the default spec must raise at construction.
9. **Curriculum.** Stages 0 to 3 all produce viable starts that can be completed to a closed board.
10. **Regression.** With `enforce_closure=False` and the same seed, `history` matches the current code's output.
11. **Compatibility.** `FEATURE_DIM == 39`, `n_actions` unchanged, and an old `policy.npz` still loads.

## 10. Acceptance criteria

| ID | Criterion |
| --- | --- |
| A1 | Fuzz tests pass with `closed_rate = 1.0` and zero mask-empty events. |
| A2 | A 2000-episode PPO run on the default spec gives `closed_rate = 1.0` and a final `chip_layout.png` with no open-port markers. |
| A3 | Oracle soundness is 100%. Report the completeness ratio and the cap-hit rate (should be about 0 at the default node limit). |
| A4 | Steps per second drops by no more than about 4x versus baseline. Profile mask and oracle and report numbers. |
| A5 | The regression test (test 10) passes. |

## 11. Implementation order

1. Snapshot baseline outputs and speed with a fixed seed.
2. Write the `closure_v2.py` helpers and tests 1 and 8.
3. Write the oracle and tests 4, 5 and 6.
4. Integrate into `environment_v2.py` behind `enforce_closure`, then tests 2, 3, 7 and 10.
5. Update the curriculum (test 9).
6. Update train, run and viz.
7. Run the fuzz and PPO acceptance runs.
8. Update the README: it still says 'Masked REINFORCE' and describes the old STOP rule.

## 12. Expected side effects and risks

- **Retrain required.** The policy sees a different action distribution. Old weights load but should be retrained.
- **Fewer cheap designs.** The 17-tile design is exactly the minimum, so every extra leg now costs real tiles. With `max_tiles=25` there are 8 spare tiles; consider 30 or more for richer resistance tuning (a `--max-tiles` choice, outside this plan).
- **Merges are rare.** They occur only where branches deliberately converge on an empty cell.
- **Conservative oracle.** The node cap or fixed fill order may hide a few viable but unusual actions. Soundness is never compromised; test 6 quantifies the loss.
- **Flow direction.** Closure does not guarantee flow direction on each leg, because the solver is undirected. A non-blocking check of leg-flow signs against inlet/outlet labels is a good follow-up.

## Appendix A. Optional separate fix: `is_newly_placed_dead`

Not part of closure, but found during analysis. When a tile is placed on a chip outlet cell, its `S` outlet points at row 8, outside the grid. That cell is not in `outlet_set` and fails the bounds check, so the function falls through and returns `True`. Every correct outlet-tile placement therefore receives a spurious `dead_branch_penalty` (-0.1), about -0.3 per successful episode.

**Fix:** at the top of the loop over `placed.variant.outlet_sides`, treat `o_side == 'S'` on a cell in `outlet_set` as a valid chip outlet and return False. Add a unit test that places a tile on each outlet cell and asserts the function returns False. This changes rewards, so apply it only if you accept shifting the reward baseline.
