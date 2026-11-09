"""Training algorithms (GAE-PPO and REINFORCE) for Stage 1 microfluidic RL topology synthesis."""
from __future__ import annotations

from collections import Counter
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .agent import PolicyNet, ValueNet, compute_gae, rollout, rollout_ppo, softmax
from .curriculum import CurriculumManager
from .library import Library, PlacedTile
from .metrics import design_id, topo_id
from .reward import Evaluation


def make_archive_entry(
    board: Dict[Tuple[int, int], PlacedTile],
    ev: Evaluation,
    ep_idx: int,
    seed: int,
) -> Dict[str, Any]:
    tid = topo_id(board)
    did = design_id(board)
    return {
        "tiles": [
            {
                "cell": [int(rc[0]), int(rc[1])],
                "variant_id": tile.variant.variant_id,
                "inlet_sides": list(tile.variant.inlet_sides),
                "outlet_sides": list(tile.variant.outlet_sides),
                "R_per_side": dict(tile.variant.R_per_side),
            }
            for rc, tile in sorted(board.items())
        ],
        "metrics": ev.metrics(),
        "score": float(ev.score),
        "topo_id": tid,
        "design_id": did,
        "episode": int(ep_idx),
        "seed": int(seed),
    }


def archive_key(entry: Dict[str, Any]) -> Tuple[float, int, float, str]:
    """Total-order ranking key for designs: lowest is best."""
    score = float(entry["score"])
    metrics = entry.get("metrics", {})
    n_tiles = int(metrics.get("n_tiles", len(entry.get("tiles", []))))
    p_in = float(metrics.get("p_in", float("inf")))
    if not np.isfinite(p_in):
        p_in = float("inf")
    did = str(entry.get("design_id", ""))
    return (-score, n_tiles, p_in, did)


def update_archive(
    archive: List[Dict[str, Any]],
    board: Dict[Tuple[int, int], PlacedTile],
    ev: Evaluation,
    ep_idx: int,
    seed: int,
    archive_k: int = 5,
) -> bool:
    """Update top-K archive with at most 1 design per topo_id. Returns True if archive changed."""
    new_entry = make_archive_entry(board, ev, ep_idx, seed)
    new_k = archive_key(new_entry)
    tid = new_entry["topo_id"]

    # 1. If topo_id is already present, replace only if new key is better
    for i, item in enumerate(archive):
        if item["topo_id"] == tid:
            if new_k < archive_key(item):
                archive[i] = new_entry
                archive.sort(key=archive_key)
                return True
            return False

    # 2. If space available, add it
    if len(archive) < archive_k:
        archive.append(new_entry)
        archive.sort(key=archive_key)
        return True

    # 3. If full, replace worst entry if new key is better
    archive.sort(key=archive_key)
    worst_k = archive_key(archive[-1])
    if new_k < worst_k:
        archive[-1] = new_entry
        archive.sort(key=archive_key)
        return True

    return False


def tiles_to_board(
    tiles_list: List[Dict[str, Any]],
    library: Library,
) -> Dict[Tuple[int, int], PlacedTile]:
    board: Dict[Tuple[int, int], PlacedTile] = {}
    for item in tiles_list:
        rc = (int(item["cell"][0]), int(item["cell"][1]))
        vid = item["variant_id"]
        v = library.by_id[vid]
        board[rc] = PlacedTile(rc, v)
    return board


def evaluate_policy(
    env,
    policy_net: PolicyNet,
    value_net: Optional[ValueNet] = None,
    eval_episodes: int = 200,
    seed: int = 0,
    global_dim: int = 18,
    archive: Optional[List[Dict[str, Any]]] = None,
    archive_k: int = 5,
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    """Fixed evaluation after training using a dedicated RNG stream (seed + 1000)."""
    if eval_episodes <= 0:
        return {}, archive

    rng_eval = np.random.default_rng(seed + 1000)

    succ_list: List[bool] = []
    tiles_list: List[int] = []
    eq_list: List[float] = []
    extra_list: List[int] = []
    loops_list: List[int] = []
    p_ratio_list: List[float] = []
    topo_ids: List[str] = []
    design_ids: List[str] = []

    # Run eval_episodes stochastic episodes
    for ep in range(eval_episodes):
        env.reset()
        if value_net is not None:
            rollout_ppo(env, policy_net, value_net, rng_eval, global_dim=global_dim, greedy=False)
        else:
            rollout(env, policy_net, rng_eval, greedy=False)

        ev = env.evaluation
        is_closed = env.is_closed() if hasattr(env, "is_closed") else True
        n_t = len(env.board)
        tid = topo_id(env.board)
        did = design_id(env.board)

        succ_list.append(bool(ev.success))
        tiles_list.append(n_t)
        eq_list.append(float(ev.e_q))
        extra_list.append(int(getattr(ev, "extra", max(0, n_t - getattr(env, "n_min", 17)))))
        loops_list.append(int(getattr(ev, "loops", 0)))
        p_ratio_list.append(float(getattr(ev, "p_ratio", 0.0)))
        topo_ids.append(tid)
        design_ids.append(did)

        if archive is not None and is_closed and ev.success:
            update_archive(archive, env.board, ev, ep_idx=-(ep + 1), seed=seed, archive_k=archive_k)

    # 1 greedy episode
    env.reset()
    if value_net is not None:
        rollout_ppo(env, policy_net, value_net, rng_eval, global_dim=global_dim, greedy=True)
    else:
        rollout(env, policy_net, rng_eval, greedy=True)

    ev_greedy = env.evaluation
    is_closed_greedy = env.is_closed() if hasattr(env, "is_closed") else True
    if archive is not None and is_closed_greedy and ev_greedy.success:
        update_archive(archive, env.board, ev_greedy, ep_idx=-999, seed=seed, archive_k=archive_k)

    n_g = len(env.board)
    greedy_metrics = {
        "success": bool(ev_greedy.success),
        "n_tiles": n_g,
        "e_q": float(ev_greedy.e_q),
        "extra": int(getattr(ev_greedy, "extra", max(0, n_g - getattr(env, "n_min", 17)))),
        "loops": int(getattr(ev_greedy, "loops", 0)),
        "p_ratio": float(getattr(ev_greedy, "p_ratio", 0.0)),
        "score": float(ev_greedy.score),
        "topo_id": topo_id(env.board),
        "design_id": design_id(env.board),
    }

    topo_counts = Counter(topo_ids)
    top_3_topo = topo_counts.most_common(3)

    summary = {
        "success_rate": float(np.mean(succ_list)) if succ_list else 0.0,
        "mean_tiles": float(np.mean(tiles_list)) if tiles_list else 0.0,
        "std_tiles": float(np.std(tiles_list)) if tiles_list else 0.0,
        "frac_at_budget": float(np.mean([t == env.spec.max_tiles for t in tiles_list])) if tiles_list else 0.0,
        "mean_extra": float(np.mean(extra_list)) if extra_list else 0.0,
        "mean_loops": float(np.mean(loops_list)) if loops_list else 0.0,
        "mean_p_ratio": float(np.mean(p_ratio_list)) if p_ratio_list else 0.0,
        "n_distinct_topo": len(topo_counts),
        "top_3_topo": [{"topo_id": tid, "count": cnt} for tid, cnt in top_3_topo],
        "greedy": greedy_metrics,
    }
    return summary, archive


def train_agent(
    env,
    episodes: int = 2000,
    batch_size: int = 16,
    lr: float = 0.01,
    entropy_coef: float = 0.01,
    entropy_final: Optional[float] = None,
    hidden: int = 64,
    seed: int = 0,
    log_every: int = 10,
    verbose: bool = True,
    archive_k: int = 5,
    eval_episodes: int = 0,
) -> Tuple[PolicyNet, Dict, float, List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    net = PolicyNet(env.feature_dim, hidden, seed)
    archive: List[Dict[str, Any]] = []
    fallback_best: Optional[Dict[str, Any]] = None
    fallback_key: Optional[Tuple[float, int, float, str]] = None

    history: List[Dict[str, Any]] = []
    n_batches = max(1, episodes // batch_size)
    t0 = time.time()
    low_entropy_consecutive = 0

    for b in range(n_batches):
        cur_entropy_coef = (
            entropy_coef
            if entropy_final is None
            else float(entropy_final + (entropy_coef - entropy_final) * (1.0 - b / max(n_batches - 1, 1)))
        )

        trajs, rewards, scores, errs = [], [], [], []
        closed_flags, tile_counts = [], []
        evals = []

        for ep_sub in range(batch_size):
            ep_idx = b * batch_size + ep_sub + 1
            traj, _ = rollout(env, net, rng)
            ev = env.evaluation
            evals.append(ev)
            trajs.append(traj)
            rewards.append(ev.reward)
            scores.append(ev.score)
            errs.append(ev.e_q)
            is_closed = env.is_closed() if hasattr(env, "is_closed") else True
            closed_flags.append(is_closed)
            tile_counts.append(len(env.board))

            cand = make_archive_entry(env.board, ev, ep_idx, seed)
            cand_k = archive_key(cand)
            if fallback_key is None or cand_k < fallback_key:
                fallback_key = cand_k
                fallback_best = cand

            if is_closed and ev.success:
                update_archive(archive, env.board, ev, ep_idx, seed, archive_k=archive_k)

        # Entropy diagnostics
        H_list: List[float] = []
        H_norm_list: List[float] = []
        maxp_list: List[float] = []
        for tr in trajs:
            for s in tr:
                p = s.p
                m = len(p)
                H = -float(np.sum(p * np.log(p + 1e-12)))
                H_list.append(H)
                if m > 1:
                    H_norm_list.append(H / float(np.log(m)))
                maxp_list.append(float(np.max(p)))

        entropy_mean = float(np.mean(H_list)) if H_list else 0.0
        entropy_norm_mean = float(np.mean(H_norm_list)) if H_norm_list else 0.0
        maxprob_mean = float(np.mean(maxp_list)) if maxp_list else 0.0

        if entropy_norm_mean < 0.10:
            low_entropy_consecutive += 1
            if low_entropy_consecutive >= 5:
                print(
                    f"WARNING: Policy entropy collapse! entropy_norm_mean ({entropy_norm_mean:.4f}) < 0.10 "
                    f"for {low_entropy_consecutive} consecutive log lines.",
                    flush=True,
                )
        else:
            low_entropy_consecutive = 0

        # Compute returns-to-go with per-timestep baseline subtraction
        returns = [np.cumsum([s.r for s in tr][::-1])[::-1] for tr in trajs]
        t_max = max(len(r) for r in returns) if returns and any(len(r) > 0 for r in returns) else 0
        if t_max > 0:
            base = np.zeros(t_max)
            for t in range(t_max):
                matching = [r[t] for r in returns if len(r) > t]
                base[t] = float(np.mean(matching)) if matching else 0.0
            advs = [r - base[:len(r)] for r in returns]
            all_advs = np.concatenate(advs) if advs and any(len(a) > 0 for a in advs) else np.array([])
            std = float(all_advs.std()) if len(all_advs) > 0 else 0.0
            scale = 1.0 / (std + 1e-8) if std > 1e-8 else 0.0

            grads = net.zero_grads()
            n_steps = sum(len(tr) for tr in trajs)
            for tr, adv in zip(trajs, advs):
                for s, a in zip(tr, adv):
                    p = s.p
                    g = np.zeros_like(p)
                    g[s.j] = 1.0
                    g = (g - p) * (a * scale)  # d(log pi)/d(scores) * advantage
                    logp = np.log(np.clip(p, 1e-12, 1.0))
                    ent_grad = -p * (logp + float(-(p * logp).sum()))  # d(entropy)/d(scores)
                    net.accumulate(grads, s.x.astype(float), s.h, g + cur_entropy_coef * ent_grad)

            for k in grads:
                grads[k] /= max(n_steps, 1)
            net.adam_step(grads, lr)

        cap_hits = env.oracle.cap_hits if hasattr(env, "oracle") else 0
        calls = env.oracle.calls if hasattr(env, "oracle") else 0
        mean_extra_n = float(np.mean([getattr(e, "struct_parts", {}).get("extra_n", 0.0) for e in evals]))
        mean_press_n = float(np.mean([getattr(e, "struct_parts", {}).get("press_n", 0.0) for e in evals]))
        mean_loops_n = float(np.mean([getattr(e, "struct_parts", {}).get("loops_n", 0.0) for e in evals]))
        mean_struct = float(np.mean([getattr(e, "struct_cost", 0.0) for e in evals]))

        cur_best_score = archive[0]["score"] if archive else (fallback_best["score"] if fallback_best else -np.inf)

        history.append({
            "episode": (b + 1) * batch_size,
            "mean_reward": float(np.mean(rewards)),
            "mean_score": float(np.mean(scores)),
            "mean_E_Q": float(np.mean(errs)),
            "best_reward": float(cur_best_score),
            "closed_rate": float(np.mean(closed_flags)) if closed_flags else 1.0,
            "mean_tiles": float(np.mean(tile_counts)) if tile_counts else 0.0,
            "entropy_mean": entropy_mean,
            "entropy_norm_mean": entropy_norm_mean,
            "maxprob_mean": maxprob_mean,
            "mean_extra_n": mean_extra_n,
            "mean_press_n": mean_press_n,
            "mean_loops_n": mean_loops_n,
            "mean_struct_cost": mean_struct,
            "oracle_cap_hits": cap_hits,
            "oracle_calls": calls,
        })

        if verbose and ((b + 1) % log_every == 0 or b == 0 or b + 1 == n_batches):
            print(
                f"[ep {(b + 1) * batch_size:6d}] mean reward {np.mean(rewards):8.4f} | "
                f"mean E_Q {np.mean(errs):7.4f} | best score {cur_best_score:8.4f} | "
                f"H_norm {entropy_norm_mean:5.3f} | {time.time() - t0:6.1f}s",
                flush=True,
            )

    # Fixed evaluation
    eval_summary, archive = evaluate_policy(
        env=env,
        policy_net=net,
        value_net=None,
        eval_episodes=eval_episodes,
        seed=seed,
        archive=archive,
        archive_k=archive_k,
    )

    eligible = len(archive) > 0
    if eligible:
        best_entry = archive[0]
        best_board = tiles_to_board(best_entry["tiles"], env.library)
        best_reward = float(best_entry["score"])
    else:
        best_board = tiles_to_board(fallback_best["tiles"], env.library) if fallback_best else dict(env.board)
        best_reward = float(fallback_best["score"]) if fallback_best else -float("inf")

    meta = {
        "archive": archive,
        "eligible": eligible,
        "eval_summary": eval_summary,
    }

    return net, best_board, float(best_reward), history, meta


def train_agent_ppo(
    env,
    episodes: int = 3000,
    rollouts_per_update: int = 8,
    K_epochs: int = 4,
    mini_batch_size: int = 64,
    lr_policy: float = 3e-3,
    lr_value: float = 1e-2,
    eps_clip: float = 0.2,
    entropy_coef: float = 0.01,
    entropy_final: Optional[float] = None,
    gamma: float = 0.99,
    gae_lam: float = 0.95,
    hidden: int = 128,
    seed: int = 0,
    global_dim: int = 18,
    log_every: int = 50,
    verbose: bool = True,
    use_curriculum: bool = True,
    curriculum_mode: str = "sequential",
    cur_window: int = 50,
    cur_threshold: float = 0.4,
    cur_q_gate: float = 0.15,
    cur_min_stage_episodes: int = 100,
    cur_max_stage_fraction: float = 0.25,
    cur_final_normal_fraction: float = 0.4,
    cur_hard: bool = False,
    probe_every: int = 25,
    probe_episodes: int = 16,
    best_min_stage: int = 0,
    adv_norm: str = "batch",
    archive_k: int = 5,
    eval_episodes: int = 0,
) -> Tuple[PolicyNet, ValueNet, Dict, float, List[Dict[str, Any]], Dict[str, Any]]:
    """GAE-PPO trainer with physics-informed advantage estimation, batch normalization, and archive."""
    rng = np.random.default_rng(seed)
    rng_probe = np.random.default_rng(seed + 11)
    policy_net = PolicyNet(env.feature_dim, hidden, seed)
    value_net = ValueNet(global_dim, hidden // 2, seed + 1)

    n_updates = max(1, episodes // rollouts_per_update)
    total_effective_episodes = n_updates * rollouts_per_update

    curriculum = None
    if use_curriculum and curriculum_mode != "off":
        curriculum = CurriculumManager(
            total_episodes=total_effective_episodes,
            window=cur_window,
            threshold=cur_threshold,
            q_gate=cur_q_gate,
            min_stage_episodes=cur_min_stage_episodes,
            max_stage_fraction=cur_max_stage_fraction,
            final_normal_fraction=cur_final_normal_fraction,
            mode=curriculum_mode,
            use_hard_stage=cur_hard,
            seed=seed,
        )

    archive: List[Dict[str, Any]] = []
    fallback_best: Optional[Dict[str, Any]] = None
    fallback_key: Optional[Tuple[float, int, float, str]] = None

    history: List[Dict[str, Any]] = []
    total_episodes = 0
    t0 = time.time()
    low_entropy_consecutive = 0

    for update in range(n_updates):
        cur_entropy_coef = (
            entropy_coef
            if entropy_final is None
            else float(entropy_final + (entropy_coef - entropy_final) * (1.0 - update / max(n_updates - 1, 1)))
        )

        episodes_data = []
        batch_rewards = []
        batch_scores = []
        batch_errs = []
        batch_closed = []
        batch_tiles = []
        batch_stages = []
        batch_evals = []
        last_trans_reason = None

        for _ in range(rollouts_per_update):
            total_episodes += 1
            if curriculum:
                stage_used = curriculum.setup_episode(env, env.library)
            else:
                stage_used = 2
                env.reset()

            traj, _ = rollout_ppo(env, policy_net, value_net, rng, global_dim=global_dim)
            raw_adv, returns = compute_gae(traj, value_net, global_dim=global_dim, gamma=gamma, lam=gae_lam)
            episodes_data.append((traj, raw_adv, returns))

            ev = env.evaluation
            batch_evals.append(ev)
            batch_rewards.append(ev.reward)
            batch_scores.append(ev.score)
            batch_errs.append(ev.e_q)
            is_closed = env.is_closed() if hasattr(env, "is_closed") else True
            batch_closed.append(is_closed)
            batch_tiles.append(len(env.board))
            batch_stages.append(stage_used)

            cand = make_archive_entry(env.board, ev, total_episodes, seed)
            cand_k = archive_key(cand)
            if fallback_key is None or cand_k < fallback_key:
                fallback_key = cand_k
                fallback_best = cand

            # Eligible: closed and ev.success
            if stage_used >= best_min_stage and is_closed and ev.success:
                update_archive(archive, env.board, ev, total_episodes, seed, archive_k=archive_k)

            if curriculum:
                trans = curriculum.advance({
                    "success": ev.success,
                    "closed": is_closed,
                    "e_q": ev.e_q,
                })
                if trans:
                    last_trans_reason = trans

        # Entropy diagnostics
        H_list = []
        H_norm_list = []
        maxp_list = []
        for tr, _, _ in episodes_data:
            for step in tr:
                p = step.action_p
                m = len(p)
                H = -float(np.sum(p * np.log(p + 1e-12)))
                H_list.append(H)
                if m > 1:
                    H_norm_list.append(H / float(np.log(m)))
                maxp_list.append(float(np.max(p)))

        entropy_mean = float(np.mean(H_list)) if H_list else 0.0
        entropy_norm_mean = float(np.mean(H_norm_list)) if H_norm_list else 0.0
        maxprob_mean = float(np.mean(maxp_list)) if maxp_list else 0.0

        if entropy_norm_mean < 0.10:
            low_entropy_consecutive += 1
            if low_entropy_consecutive >= 5:
                print(
                    f"WARNING: Policy entropy collapse! entropy_norm_mean ({entropy_norm_mean:.4f}) < 0.10 "
                    f"for {low_entropy_consecutive} consecutive log lines.",
                    flush=True,
                )
        else:
            low_entropy_consecutive = 0

        # Advantage normalization (Step 5.1)
        buffer = []
        if adv_norm == "batch":
            all_adv_list = [a for (_, a, _) in episodes_data if len(a) > 0]
            if all_adv_list:
                all_adv = np.concatenate(all_adv_list)
                mu = float(all_adv.mean())
                sd = float(all_adv.std())
            else:
                all_adv = np.array([])
                mu, sd = 0.0, 0.0

            for tr, raw_adv, rets in episodes_data:
                if len(raw_adv) == 0:
                    continue
                if len(all_adv) >= 2 and sd > 1e-8:
                    adv_normed = (raw_adv - mu) / (sd + 1e-8)
                else:
                    adv_normed = raw_adv - mu
                for t, step in enumerate(tr):
                    buffer.append((
                        step.global_x,
                        step.action_x,
                        step.action_j,
                        step.log_prob,
                        step.action_h,
                        adv_normed[t],
                        rets[t],
                        step.prior,
                    ))
        else:
            # Legacy: per-episode normalization
            for tr, raw_adv, rets in episodes_data:
                if len(raw_adv) == 0:
                    continue
                sd = float(raw_adv.std())
                adv_normed = (raw_adv - raw_adv.mean()) / (sd + 1e-8) if sd > 1e-8 else raw_adv - raw_adv.mean()
                for t, step in enumerate(tr):
                    buffer.append((
                        step.global_x,
                        step.action_x,
                        step.action_j,
                        step.log_prob,
                        step.action_h,
                        adv_normed[t],
                        rets[t],
                        step.prior,
                    ))

        # PPO update on collected buffer
        if buffer:
            for _ in range(K_epochs):
                indices = np.arange(len(buffer))
                rng.shuffle(indices)

                for start in range(0, len(buffer), mini_batch_size):
                    mb_idx = indices[start:start + mini_batch_size]
                    if len(mb_idx) == 0:
                        continue

                    policy_grads = policy_net.zero_grads()
                    value_grads = value_net.zero_grads()

                    for idx in mb_idx:
                        gx, ax, aj, lp_old, ah_old, adv_t, ret_t, prior = buffer[idx]

                        # Policy forward
                        s_new, h_new = policy_net.forward(ax)
                        if prior is not None:
                            s_new = s_new + prior

                        p_new = softmax(s_new)
                        lp_new = float(np.log(np.clip(p_new[aj], 1e-12, 1.0)))

                        ratio = float(np.exp(lp_new - lp_old))
                        ratio_clipped = float(np.clip(ratio, 1.0 - eps_clip, 1.0 + eps_clip))

                        # Surrogate gradient
                        if ratio * adv_t <= ratio_clipped * adv_t:
                            g_chosen = ratio * adv_t
                            g_scores = np.zeros(len(p_new))
                            g_scores[aj] = 1.0
                            g_scores = (g_scores - p_new) * g_chosen
                        else:
                            g_scores = np.zeros(len(p_new))

                        # Entropy gradient
                        logp = np.log(np.clip(p_new, 1e-12, 1.0))
                        ent_grad = -p_new * (logp + float(-(p_new * logp).sum()))

                        policy_net.accumulate(
                            policy_grads, ax.astype(float), h_new, g_scores + cur_entropy_coef * ent_grad
                        )

                        # Value gradient
                        v_pred, h_v = value_net.forward(gx)
                        v_loss_grad = 2.0 * float(v_pred - ret_t)
                        value_net.accumulate_value(value_grads, gx, h_v, v_loss_grad)

                    n_mb = len(mb_idx)
                    for k in policy_grads:
                        policy_grads[k] /= max(n_mb, 1)
                    for k in value_grads:
                        value_grads[k] /= max(n_mb, 1)

                    policy_net.adam_step(policy_grads, lr_policy)
                    value_net.adam_step(value_grads, lr_value)

        mean_r = float(np.mean(batch_rewards)) if batch_rewards else 0.0
        mean_score = float(np.mean(batch_scores)) if batch_scores else 0.0
        mean_eq = float(np.mean(batch_errs)) if batch_errs else 0.0

        # Run probe episodes from normal reset
        probe_res = {}
        if probe_episodes > 0 and probe_every > 0 and (
            (update + 1) % probe_every == 0 or update == 0 or update + 1 == n_updates
        ):
            p_succ, p_eq, p_tiles = [], [], []
            t_p0 = time.time()
            for _ in range(probe_episodes):
                env.reset()
                rollout_ppo(env, policy_net, value_net, rng_probe, global_dim=global_dim)
                p_succ.append(float(env.evaluation.success))
                p_eq.append(float(env.evaluation.e_q))
                p_tiles.append(float(len(env.board)))
            probe_res = {
                "probe_success_rate": float(np.mean(p_succ)),
                "probe_mean_e_q": float(np.mean(p_eq)),
                "probe_mean_tiles": float(np.mean(p_tiles)),
                "t_probe": float(time.time() - t_p0),
            }

        st_counts = Counter(batch_stages)
        stage_counts_str = ",".join(f"{k}:{v}" for k, v in sorted(st_counts.items()))

        cur_snap = curriculum.snapshot() if curriculum else None
        cur_stage_end = curriculum.stage if curriculum else 2
        cur_rate = cur_snap["rate"] if cur_snap else 0.0

        cap_hits = env.oracle.cap_hits if hasattr(env, "oracle") else 0
        calls = env.oracle.calls if hasattr(env, "oracle") else 0
        mean_extra_n = float(np.mean([getattr(e, "struct_parts", {}).get("extra_n", 0.0) for e in batch_evals]))
        mean_press_n = float(np.mean([getattr(e, "struct_parts", {}).get("press_n", 0.0) for e in batch_evals]))
        mean_loops_n = float(np.mean([getattr(e, "struct_parts", {}).get("loops_n", 0.0) for e in batch_evals]))
        mean_struct = float(np.mean([getattr(e, "struct_cost", 0.0) for e in batch_evals]))

        cur_best_score = archive[0]["score"] if archive else (fallback_best["score"] if fallback_best else -np.inf)

        hist_entry = {
            "episode": total_episodes,
            "mean_reward": mean_r,
            "mean_score": mean_score,
            "mean_E_Q": mean_eq,
            "best_reward": float(cur_best_score),
            "curriculum_stage": cur_stage_end,
            "curriculum_stage_end": cur_stage_end,
            "stage_counts": stage_counts_str,
            "curriculum_rate": cur_rate,
            "transition": last_trans_reason or "",
            "closed_rate": float(np.mean(batch_closed)) if batch_closed else 1.0,
            "mean_tiles": float(np.mean(batch_tiles)) if batch_tiles else 0.0,
            "entropy_mean": entropy_mean,
            "entropy_norm_mean": entropy_norm_mean,
            "maxprob_mean": maxprob_mean,
            "mean_extra_n": mean_extra_n,
            "mean_press_n": mean_press_n,
            "mean_loops_n": mean_loops_n,
            "mean_struct_cost": mean_struct,
            "oracle_cap_hits": cap_hits,
            "oracle_calls": calls,
        }
        hist_entry.update(probe_res)
        history.append(hist_entry)

        if verbose and ((update + 1) % log_every == 0 or update == 0 or update + 1 == n_updates):
            stage_name = curriculum.STAGES[cur_stage_end] if curriculum else "standard"
            probe_str = f" | probe E_Q: {probe_res['probe_mean_e_q']:.4f}" if "probe_mean_e_q" in probe_res else ""
            print(
                f"[ep {total_episodes:5d}] mean reward {mean_r:8.4f} | "
                f"mean E_Q {mean_eq:7.4f} | best score {cur_best_score:8.4f} | "
                f"H_norm {entropy_norm_mean:5.3f} | "
                f"stage: {stage_name} ({cur_stage_end}) [{stage_counts_str}]{probe_str} | "
                f"{time.time() - t0:6.1f}s",
                flush=True,
            )

    # Fixed evaluation (Step 2.4)
    eval_summary, archive = evaluate_policy(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        eval_episodes=eval_episodes,
        seed=seed,
        global_dim=global_dim,
        archive=archive,
        archive_k=archive_k,
    )

    eligible = len(archive) > 0
    if eligible:
        best_entry = archive[0]
        final_board = tiles_to_board(best_entry["tiles"], env.library)
        final_reward = float(best_entry["score"])
    else:
        final_board = tiles_to_board(fallback_best["tiles"], env.library) if fallback_best else dict(env.board)
        final_reward = float(fallback_best["score"]) if fallback_best else -float("inf")

    meta = {
        "archive": archive,
        "eligible": eligible,
        "eval_summary": eval_summary,
        "found_in_stage": best_entry.get("episode", -1) if eligible else -1,
        "best_any_stage": -1,
    }

    return policy_net, value_net, final_board, float(final_reward), history, meta
