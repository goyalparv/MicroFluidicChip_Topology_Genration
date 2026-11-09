"""Masked policy-gradient (REINFORCE) agent in pure numpy (no torch dependency)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


class PolicyNet:
    """score(x) = w2 . tanh(W1 x + b1) + b2, trained with manual backprop + Adam."""

    def __init__(self, in_dim: int, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.params: Dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, 1.0 / np.sqrt(in_dim), (in_dim, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0.0, 0.1, hidden),
            "b2": np.zeros(1),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self.params
        h = np.tanh(x @ p["W1"] + p["b1"])
        return h @ p["w2"] + p["b2"][0], h

    def zero_grads(self) -> Dict[str, np.ndarray]:
        return {k: np.zeros_like(v) for k, v in self.params.items()}

    def accumulate(
        self, grads: Dict[str, np.ndarray], x: np.ndarray, h: np.ndarray, g_scores: np.ndarray
    ) -> None:
        """Add d(objective)/d(params) given d(objective)/d(scores) = g_scores (gradient ascent)."""
        p = self.params
        grads["w2"] += h.T @ g_scores
        grads["b2"] += g_scores.sum()
        dz = (g_scores[:, None] * p["w2"][None, :]) * (1.0 - h * h)
        grads["W1"] += x.T @ dz
        grads["b1"] += dz.sum(axis=0)

    def adam_step(
        self,
        grads: Dict[str, np.ndarray],
        lr: float,
        clip: float = 5.0,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = clip / norm if norm > clip else 1.0
        self._t += 1
        for k, g in grads.items():
            g = g * scale
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            m_hat = self._m[k] / (1 - b1 ** self._t)
            v_hat = self._v[k] / (1 - b2 ** self._t)
            self.params[k] = self.params[k] + lr * m_hat / (np.sqrt(v_hat) + eps)

    def state(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def load(self, state: Dict[str, np.ndarray]) -> None:
        self.params = {k: np.array(v, dtype=float) for k, v in state.items()}

    def save(self, path: str) -> None:
        np.savez(path, **self.params)

    @staticmethod
    def from_file(path: str) -> "PolicyNet":
        data = np.load(path)
        w1 = data["W1"]
        net = PolicyNet(w1.shape[0], w1.shape[1])
        net.load({k: data[k] for k in data.files})
        return net


@dataclass
class Step:
    x: np.ndarray       # (m, d) features of valid actions
    h: np.ndarray       # (m, H) hidden layer activations
    p: np.ndarray       # (m,) policy probabilities
    j: int              # index of chosen row
    r: float            # reward


def rollout(env, net: PolicyNet, rng: np.random.Generator, greedy: bool = False, keep: bool = True):
    env.reset()
    traj: List[Step] = []
    while not env.done:
        x, ids = env.features()
        if len(ids) == 0:
            break
        s, h = net.forward(x)
        p = softmax(s)
        j = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        r, _, _ = env.step(int(ids[j]))
        if keep:
            traj.append(Step(x.astype(np.float32), h, p, j, r))
    return traj, env.evaluation.cost


class ValueNet:
    """State-value estimator V(s) = w2 . tanh(W1 x + b1) + b2 for GAE advantage computation."""

    def __init__(self, in_dim: int, hidden: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.params: Dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, 1.0 / np.sqrt(in_dim), (in_dim, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0.0, 0.1, hidden),
            "b2": np.zeros(1),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    def forward(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """x: 1D array of shape (in_dim,). Returns scalar value and hidden activations."""
        p = self.params
        h = np.tanh(x @ p["W1"] + p["b1"])
        v = float(h @ p["w2"] + p["b2"][0])
        return v, h

    def zero_grads(self) -> Dict[str, np.ndarray]:
        return {k: np.zeros_like(v) for k, v in self.params.items()}

    def accumulate_value(
        self, grads: Dict[str, np.ndarray], x: np.ndarray, h: np.ndarray, g_v: float
    ) -> None:
        """Add d(loss)/d(params) where g_v = d(loss)/d(v_pred) (gradient descent step)."""
        p = self.params
        grads["w2"] += h * g_v
        grads["b2"] += g_v
        dz = (g_v * p["w2"]) * (1.0 - h * h)
        grads["W1"] += np.outer(x, dz)
        grads["b1"] += dz

    def adam_step(
        self,
        grads: Dict[str, np.ndarray],
        lr: float,
        clip: float = 5.0,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = clip / norm if norm > clip else 1.0
        self._t += 1
        for k, g in grads.items():
            g = g * scale
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            m_hat = self._m[k] / (1 - b1 ** self._t)
            v_hat = self._v[k] / (1 - b2 ** self._t)
            self.params[k] = self.params[k] - lr * m_hat / (np.sqrt(v_hat) + eps)

    def state(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def load(self, state: Dict[str, np.ndarray]) -> None:
        self.params = {k: np.array(v, dtype=float) for k, v in state.items()}

    def save(self, path: str) -> None:
        np.savez(path, **self.params)

    @staticmethod
    def from_file(path: str) -> "ValueNet":
        data = np.load(path)
        w1 = data["W1"]
        net = ValueNet(w1.shape[0], w1.shape[1])
        net.load({k: data[k] for k in data.files})
        return net


def resistance_soft_prior(
    env, valid_ids: np.ndarray, scale: float = 0.5
) -> np.ndarray:
    """Soft resistance-compatibility prior in log-probability space.

    Positive bias = preferred; negative = discouraged.
    Added to policy logits before softmax.
    """
    biases = np.zeros(len(valid_ids), dtype=np.float32)
    ev = env.evaluation
    sr = ev.solve
    target = env.target
    lib = env.library

    for i, act in enumerate(valid_ids):
        if act == env.stop_action:
            continue
        cell_idx, vi = divmod(act, env.V)
        r, c = divmod(cell_idx, env.N)
        v = lib[vi]

        d_outlets = [abs(r - or_) + abs(c - oc) for or_, oc in env.outlet_cells]
        k_nearest = int(np.argmin(d_outlets))

        if sr.ok and not sr.stranded and sr.outlet_flows[k_nearest] > 0:
            flow_deficit = target[k_nearest] - sr.outlet_flows[k_nearest]
            v_norm = v.total_r / env.max_r
            if flow_deficit > 0:
                biases[i] = (1.0 - v_norm) * scale
            else:
                biases[i] = v_norm * scale

    return biases


@dataclass
class RolloutStep:
    global_x: np.ndarray    # (global_dim,) global features (for ValueNet)
    action_x: np.ndarray    # (m, feature_dim) all-action features (for PolicyNet)
    action_h: np.ndarray    # (m, hidden) hidden activations
    action_p: np.ndarray    # (m,) policy probabilities
    action_j: int           # index of chosen action
    log_prob: float         # log(action_p[action_j])
    reward: float
    value: float            # V(s) at this step
    done: bool
    prior: Optional[np.ndarray] = None  # optional soft prior logits


def rollout_ppo(
    env,
    policy_net: PolicyNet,
    value_net: ValueNet,
    rng: np.random.Generator,
    global_dim: int = 18,
    greedy: bool = False,
    use_soft_prior: bool = True,
) -> Tuple[List[RolloutStep], float]:
    """Roll out one episode and record RolloutStep transitions for PPO training."""
    traj: List[RolloutStep] = []
    while not env.done:
        x, ids = env.features()
        if len(ids) == 0:
            break
        global_x = x[0, :global_dim].astype(float)

        s, h = policy_net.forward(x)
        prior = resistance_soft_prior(env, ids, scale=0.5) if use_soft_prior else None
        if prior is not None:
            s = s + prior

        p = softmax(s)
        j = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        log_prob = float(np.log(np.clip(p[j], 1e-12, 1.0)))

        v, _ = value_net.forward(global_x)
        r, done, _ = env.step(int(ids[j]))

        traj.append(RolloutStep(
            global_x=global_x,
            action_x=x.astype(np.float32),
            action_h=h,
            action_p=p,
            action_j=j,
            log_prob=log_prob,
            reward=r,
            value=v,
            done=done,
            prior=prior,
        ))
    return traj, env.evaluation.cost


def compute_gae(
    traj: List[RolloutStep],
    value_net: ValueNet,
    global_dim: int = 18,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) and returns-to-go."""
    T = len(traj)
    if T == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    last_step = traj[-1]
    if last_step.done:
        v_next = 0.0
    else:
        v_next, _ = value_net.forward(last_step.global_x)

    adv = np.zeros(T, dtype=float)
    gae = 0.0
    values = [s.value for s in traj] + [v_next]

    for t in reversed(range(T)):
        delta = traj[t].reward + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae

    returns = adv + np.array([s.value for s in traj], dtype=float)
    return adv, returns

