"""
coverage_reward.py — Reward function for cooperative area coverage.

Reward components (all configurable via weights in config YAML):
  + w_cov  * delta_coverage      : team reward for newly explored cells
  + w_prox * proximity_bonus     : potential-based shaping toward uncovered cells
  - w_col  * collision_count     : penalty per collision event
  - w_jerk * action_smoothness   : penalty for abrupt control changes
  + w_done * terminal_bonus      : one-time bonus when coverage >= done_threshold
"""

import numpy as np
from typing import List, Optional


class CoverageReward:
    def __init__(
        self,
        w_cov: float = 0.5,
        w_prox: float = 0.0,
        w_col: float = 5.0,
        w_jerk: float = 0.001,
        w_done: float = 10.0,
        w_step: float = 0.01,
        done_threshold: float = 0.90,
        n_agents: int = 3,
    ):
        self.w_cov = w_cov
        self.w_prox = w_prox
        self.w_col = w_col
        self.w_jerk = w_jerk
        self.w_done = w_done
        self.w_step = w_step
        self.done_threshold = done_threshold
        self.n_agents = n_agents

        # Potential-based shaping state (Ng 1999) — distance to nearest uncovered cell
        self._prev_potentials: Optional[np.ndarray] = None  # (N,)
        self._terminal_bonus_given = False

    def reset(self) -> None:
        """Call at the start of each episode."""
        self._prev_potentials = None
        self._terminal_bonus_given = False

    def compute(
        self,
        delta_coverage: float,
        positions: np.ndarray,          # (N, 3)
        prev_actions: Optional[np.ndarray],  # (N, 4) — previous step actions
        curr_actions: np.ndarray,       # (N, 4)
        collisions: np.ndarray,         # (N,) bool — True if drone i collided this step
        coverage_map,                   # CoverageMap instance
        coverage_pct: float,
    ) -> np.ndarray:
        """
        Compute per-agent reward vector.

        All agents receive the same team reward (shared delta_coverage) plus
        individual shaping terms. CTDE critic sees the full global state; actors
        see only local observations — so per-agent rewards are correct under CTDE.

        Returns:
            rewards: (N,) float32 array of per-agent rewards.
        """
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        # --- 1. Team coverage reward ---
        team_coverage_reward = self.w_cov * delta_coverage
        rewards += team_coverage_reward

        # --- 2. Potential-based proximity shaping (per agent) ---
        # Potential Φ(s) = -distance to nearest uncovered cell (negative distance)
        # Shaping: F = γ*Φ(s') - Φ(s); with γ≈1 this is just the delta.
        # Guarantees optimal policy invariance (Ng et al. 1999).
        potentials = np.array([
            -self._nearest_uncovered_distance(positions[i], coverage_map)
            for i in range(self.n_agents)
        ], dtype=np.float32)

        if self._prev_potentials is not None:
            shaping = self.w_prox * (0.99 * potentials - self._prev_potentials)
            rewards += shaping

        self._prev_potentials = potentials

        # --- 3. Collision penalty (per agent) ---
        rewards -= self.w_col * collisions.astype(np.float32)

        # --- 4. Action smoothness penalty (anti-jerk, per agent) ---
        if prev_actions is not None:
            jerk = np.linalg.norm(curr_actions - prev_actions, axis=-1)  # (N,)
            rewards -= self.w_jerk * jerk

        # --- 5. Step cost (shared) — small constant penalty each step ---
        rewards -= self.w_step

        # --- 6. Terminal bonus (shared, one-time) ---
        if coverage_pct >= self.done_threshold and not self._terminal_bonus_given:
            rewards += self.w_done
            self._terminal_bonus_given = True

        return rewards

    def component_breakdown(
        self,
        delta_coverage: float,
        positions: np.ndarray,
        prev_actions: Optional[np.ndarray],
        curr_actions: np.ndarray,
        collisions: np.ndarray,
        coverage_map,
        coverage_pct: float,
    ) -> dict:
        """Return a dict of each reward component for logging/debugging."""
        team_cov = self.w_cov * delta_coverage
        potentials = np.array([
            -self._nearest_uncovered_distance(positions[i], coverage_map)
            for i in range(self.n_agents)
        ])
        shaping = np.zeros(self.n_agents)
        if self._prev_potentials is not None:
            shaping = self.w_prox * (0.99 * potentials - self._prev_potentials)

        col_pen = self.w_col * collisions.astype(float)
        jerk_pen = np.zeros(self.n_agents)
        if prev_actions is not None:
            jerk_pen = self.w_jerk * np.linalg.norm(curr_actions - prev_actions, axis=-1)

        terminal = self.w_done if (coverage_pct >= self.done_threshold and not self._terminal_bonus_given) else 0.0

        return {
            "team_coverage": float(team_cov),
            "proximity_shaping": shaping.tolist(),
            "collision_penalty": col_pen.tolist(),
            "jerk_penalty": jerk_pen.tolist(),
            "terminal_bonus": float(terminal),
        }

    @staticmethod
    def _nearest_uncovered_distance(position: np.ndarray, coverage_map) -> float:
        """Distance (metres) from drone to nearest uncovered cell centre."""
        direction = coverage_map.nearest_uncovered_direction(position)
        if np.linalg.norm(direction) < 1e-6:
            return 0.0  # all covered

        pos_xy = position[:2]
        uncovered_mask = coverage_map._grid == 0.0
        if not uncovered_mask.any():
            return 0.0

        centres = coverage_map._cell_centres[uncovered_mask]
        dists = np.linalg.norm(centres - pos_xy, axis=-1)
        return float(dists.min())
