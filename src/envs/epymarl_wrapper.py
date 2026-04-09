"""
epymarl_wrapper.py — Adapts SwarmCoverageEnv to the EPyMARL MultiAgentEnv interface.

EPyMARL expects:
  - reset() → None (state stored internally)
  - step(actions: list[int|float]) → reward, done, info
  - get_obs() → list[np.ndarray]
  - get_state() → np.ndarray
  - get_obs_size() → int
  - get_state_size() → int
  - get_total_actions() → int
  - get_avail_actions() → list[np.ndarray]
  - n_agents: int
  - episode_limit: int

For continuous actions EPyMARL uses the action as a float array per agent.
"""

import numpy as np
from typing import List, Optional, Tuple, Any


class EPyMARLWrapper:
    """
    Wraps SwarmCoverageEnv for use with EPyMARL's MAPPO/QMIX runners.

    Usage:
        env = EPyMARLWrapper(SwarmCoverageEnv(config))
        env.reset()
        obs = env.get_obs()
        state = env.get_state()
        actions = [policy_i(obs[i]) for i in range(env.n_agents)]
        reward, done, info = env.step(actions)
    """

    def __init__(self, env):
        self._env = env
        self._last_obs: Optional[List[np.ndarray]] = None
        self._last_info: dict = {}
        self._episode_return: float = 0.0
        self._step: int = 0

    # ------------------------------------------------------------------
    # EPyMARL required interface
    # ------------------------------------------------------------------

    @property
    def n_agents(self) -> int:
        return self._env.n_agents

    @property
    def episode_limit(self) -> int:
        return self._env.max_steps

    def get_env_info(self) -> dict:
        """Return environment metadata expected by EPyMARL."""
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def reset(self, seed=None) -> None:
        """Reset the environment. Observations are stored internally."""
        obs_dict, _ = self._env.reset(seed=seed)
        self._last_obs = [obs_dict[a] for a in self._env.agents]
        self._episode_return = 0.0
        self._step = 0

    def step(self, actions: List[np.ndarray]) -> Tuple[float, bool, dict]:
        """
        Step with a list of per-agent actions.

        Args:
            actions: list of (action_dim,) arrays, one per agent.

        Returns:
            reward (float, shared team reward mean),
            done (bool),
            info (dict with extra metrics).
        """
        action_dict = {a: np.asarray(actions[i], dtype=np.float32)
                       for i, a in enumerate(self._env.agents)}

        obs_dict, rewards_dict, terminateds, truncateds, infos = self._env.step(action_dict)

        self._last_obs = [obs_dict[a] for a in self._env.agents]
        self._last_info = infos.get(self._env.agents[0], {})
        self._step += 1

        team_reward = float(np.mean(list(rewards_dict.values())))
        self._episode_return += team_reward
        done = bool(terminateds.get("__all__", False))

        # Any-agent collision: check all agents so the metric is not silently zero
        any_collision = any(
            infos.get(a, {}).get("collision", False) for a in self._env.agents
        )

        info = {
            "coverage_pct": self._last_info.get("coverage_pct", 0.0),
            "delta_coverage": self._last_info.get("delta_coverage", 0.0),
            "collision": any_collision,
            "episode_return": self._episode_return,
            "step": self._step,
        }

        return team_reward, done, info

    def get_obs(self) -> List[np.ndarray]:
        """Return list of per-agent observations."""
        if self._last_obs is None:
            raise RuntimeError("Call reset() before get_obs()")
        return list(self._last_obs)

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        return self._last_obs[agent_id]

    def get_obs_size(self) -> int:
        return self._env.get_obs_size()

    def get_state(self) -> np.ndarray:
        """Return global state for centralized critic."""
        return self._env.get_state()

    def get_state_size(self) -> int:
        return self._env.get_state_size()

    def get_avail_actions(self) -> List[np.ndarray]:
        """All actions available (no masking for continuous)."""
        return self._env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id: int) -> np.ndarray:
        return np.ones(self.get_total_actions(), dtype=np.float32)

    def get_total_actions(self) -> int:
        return self._env.get_total_actions()

    # ------------------------------------------------------------------
    # Curriculum pass-through
    # ------------------------------------------------------------------

    def get_current_stage(self) -> int:
        return self._env.get_current_stage()

    def advance_curriculum(self) -> bool:
        return self._env.advance_curriculum()

    # ------------------------------------------------------------------
    # Rendering / closing
    # ------------------------------------------------------------------

    def render(self) -> Optional[np.ndarray]:
        return None  # visualization handled separately by visualize.py

    def close(self) -> None:
        self._env.close()

    def seed(self, s: int) -> None:
        np.random.seed(s)
