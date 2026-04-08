"""
swarm_coverage_env.py — Multi-agent quadcopter area coverage environment.

Built on top of gym-pybullet-drones MultiHoverAviary (v2.0 API).

Observation space per drone (21-dim):
  [pos_x, pos_y, pos_z,            (3) own position (GPS-noisy in DR mode)
   vel_x, vel_y, vel_z,            (3) own velocity
   roll, pitch, yaw,               (3) own Euler angles
   rel_pos_n1(3), rel_pos_n2(3),   (6) relative positions of 2 nearest neighbours
   dist_n1, dist_n2,               (2) distances to neighbours
   local_coverage_pct,             (1) fraction of local window already covered
   steps_remaining_norm]           (1) episode progress in [0, 1]

Action space per drone (4-dim continuous in [-1, 1]):
  [vx, vy, vz, yaw_rate] — velocity setpoints scaled to physical limits
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any

from src.envs.coverage_map import CoverageMap
from src.envs.domain_rand import DomainRandomizer, DRParams
from src.rewards.coverage_reward import CoverageReward

try:
    from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
    _GYM_PB_AVAILABLE = True
except ImportError as e:
    _GYM_PB_AVAILABLE = False
    _GYM_PB_ERROR = str(e)
    MultiHoverAviary = object

# Physical limits for velocity setpoint scaling
_VXY_MAX  = 2.0          # m/s horizontal
_VZ_MAX   = 1.0          # m/s vertical
_YAW_MAX  = np.pi / 4    # rad/s


class SwarmCoverageEnv:
    """
    Cooperative area coverage environment for N quadcopter drones.

    Wraps gym-pybullet-drones MultiHoverAviary and exposes:
      - gymnasium-style reset() / step() interface
      - Per-agent observations and rewards
      - Global state for centralized critic
      - Curriculum stage control
      - Domain randomization support
    """

    def __init__(self, config: dict):
        if not _GYM_PB_AVAILABLE:
            raise ImportError(
                f"gym-pybullet-drones not installed or import failed: {_GYM_PB_ERROR}\n"
                "Run: pip install \"git+https://github.com/utiasDSL/gym-pybullet-drones.git\""
            )

        env_cfg = config.get("env", {})
        self.n_agents:      int   = env_cfg.get("n_agents", 3)
        self.grid_size:     int   = env_cfg.get("grid_size", 10)
        self.cell_size:     float = env_cfg.get("cell_size", 1.0)
        self.sensor_radius: float = env_cfg.get("sensor_radius", 1.5)
        self.arena_height:  float = env_cfg.get("arena_height", 3.0)
        self.max_steps:     int   = env_cfg.get("max_steps", 2000)
        self.seed_val:      int   = env_cfg.get("seed", 42)

        self.agents:          List[str] = [f"drone_{i}" for i in range(self.n_agents)]
        self.possible_agents: List[str] = list(self.agents)

        # Curriculum
        self._curriculum_cfg:    List[dict] = config.get("curriculum", {}).get("stages", [])
        self._current_stage_idx: int = max(0, config.get("curriculum", {}).get("start_stage", 1) - 1)
        self._stage_cfg:         dict = self._get_stage_cfg(self._current_stage_idx)

        # Sub-modules
        self._coverage_map = CoverageMap(self.grid_size, self.cell_size, self.sensor_radius)
        self._reward_fn    = CoverageReward(n_agents=self.n_agents, **config.get("reward", {}))
        self._domain_rand  = DomainRandomizer(config.get("domain_randomization", {}))
        self._dr_params:   DRParams = DRParams()

        obs_dim = self._obs_dim()
        self.observation_spaces: Dict[str, spaces.Box] = {
            a: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for a in self.agents
        }
        self.action_spaces: Dict[str, spaces.Box] = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            for a in self.agents
        }
        self._global_state_dim = obs_dim * self.n_agents + self.grid_size ** 2

        self._pybullet_env: Optional[MultiHoverAviary] = None
        self._step_count:   int = 0
        self._prev_actions: Optional[np.ndarray] = None
        self._obs_cache:    Optional[Dict[str, np.ndarray]] = None
        self._true_positions: Optional[np.ndarray] = None  # (N, 3)

        np.random.seed(self.seed_val)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        if seed is not None:
            np.random.seed(seed)

        self._dr_params = self._domain_rand.sample()
        self._apply_stage_cfg()
        self._coverage_map.reset()
        self._reward_fn.reset()
        self._step_count    = 0
        self._prev_actions  = None

        if self._pybullet_env is not None:
            self._pybullet_env.close()

        self._pybullet_env = self._build_pybullet_env()
        raw_obs, _ = self._pybullet_env.reset(seed=seed)  # (N, 72)

        # Apply mass randomization after reset (DR Phase 4)
        if self._dr_params.drone_mass_multiplier != 1.0:
            import pybullet as p
            client_id = self._pybullet_env.CLIENT
            for drone_id in self._pybullet_env.DRONE_IDS:
                base_mass = p.getDynamicsInfo(drone_id, -1, physicsClientId=client_id)[0]
                p.changeDynamics(
                    drone_id, -1,
                    mass=base_mass * self._dr_params.drone_mass_multiplier,
                    physicsClientId=client_id,
                )

        self._refresh_positions()
        obs = self._build_obs(raw_obs)
        self._obs_cache = obs
        return obs, {a: {} for a in self.agents}

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        # Build (N, 4) numpy array from action dict, clip and scale
        action_array   = np.array([actions[a] for a in self.agents], dtype=np.float32)
        action_array   = np.clip(action_array, -1.0, 1.0)
        velocity_cmds  = self._scale_actions(action_array)

        # Domain randomization: motor noise + wind
        if self._dr_params.motor_noise_std > 0.0:
            velocity_cmds += np.random.normal(0, self._dr_params.motor_noise_std,
                                              velocity_cmds.shape).astype(np.float32)
        velocity_cmds[:, :2] += self._dr_params.wind_force[:2]

        # Step PyBullet — takes (N, 4) array, returns (N, 72) obs
        raw_obs, _, raw_term, raw_trunc, raw_info = self._pybullet_env.step(velocity_cmds)

        self._step_count += 1
        self._refresh_positions()

        # Noisy positions for observations (GPS noise)
        noisy_pos = self._domain_rand.apply_gps_noise(self._true_positions, self._dr_params)

        # Update coverage
        delta_coverage = self._coverage_map.update(self._true_positions)
        coverage_pct   = self._coverage_map.coverage_pct()

        # Collisions — MultiHoverAviary doesn't track per-agent collisions explicitly;
        # approximate by checking if any drone altitude drops critically
        collisions = self._detect_collisions()

        # Compute rewards
        reward_array = self._reward_fn.compute(
            delta_coverage=delta_coverage,
            positions=self._true_positions,
            prev_actions=self._prev_actions,
            curr_actions=action_array,
            collisions=collisions,
            coverage_map=self._coverage_map,
            coverage_pct=coverage_pct,
        )
        self._prev_actions = action_array

        obs = self._build_obs(raw_obs, noisy_pos=noisy_pos)
        self._obs_cache = obs

        done = (
            coverage_pct >= self._reward_fn.done_threshold
            or bool(raw_term)
            or self._step_count >= self.max_steps
            or bool(collisions.any())   # terminate immediately if any drone crashes
        )

        terminateds = {a: done for a in self.agents}
        terminateds["__all__"] = done
        truncateds  = {a: False for a in self.agents}
        truncateds["__all__"] = False

        rewards = {a: float(reward_array[i]) for i, a in enumerate(self.agents)}
        infos   = {
            a: {
                "coverage_pct":   coverage_pct,
                "delta_coverage": delta_coverage,
                "collision":      bool(collisions[i]),
                "step":           self._step_count,
            }
            for i, a in enumerate(self.agents)
        }
        return obs, rewards, terminateds, truncateds, infos

    def close(self):
        if self._pybullet_env is not None:
            self._pybullet_env.close()
            self._pybullet_env = None

    # ------------------------------------------------------------------
    # EPyMARL / CTDE interface
    # ------------------------------------------------------------------

    def get_obs(self) -> List[np.ndarray]:
        if self._obs_cache is None:
            raise RuntimeError("Call reset() before get_obs()")
        return [self._obs_cache[a] for a in self.agents]

    def get_state(self) -> np.ndarray:
        """Global state for centralized critic = concat(all obs) + coverage grid."""
        agent_obs = self.get_obs()
        grid = self._coverage_map.get_flat_grid()
        return np.concatenate(agent_obs + [grid], axis=0).astype(np.float32)

    def get_state_size(self) -> int:
        return self._global_state_dim

    def get_obs_size(self) -> int:
        return self._obs_dim()

    def get_total_actions(self) -> int:
        return 4

    def get_avail_actions(self) -> List[np.ndarray]:
        return [np.ones(4, dtype=np.float32) for _ in self.agents]

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def get_current_stage(self) -> int:
        return self._current_stage_idx + 1

    def advance_curriculum(self) -> bool:
        if self._current_stage_idx >= len(self._curriculum_cfg) - 1:
            return False
        self._current_stage_idx += 1
        self._stage_cfg = self._get_stage_cfg(self._current_stage_idx)
        self._apply_stage_cfg()
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _obs_dim(self) -> int:
        # pos(3) + vel(3) + euler(3) + 2×neighbor_rel_pos(3) + 2×dist(2)
        # + local_cov(1) + progress(1) + nearest_uncovered_dir(2)
        # 3 + 3 + 3 + 3 + 3 + 2 + 1 + 1 + 2 = 21
        return 21

    def _build_pybullet_env(self) -> MultiHoverAviary:
        arena = self.grid_size * self.cell_size
        initial_xyzs = self._sample_start_positions(arena)
        return MultiHoverAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.n_agents,
            initial_xyzs=initial_xyzs,
            initial_rpys=np.zeros((self.n_agents, 3)),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=24,
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
        )

    def _sample_start_positions(self, arena: float) -> np.ndarray:
        margin = 1.5
        noise  = self._dr_params.start_pos_noise if self._dr_params else 0.0
        positions = []
        for i in range(self.n_agents):
            x = margin + (i + 1) * (arena - 2 * margin) / (self.n_agents + 1)
            y = arena / 2.0
            x = float(np.clip(x + np.random.uniform(-noise, noise), margin, arena - margin))
            y = float(np.clip(y + np.random.uniform(-noise, noise), margin, arena - margin))
            positions.append([x, y, self.arena_height])
        return np.array(positions, dtype=np.float32)

    def _refresh_positions(self):
        """Read current true positions from PyBullet state vectors."""
        self._true_positions = np.array(
            [np.array(self._pybullet_env._getDroneStateVector(i))[:3]
             for i in range(self.n_agents)],
            dtype=np.float32
        )  # (N, 3)

    def _build_obs(
        self,
        raw_obs: np.ndarray,            # (N, 72) from MultiHoverAviary
        noisy_pos: Optional[np.ndarray] = None,  # (N, 3)
    ) -> Dict[str, np.ndarray]:
        """Build 22-dim per-agent observation vectors."""
        steps_norm = float(self._step_count) / float(self.max_steps)

        obs_positions = noisy_pos if noisy_pos is not None else self._true_positions

        obs_dict: Dict[str, np.ndarray] = {}
        for i, agent in enumerate(self.agents):
            # Parse state vector: [x,y,z, q0-3, r,p,y, vx,vy,vz, wx,wy,wz, act×4]
            sv = np.array(self._pybullet_env._getDroneStateVector(i), dtype=np.float32)
            own_pos   = obs_positions[i]   # (3,) — possibly GPS-noisy
            own_vel   = sv[10:13]          # (3,) vx, vy, vz
            own_euler = sv[7:10]           # (3,) roll, pitch, yaw

            n1_rel, n2_rel, d1, d2 = self._neighbour_obs(i, obs_positions)
            local_cov = self._coverage_map.local_coverage_pct(own_pos)
            uncov_dir = self._coverage_map.nearest_uncovered_direction(own_pos)  # (2,) unit vec

            obs_dict[agent] = np.concatenate([
                own_pos,         # 3
                own_vel,         # 3
                own_euler,       # 3
                n1_rel,          # 3
                n2_rel,          # 3
                [d1, d2],        # 2
                [local_cov],     # 1
                [steps_norm],    # 1
                uncov_dir,       # 2  ← direction to nearest uncovered cell
            ]).astype(np.float32)  # = 21

        return obs_dict

    def _neighbour_obs(
        self, idx: int, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        own = positions[idx]
        others = [(j, positions[j]) for j in range(self.n_agents) if j != idx]
        if not others:
            return np.zeros(3), np.zeros(3), 0.0, 0.0

        dists = sorted([(j, float(np.linalg.norm(own - p))) for j, p in others], key=lambda x: x[1])
        n1_rel = (positions[dists[0][0]] - own).astype(np.float32)
        d1     = dists[0][1]
        if len(dists) >= 2:
            n2_rel = (positions[dists[1][0]] - own).astype(np.float32)
            d2     = dists[1][1]
        else:
            n2_rel = np.zeros(3, dtype=np.float32)
            d2     = 0.0
        return n1_rel, n2_rel, d1, d2

    def _scale_actions(self, actions: np.ndarray) -> np.ndarray:
        scaled = np.zeros_like(actions)
        scaled[:, 0] = actions[:, 0] * _VXY_MAX
        scaled[:, 1] = actions[:, 1] * _VXY_MAX
        scaled[:, 2] = actions[:, 2] * _VZ_MAX
        scaled[:, 3] = actions[:, 3] * _YAW_MAX
        return scaled

    def _detect_collisions(self) -> np.ndarray:
        """Mark a drone as collided if altitude drops below safe threshold."""
        altitudes  = self._true_positions[:, 2]
        collisions = altitudes < 0.15  # dangerously low altitude
        return collisions.astype(bool)

    def _get_stage_cfg(self, idx: int) -> dict:
        if not self._curriculum_cfg or idx >= len(self._curriculum_cfg):
            return {}
        return self._curriculum_cfg[idx]

    def _apply_stage_cfg(self):
        stage = self._stage_cfg
        if not stage:
            return
        if "grid_size" in stage:
            self.grid_size = stage["grid_size"]
            self._coverage_map = CoverageMap(self.grid_size, self.cell_size, self.sensor_radius)
        if "n_agents" in stage:
            self.n_agents   = stage["n_agents"]
            self.agents     = [f"drone_{i}" for i in range(self.n_agents)]
            self.possible_agents = list(self.agents)
            obs_dim = self._obs_dim()
            self.observation_spaces = {
                a: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                for a in self.agents
            }
            self.action_spaces = {
                a: spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
                for a in self.agents
            }
            self._global_state_dim = obs_dim * self.n_agents + self.grid_size ** 2
            self._reward_fn.n_agents = self.n_agents
