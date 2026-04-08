"""
domain_rand.py — Per-episode domain randomization for sim-to-real robustness.

All ranges are sampled uniformly at the start of each episode.
Validated by "One Net to Rule Them All" (2025) and prior sim-to-real literature.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DRParams:
    """Sampled randomization parameters for one episode."""
    drone_mass_multiplier: float = 1.0      # applied to PyBullet drone mass
    motor_noise_std: float = 0.0            # std of gaussian noise on each motor thrust
    gps_noise_std: float = 0.0             # std of gaussian noise on position observation (m)
    wind_force: np.ndarray = field(default_factory=lambda: np.zeros(3))  # constant wind (N)
    obstacle_jitter: float = 0.0           # max offset for obstacle positions (m) — unused until obstacles are implemented in env
    start_pos_noise: float = 0.0           # max offset for drone start positions (m)


class DomainRandomizer:
    """
    Samples DR parameters at episode reset.
    Pass the returned DRParams to the environment to apply.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: domain_randomization section from YAML config.
        """
        self.enabled = config.get("enabled", False)
        self.mass_range:         Tuple[float, float] = tuple(config.get("drone_mass_range", [1.0, 1.0]))
        self.motor_noise_range:  Tuple[float, float] = tuple(config.get("motor_noise_std_range", [0.0, 0.0]))
        self.gps_noise_range:    Tuple[float, float] = tuple(config.get("gps_noise_std_range", [0.0, 0.0]))
        self.wind_range:         Tuple[float, float] = tuple(config.get("wind_force_range", [0.0, 0.0]))
        self.obstacle_jitter_range: Tuple[float, float] = tuple(config.get("obstacle_jitter_range", [0.0, 0.0]))
        self.start_pos_noise_range: Tuple[float, float] = tuple(config.get("start_pos_noise_range", [0.0, 0.0]))

    def sample(self) -> DRParams:
        """Sample fresh DR parameters for one episode."""
        if not self.enabled:
            return DRParams()

        return DRParams(
            drone_mass_multiplier=float(np.random.uniform(*self.mass_range)),
            motor_noise_std=float(np.random.uniform(*self.motor_noise_range)),
            gps_noise_std=float(np.random.uniform(*self.gps_noise_range)),
            wind_force=np.array([
                np.random.uniform(*self.wind_range),
                np.random.uniform(*self.wind_range),
                0.0,  # no vertical wind (keeps drones at altitude)
            ], dtype=np.float32),
            obstacle_jitter=float(np.random.uniform(*self.obstacle_jitter_range)),
            start_pos_noise=float(np.random.uniform(*self.start_pos_noise_range)),
        )

    def apply_gps_noise(self, positions: np.ndarray, params: DRParams) -> np.ndarray:
        """
        Add Gaussian position noise to simulate GPS error.

        Args:
            positions: (N, 3) true drone positions.
            params:    Current episode DR parameters.

        Returns:
            noisy_positions: (N, 3) — what the drone "observes" as its position.
        """
        if params.gps_noise_std <= 0.0:
            return positions
        noise = np.random.normal(0.0, params.gps_noise_std, size=positions.shape).astype(np.float32)
        noise[:, 2] *= 0.5  # less altitude noise (barometer more accurate than GPS)
        return positions + noise

    # NOTE: motor noise is applied directly in SwarmCoverageEnv.step() as additive
    # Gaussian noise on velocity commands (ActionType.VEL). No separate method needed.
