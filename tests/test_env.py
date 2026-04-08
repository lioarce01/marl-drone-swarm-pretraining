"""
tests/test_env.py — Smoke tests for SwarmCoverageEnv and sub-modules.

Run with: pytest tests/test_env.py -v
These tests run WITHOUT gym-pybullet-drones (testing sub-modules in isolation).
Full integration tests require gym-pybullet-drones installed.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.coverage_map import CoverageMap
from src.envs.domain_rand import DomainRandomizer, DRParams
from src.rewards.coverage_reward import CoverageReward


# =====================================================================
# CoverageMap tests
# =====================================================================

class TestCoverageMap:
    def setup_method(self):
        self.cmap = CoverageMap(grid_size=10, cell_size=1.0, sensor_radius=1.5)

    def test_initial_coverage_zero(self):
        assert self.cmap.coverage_pct() == 0.0

    def test_reset_clears_grid(self):
        pos = np.array([[5.0, 5.0, 3.0]])
        self.cmap.update(pos)
        assert self.cmap.coverage_pct() > 0.0
        self.cmap.reset()
        assert self.cmap.coverage_pct() == 0.0

    def test_update_marks_cells(self):
        pos = np.array([[5.0, 5.0, 3.0]])
        delta = self.cmap.update(pos)
        assert delta > 0.0
        assert self.cmap.coverage_pct() > 0.0

    def test_update_no_double_count(self):
        pos = np.array([[5.0, 5.0, 3.0]])
        delta1 = self.cmap.update(pos)
        delta2 = self.cmap.update(pos)   # same position — no new cells
        assert delta2 == 0.0

    def test_full_coverage_possible(self):
        # Move a drone across the whole grid
        for x in np.arange(0.5, 10.0, 1.0):
            for y in np.arange(0.5, 10.0, 1.0):
                self.cmap.update(np.array([[x, y, 3.0]]))
        assert self.cmap.coverage_pct() == pytest.approx(1.0, abs=0.01)

    def test_local_coverage_pct_in_range(self):
        pos = np.array([5.0, 5.0, 3.0])
        self.cmap.update(pos[np.newaxis])
        local = self.cmap.local_coverage_pct(pos)
        assert 0.0 <= local <= 1.0

    def test_nearest_uncovered_direction_unit_vector(self):
        direction = self.cmap.nearest_uncovered_direction(np.array([5.0, 5.0, 3.0]))
        norm = np.linalg.norm(direction)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit vector, got norm={norm}"

    def test_nearest_uncovered_direction_all_covered(self):
        # Cover everything
        for x in np.arange(0.5, 10.0, 1.0):
            for y in np.arange(0.5, 10.0, 1.0):
                self.cmap.update(np.array([[x, y, 3.0]]))
        direction = self.cmap.nearest_uncovered_direction(np.array([5.0, 5.0, 3.0]))
        assert np.linalg.norm(direction) == pytest.approx(0.0, abs=1e-6)

    def test_get_flat_grid_shape(self):
        flat = self.cmap.get_flat_grid()
        assert flat.shape == (100,)  # 10x10

    def test_multi_drone_update(self):
        positions = np.array([
            [2.0, 2.0, 3.0],
            [7.0, 7.0, 3.0],
            [5.0, 5.0, 3.0],
        ])
        delta = self.cmap.update(positions)
        assert delta > 0.0
        # Three drones across the grid should cover substantial area
        assert self.cmap.coverage_pct() > 0.05


# =====================================================================
# DomainRandomizer tests
# =====================================================================

class TestDomainRandomizer:
    def test_disabled_returns_defaults(self):
        dr = DomainRandomizer({"enabled": False})
        params = dr.sample()
        assert params.drone_mass_multiplier == 1.0
        assert params.motor_noise_std == 0.0
        assert params.gps_noise_std == 0.0
        assert np.all(params.wind_force == 0.0)

    def test_enabled_samples_in_range(self):
        cfg = {
            "enabled": True,
            "drone_mass_range": [0.9, 1.1],
            "motor_noise_std_range": [0.0, 0.02],
            "gps_noise_std_range": [0.0, 0.1],
            "wind_force_range": [-0.05, 0.05],
            "obstacle_jitter_range": [-0.5, 0.5],
            "start_pos_noise_range": [-1.0, 1.0],
        }
        dr = DomainRandomizer(cfg)
        for _ in range(20):
            params = dr.sample()
            assert 0.9 <= params.drone_mass_multiplier <= 1.1
            assert 0.0 <= params.motor_noise_std <= 0.02
            assert 0.0 <= params.gps_noise_std <= 0.1
            assert -0.05 <= params.wind_force[0] <= 0.05
            assert params.wind_force[2] == 0.0  # no vertical wind

    def test_gps_noise_applied(self):
        cfg = {"enabled": True, "gps_noise_std_range": [0.1, 0.1]}
        dr = DomainRandomizer(cfg)
        params = dr.sample()
        positions = np.ones((3, 3), dtype=np.float32)
        noisy = dr.apply_gps_noise(positions, params)
        assert not np.allclose(positions, noisy), "GPS noise not applied"

    def test_gps_noise_zero_when_std_zero(self):
        dr = DomainRandomizer({"enabled": False})
        params = DRParams(gps_noise_std=0.0)
        positions = np.ones((3, 3), dtype=np.float32)
        noisy = dr.apply_gps_noise(positions, params)
        assert np.allclose(positions, noisy)


# =====================================================================
# CoverageReward tests
# =====================================================================

class TestCoverageReward:
    def setup_method(self):
        self.cmap = CoverageMap(grid_size=10, cell_size=1.0, sensor_radius=1.5)
        self.reward_fn = CoverageReward(
            w_cov=0.5, w_prox=0.1, w_col=5.0, w_jerk=0.01, w_done=2.0,
            done_threshold=0.9, n_agents=3,
        )

    def test_positive_reward_on_new_coverage(self):
        self.reward_fn.reset()
        positions = np.array([[2.0, 2.0, 3.0], [5.0, 5.0, 3.0], [8.0, 8.0, 3.0]])
        delta = self.cmap.update(positions)
        rewards = self.reward_fn.compute(
            delta_coverage=delta,
            positions=positions,
            prev_actions=None,
            curr_actions=np.zeros((3, 4)),
            collisions=np.zeros(3, dtype=bool),
            coverage_map=self.cmap,
            coverage_pct=self.cmap.coverage_pct(),
        )
        assert rewards.shape == (3,)
        # Team coverage reward should be positive
        assert rewards.mean() > 0.0

    def test_collision_penalty_applied(self):
        self.reward_fn.reset()
        positions = np.array([[5.0, 5.0, 3.0]] * 3)
        collisions = np.array([True, False, False], dtype=bool)
        rewards = self.reward_fn.compute(
            delta_coverage=0.0,
            positions=positions,
            prev_actions=None,
            curr_actions=np.zeros((3, 4)),
            collisions=collisions,
            coverage_map=self.cmap,
            coverage_pct=0.0,
        )
        assert rewards[0] < rewards[1], "Colliding drone should get lower reward"

    def test_terminal_bonus_given_once(self):
        self.reward_fn.reset()
        positions = np.array([[5.0, 5.0, 3.0]] * 3)
        # First call at done threshold
        rewards1 = self.reward_fn.compute(
            delta_coverage=0.0, positions=positions, prev_actions=None,
            curr_actions=np.zeros((3, 4)), collisions=np.zeros(3, dtype=bool),
            coverage_map=self.cmap, coverage_pct=0.95,
        )
        # Second call still above threshold — no more bonus
        rewards2 = self.reward_fn.compute(
            delta_coverage=0.0, positions=positions, prev_actions=None,
            curr_actions=np.zeros((3, 4)), collisions=np.zeros(3, dtype=bool),
            coverage_map=self.cmap, coverage_pct=0.95,
        )
        assert rewards1.mean() > rewards2.mean(), "Terminal bonus should only fire once"

    def test_jerk_penalty_applied(self):
        self.reward_fn.reset()
        positions = np.array([[5.0, 5.0, 3.0]] * 3)
        prev = np.zeros((3, 4))
        curr = np.ones((3, 4))  # large action change
        rewards = self.reward_fn.compute(
            delta_coverage=0.0, positions=positions, prev_actions=prev,
            curr_actions=curr, collisions=np.zeros(3, dtype=bool),
            coverage_map=self.cmap, coverage_pct=0.0,
        )
        rewards_no_jerk = CoverageReward(
            w_jerk=0.0, n_agents=3
        )
        rewards_no_jerk.reset()
        rewards_nj = rewards_no_jerk.compute(
            delta_coverage=0.0, positions=positions, prev_actions=prev,
            curr_actions=curr, collisions=np.zeros(3, dtype=bool),
            coverage_map=self.cmap, coverage_pct=0.0,
        )
        assert rewards.mean() < rewards_nj.mean(), "Jerk penalty should reduce reward"

    def test_reward_shape(self):
        self.reward_fn.reset()
        positions = np.array([[2.0, 2.0, 3.0], [5.0, 5.0, 3.0], [8.0, 8.0, 3.0]])
        rewards = self.reward_fn.compute(
            delta_coverage=0.01, positions=positions, prev_actions=None,
            curr_actions=np.zeros((3, 4)), collisions=np.zeros(3, dtype=bool),
            coverage_map=self.cmap, coverage_pct=0.1,
        )
        assert rewards.shape == (3,)
        assert not np.any(np.isnan(rewards)), "Reward contains NaN"
        assert not np.any(np.isinf(rewards)), "Reward contains Inf"
