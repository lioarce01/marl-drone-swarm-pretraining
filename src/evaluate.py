"""
evaluate.py — Load a trained checkpoint and run rigorous evaluation.

Usage:
    python src/evaluate.py --checkpoint logs/mappo_mlp/checkpoints/ckpt_best.pt
    python src/evaluate.py --checkpoint logs/mappo_gnn/checkpoints/ckpt_best.pt --episodes 100

Outputs:
    - Console: formatted metrics table
    - logs/<run>/eval_report.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os
import numpy as np
import torch
from collections import defaultdict

from src.train import load_config
from src.networks.gnn_actor_critic import build_actor_critic

try:
    from src.envs.swarm_coverage_env import SwarmCoverageEnv
    from src.envs.epymarl_wrapper import EPyMARLWrapper
    _ENV_AVAILABLE = True
except ImportError:
    _ENV_AVAILABLE = False


def evaluate_checkpoint(
    checkpoint_path: str,
    n_episodes: int = 100,
    override_config: dict = None,
) -> dict:
    """
    Load checkpoint and evaluate over n_episodes.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        n_episodes:      Number of deterministic evaluation episodes.
        override_config: Optional config overrides (e.g. to change n_agents).

    Returns:
        metrics: dict with mean/std of all evaluation metrics.
    """
    if not _ENV_AVAILABLE:
        raise ImportError("gym-pybullet-drones not installed.")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    if override_config:
        config.update(override_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_raw = SwarmCoverageEnv(config)
    env     = EPyMARLWrapper(env_raw)
    env.reset()

    obs_dim    = env.get_obs_size()
    state_dim  = env.get_state_size()
    action_dim = env.get_total_actions()
    n_agents   = env.n_agents

    model = build_actor_critic(obs_dim, state_dim, action_dim, config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Evaluating {checkpoint_path}")
    print(f"  Agents: {n_agents} | Grid: {env_raw.grid_size}x{env_raw.grid_size} "
          f"| Stage: {env.get_current_stage()} | Episodes: {n_episodes}")
    print("-" * 60)

    ep_coverage, ep_collisions, ep_lengths, ep_rewards = [], [], [], []
    ep_time_to_80 = []  # steps to reach 80% coverage (None if never reached)

    for ep in range(n_episodes):
        env.reset()
        obs_list = env.get_obs()
        ep_reward = 0.0
        ep_col = 0
        ep_len = 0
        time_to_80 = None

        done = False
        while not done:
            obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
            with torch.no_grad():
                actions_tensor, _ = model.get_actions(obs_tensor, deterministic=True)
            actions_np = actions_tensor.cpu().numpy()

            reward, done, info = env.step([actions_np[i] for i in range(n_agents)])
            obs_list = env.get_obs()
            ep_reward += reward
            ep_col    += int(info.get("collision", False))
            ep_len    += 1

            if time_to_80 is None and info.get("coverage_pct", 0.0) >= 0.80:
                time_to_80 = ep_len

        final_coverage = info.get("coverage_pct", 0.0)
        ep_coverage.append(final_coverage)
        ep_collisions.append(ep_col)
        ep_lengths.append(ep_len)
        ep_rewards.append(ep_reward)
        ep_time_to_80.append(time_to_80)

        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d}/{n_episodes} | "
                  f"cov={final_coverage:.3f} col={ep_col} len={ep_len}")

    env.close()

    # Compute metrics
    t80_valid = [t for t in ep_time_to_80 if t is not None]
    metrics = {
        "checkpoint": checkpoint_path,
        "n_episodes": n_episodes,
        "coverage_pct_mean":        float(np.mean(ep_coverage)),
        "coverage_pct_std":         float(np.std(ep_coverage)),
        "coverage_pct_min":         float(np.min(ep_coverage)),
        "coverage_pct_max":         float(np.max(ep_coverage)),
        "collision_rate_mean":      float(np.mean(ep_collisions)),
        "collision_rate_std":       float(np.std(ep_collisions)),
        "episode_length_mean":      float(np.mean(ep_lengths)),
        "episode_reward_mean":      float(np.mean(ep_rewards)),
        "success_rate_80pct":       float(np.mean([c >= 0.80 for c in ep_coverage])),
        "success_rate_70pct":       float(np.mean([c >= 0.70 for c in ep_coverage])),
        "time_to_80pct_mean":       float(np.mean(t80_valid)) if t80_valid else None,
        "time_to_80pct_std":        float(np.std(t80_valid)) if t80_valid else None,
        "time_to_80pct_rate":       float(len(t80_valid) / n_episodes),
        "n_agents":                 n_agents,
        "grid_size":                env_raw.grid_size,
    }

    # Pretty-print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Coverage (mean ± std):    {metrics['coverage_pct_mean']:.3f} ± {metrics['coverage_pct_std']:.3f}")
    print(f"  Coverage (min / max):     {metrics['coverage_pct_min']:.3f} / {metrics['coverage_pct_max']:.3f}")
    print(f"  Success rate (≥80%):      {metrics['success_rate_80pct']:.1%}")
    print(f"  Success rate (≥70%):      {metrics['success_rate_70pct']:.1%}")
    print(f"  Collisions (mean):        {metrics['collision_rate_mean']:.2f}")
    if metrics["time_to_80pct_mean"] is not None:
        print(f"  Steps to 80% (mean):      {metrics['time_to_80pct_mean']:.0f}")
    print(f"  Episode length (mean):    {metrics['episode_length_mean']:.0f}")
    print(f"  Episode reward (mean):    {metrics['episode_reward_mean']:.3f}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained MARL checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes",   type=int, default=100)
    parser.add_argument("--output",     type=str, default=None,
                        help="Path to save JSON report (defaults to checkpoint dir)")
    args = parser.parse_args()

    metrics = evaluate_checkpoint(args.checkpoint, args.episodes)

    # Save report
    out_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "eval_report.json"
    )
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nReport saved to: {out_path}")
