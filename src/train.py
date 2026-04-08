"""
train.py — MAPPO training loop with curriculum learning.

Usage:
    python src/train.py --config config/mappo_mlp.yaml
    python src/train.py --config config/mappo_mlp_dr.yaml --resume logs/mappo_mlp/ckpt_latest.pt

Architecture: Centralized Training, Decentralized Execution (CTDE)
  - Shared actor across all agents (parameter sharing)
  - Centralized critic takes global state (all obs + coverage grid)
  - PPO update with GAE advantage estimation
"""

import sys
import os
# Add project root to path so 'src' is importable regardless of how this is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports (gym-pybullet-drones required for SwarmCoverageEnv)
try:
    from src.envs.swarm_coverage_env import SwarmCoverageEnv
    from src.envs.epymarl_wrapper import EPyMARLWrapper
    _ENV_AVAILABLE = True
except ImportError:
    _ENV_AVAILABLE = False

from src.networks.gnn_actor_critic import build_actor_critic


# =====================================================================
# Config loading
# =====================================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Load base config if specified
    base_path = cfg.pop("_base", None)
    if base_path:
        base_dir = os.path.dirname(path)
        with open(os.path.join(base_dir, base_path), "r") as f:
            base = yaml.safe_load(f)
        base_path_ref = base.pop("_base", None)  # ignore nested base for now
        base.update(cfg)
        cfg = base
    return cfg


# =====================================================================
# Running mean/std normalizer for return normalization (PopArt-style)
# Recommended by Yu et al. MAPPO paper — "often helps and never hurts"
# =====================================================================

class RunningMeanStd:
    """Welford online algorithm for running mean and variance."""
    def __init__(self):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray):
        x = x.flatten().astype(np.float64)
        batch_mean  = x.mean()
        batch_var   = x.var()
        batch_count = x.size
        delta       = batch_mean - self.mean
        tot         = self.count + batch_count
        self.mean  += delta * batch_count / tot
        self.var    = (self.var * self.count + batch_var * batch_count
                       + delta**2 * self.count * batch_count / tot) / tot
        self.count  = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))


# =====================================================================
# Rollout buffer — stores one batch of on-policy trajectories
# =====================================================================

class RolloutBuffer:
    """Stores n_steps of experience for all agents, then computes GAE advantages."""

    def __init__(self, n_steps: int, n_agents: int, obs_dim: int, state_dim: int, action_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.device = device

        # Shape: (n_steps, n_agents, dim)
        self.obs      = torch.zeros(n_steps, n_agents, obs_dim,    device=device)
        self.states   = torch.zeros(n_steps, state_dim,             device=device)
        self.actions  = torch.zeros(n_steps, n_agents, action_dim, device=device)
        self.log_probs = torch.zeros(n_steps, n_agents,             device=device)
        self.rewards   = torch.zeros(n_steps, n_agents,             device=device)
        self.values    = torch.zeros(n_steps,                       device=device)
        self.dones     = torch.zeros(n_steps,                       device=device)

        self.ptr = 0
        self.full = False

    def add(self, obs, state, actions, log_probs, rewards, value, done):
        i = self.ptr
        self.obs[i]       = obs
        self.states[i]    = state
        self.actions[i]   = actions
        self.log_probs[i] = log_probs
        self.rewards[i]   = rewards
        self.values[i]    = value.squeeze(-1)
        self.dones[i]     = float(done)
        self.ptr = (self.ptr + 1) % self.n_steps
        if self.ptr == 0:
            self.full = True

    def compute_gae(self, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and returns in-place."""
        advantages = torch.zeros(self.n_steps, device=self.device)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value.squeeze()
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            # Mean reward across agents for critic target
            mean_reward = self.rewards[t].mean()
            delta = mean_reward + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values

    def get_minibatches(self, n_minibatches: int):
        """Yield shuffled minibatches of (obs, state, actions, log_probs, returns, advantages)."""
        indices = torch.randperm(self.n_steps, device=self.device)
        mb_size = self.n_steps // n_minibatches
        for start in range(0, self.n_steps, mb_size):
            idx = indices[start:start + mb_size]
            yield (
                self.obs[idx],           # (mb, n_agents, obs_dim)
                self.states[idx],        # (mb, state_dim)
                self.actions[idx],       # (mb, n_agents, action_dim)
                self.log_probs[idx],     # (mb, n_agents)
                self.returns[idx],       # (mb,)
                self.advantages[idx],    # (mb,)
            )

    def reset(self):
        self.ptr = 0
        self.full = False


# =====================================================================
# MAPPO Trainer
# =====================================================================

class MAPPOTrainer:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.hyp = config.get("hyperparameters", {})
        self.train_cfg = config.get("training", {})

        self.n_steps      = self.hyp.get("n_steps", 2048)
        self.n_minibatches = self.hyp.get("n_minibatches", 8)
        self.n_epochs     = self.hyp.get("n_epochs", 10)
        self.gamma        = self.hyp.get("gamma", 0.99)
        self.gae_lambda   = self.hyp.get("gae_lambda", 0.95)
        self.clip_eps     = self.hyp.get("clip_epsilon", 0.2)
        self.entropy_coef = self.hyp.get("entropy_coef", 0.01)
        self.vf_coef      = self.hyp.get("value_loss_coef", 0.5)
        self.max_grad_norm = self.hyp.get("max_grad_norm", 10.0)

        self.model = None
        self.optimizer: torch.optim.Adam = None
        self.ret_rms = RunningMeanStd()   # return normalizer

    def build_model(self, obs_dim: int, state_dim: int, action_dim: int) -> None:
        self.model = build_actor_critic(obs_dim, state_dim, action_dim, self.config).to(self.device)
        lr_actor  = self.hyp.get("lr_actor", 3e-4)
        lr_critic = self.hyp.get("lr_critic", 1e-3)
        self.optimizer = torch.optim.Adam([
            {"params": self.model.actor_parameters(),  "lr": lr_actor},
            {"params": self.model.critic_parameters(), "lr": lr_critic},
        ])

    def update(self, buffer: RolloutBuffer, global_step: int = 0, total_steps: int = 1) -> dict:
        """Run n_epochs of PPO updates over the buffer. Returns loss metrics."""
        metrics = defaultdict(list)

        # Normalize returns in-place using running statistics (PopArt-style, Yu et al. MAPPO)
        # Must update buffer.returns BEFORE calling get_minibatches so indices stay aligned.
        returns_np = buffer.returns.cpu().numpy()
        self.ret_rms.update(returns_np)
        buffer.returns = (buffer.returns - self.ret_rms.mean) / self.ret_rms.std

        # Entropy coefficient schedule: linear decay from entropy_coef → entropy_coef_end
        # over entropy_decay_steps, then held flat. Reaches floor much faster than total_steps.
        entropy_coef_end = self.hyp.get("entropy_coef_end", self.entropy_coef)
        decay_steps = self.hyp.get("entropy_decay_steps", total_steps)
        t = min(float(global_step) / float(decay_steps), 1.0)
        current_entropy_coef = self.entropy_coef + (entropy_coef_end - self.entropy_coef) * t

        for _ in range(self.n_epochs):
            for obs, states, actions, old_log_probs, returns_mb, advantages in \
                    buffer.get_minibatches(self.n_minibatches):

                mb   = obs.shape[0]
                n_ag = obs.shape[1]

                # Flatten agents into batch dim for shared actor
                obs_flat     = obs.view(mb * n_ag, -1)
                actions_flat = actions.view(mb * n_ag, -1)

                # Expand state, returns, advantages per agent
                states_exp  = states.unsqueeze(1).expand(-1, n_ag, -1).reshape(mb * n_ag, -1)
                adv_exp     = advantages.unsqueeze(1).expand(-1, n_ag).reshape(mb * n_ag)
                returns_exp = returns_mb.unsqueeze(1).expand(-1, n_ag).reshape(mb * n_ag)
                old_lp_flat = old_log_probs.view(mb * n_ag)

                # Normalize advantages
                adv_exp = (adv_exp - adv_exp.mean()) / (adv_exp.std() + 1e-8)

                # Evaluate actions
                log_probs, values, entropy = self.model.evaluate(obs_flat, states_exp, actions_flat)

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - old_lp_flat)
                surr1 = ratio * adv_exp
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_exp
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss — Huber (delta=10), more robust than MSE for MARL returns
                values = values.squeeze(-1)
                value_loss = nn.functional.huber_loss(values, returns_exp, delta=10.0)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.vf_coef * value_loss + current_entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics["actor_loss"].append(actor_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(-entropy_loss.item())
                metrics["total_loss"].append(loss.item())

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                    metrics["clip_fraction"].append(clip_frac)

        return {k: float(np.mean(v)) for k, v in metrics.items()}


# =====================================================================
# Training loop
# =====================================================================

def train(config: dict, resume_from: str = None) -> None:
    if not _ENV_AVAILABLE:
        raise ImportError("gym-pybullet-drones not installed. Run scripts/install_deps.bat")

    device_str = config.get("training", {}).get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Build environment
    env_raw = SwarmCoverageEnv(config)
    env = EPyMARLWrapper(env_raw)
    env.reset()

    obs_dim    = env.get_obs_size()
    state_dim  = env.get_state_size()
    action_dim = env.get_total_actions()
    n_agents   = env.n_agents

    print(f"Env: {n_agents} agents | obs={obs_dim} | state={state_dim} | actions={action_dim}")

    trainer = MAPPOTrainer(config, device)
    trainer.build_model(obs_dim, state_dim, action_dim)

    n_steps = trainer.n_steps
    buffer = RolloutBuffer(n_steps, n_agents, obs_dim, state_dim, action_dim, device)

    train_cfg   = config.get("training", {})
    total_steps = train_cfg.get("total_timesteps", 5_000_000)
    ckpt_every  = train_cfg.get("checkpoint_every", 100_000)
    eval_every  = train_cfg.get("eval_every", 50_000)
    eval_eps    = train_cfg.get("eval_episodes", 20)
    log_dir     = train_cfg.get("log_dir", "logs/run")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))

    # Curriculum config
    curriculum_cfg = config.get("curriculum", {})
    curriculum_enabled = curriculum_cfg.get("enabled", False)
    curriculum_stages  = curriculum_cfg.get("stages", [])

    # Resume checkpoint
    global_step = 0
    best_coverage = 0.0
    episode = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        global_step = ckpt.get("global_step", 0)
        best_coverage = ckpt.get("best_coverage", 0.0)
        episode = ckpt.get("episode", 0)
        print(f"Resumed from {resume_from} at step {global_step}")

    # Episode tracking
    ep_rewards: list = []
    ep_coverages: list = []
    ep_collisions: list = []
    ep_lengths: list = []
    t_start = time.time()

    env.reset()
    obs_list = env.get_obs()
    state = env.get_state()
    ep_reward = 0.0
    ep_step = 0
    ep_col = 0

    pbar = tqdm(total=total_steps, initial=global_step, unit="steps")

    while global_step < total_steps:
        # --- Collect n_steps of experience ---
        buffer.reset()
        for _ in range(n_steps):
            # Stack per-agent observations
            obs_tensor  = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)  # (N, obs_dim)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)              # (state_dim,)

            with torch.no_grad():
                actions_tensor, log_probs_tensor = trainer.model.get_actions(obs_tensor)
                value = trainer.model.get_value(state_tensor.unsqueeze(0))

            actions_np = actions_tensor.cpu().numpy()
            log_probs_np = log_probs_tensor.cpu().numpy()

            reward, done, info = env.step([actions_np[i] for i in range(n_agents)])

            next_obs_list = env.get_obs()
            next_state    = env.get_state()

            rewards_per_agent = np.full(n_agents, reward, dtype=np.float32)

            buffer.add(
                obs=obs_tensor,
                state=state_tensor,
                actions=actions_tensor,
                log_probs=log_probs_tensor,
                rewards=torch.tensor(rewards_per_agent, device=device),
                value=value,
                done=done,
            )

            obs_list = next_obs_list
            state    = next_state
            ep_reward += reward
            ep_step   += 1
            ep_col    += int(info.get("collision", False))
            global_step += 1
            pbar.update(1)

            if done:
                ep_rewards.append(ep_reward)
                ep_coverages.append(info.get("coverage_pct", 0.0))
                ep_collisions.append(ep_col)
                ep_lengths.append(ep_step)
                episode += 1

                # Log episode metrics
                if len(ep_rewards) > 0:
                    writer.add_scalar("train/episode_reward", ep_reward, global_step)
                    writer.add_scalar("train/coverage_pct", info.get("coverage_pct", 0.0), global_step)
                    writer.add_scalar("train/episode_length", ep_step, global_step)
                    writer.add_scalar("train/collision_count", ep_col, global_step)
                    if curriculum_enabled:
                        writer.add_scalar("train/curriculum_stage", env.get_current_stage(), global_step)

                ep_reward = 0.0
                ep_step   = 0
                ep_col    = 0
                env.reset()
                obs_list = env.get_obs()
                state    = env.get_state()

        # --- PPO update ---
        with torch.no_grad():
            last_state  = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            last_value  = trainer.model.get_value(last_state)

        buffer.compute_gae(last_value, trainer.gamma, trainer.gae_lambda)
        loss_metrics = trainer.update(buffer, global_step=global_step, total_steps=total_steps)

        # Log training losses
        for k, v in loss_metrics.items():
            writer.add_scalar(f"train/{k}", v, global_step)
        entropy_coef_end = trainer.hyp.get("entropy_coef_end", trainer.entropy_coef)
        decay_steps = trainer.hyp.get("entropy_decay_steps", total_steps)
        t = min(float(global_step) / float(decay_steps), 1.0)
        current_ec = trainer.entropy_coef + (entropy_coef_end - trainer.entropy_coef) * t
        writer.add_scalar("train/entropy_coef", current_ec, global_step)

        # --- Periodic evaluation ---
        if global_step % eval_every < n_steps:
            eval_metrics = evaluate(trainer.model, env, eval_eps, device, n_agents)
            # Refresh env state — evaluate() resets the env multiple times,
            # leaving obs_list/state stale and causing value_loss spikes
            env.reset()
            obs_list = env.get_obs()
            state    = env.get_state()
            ep_reward = 0.0
            ep_step   = 0
            ep_col    = 0
            for k, v in eval_metrics.items():
                writer.add_scalar(f"eval/{k}", v, global_step)

            mean_cov = eval_metrics.get("coverage_pct_mean", 0.0)
            print(f"\n[Step {global_step:,}] eval coverage={mean_cov:.3f} "
                  f"collisions={eval_metrics.get('collision_rate_mean', 0):.2f} "
                  f"stage={env.get_current_stage()}")

            # Auto-advance curriculum
            if curriculum_enabled:
                stage_idx = env.get_current_stage() - 1
                if stage_idx < len(curriculum_stages):
                    stage_cfg = curriculum_stages[stage_idx]
                    threshold = stage_cfg.get("advance_threshold")
                    min_eps   = stage_cfg.get("advance_min_episodes", 20)
                    if (threshold is not None
                            and mean_cov >= threshold
                            and len(ep_coverages) >= min_eps
                            and np.mean(ep_coverages[-min_eps:]) >= threshold):
                        advanced = env.advance_curriculum()
                        if advanced:
                            print(f"  >>> Curriculum advanced to stage {env.get_current_stage()}")
                            new_obs_dim   = env.get_obs_size()
                            new_state_dim = env.get_state_size()
                            new_n_agents  = env.n_agents
                            if new_obs_dim != obs_dim or new_state_dim != state_dim or new_n_agents != n_agents:
                                # Preserve actor weights when only state_dim changes (grid resize).
                                # Actor obs_dim never changes across stages — only critic input does.
                                # Rebuilding the full model would wipe all learned actor behavior.
                                # Preserve actor whenever obs_dim is unchanged — even if n_agents
                                # changes (3→6), the per-agent obs semantics are the same (2 nearest
                                # neighbors regardless of swarm size). Only wipe actor if obs_dim changes.
                                preserve_actor = (new_obs_dim == obs_dim)
                                old_actor_state = trainer.model.actor.state_dict() if preserve_actor else None

                                obs_dim   = new_obs_dim
                                state_dim = new_state_dim
                                n_agents  = new_n_agents
                                trainer.build_model(obs_dim, state_dim, action_dim)
                                buffer = RolloutBuffer(n_steps, n_agents, obs_dim, state_dim, action_dim, device)

                                if old_actor_state is not None:
                                    trainer.model.actor.load_state_dict(old_actor_state)
                                    print(f"  >>> Actor weights preserved (only critic rebuilt for new state_dim={state_dim})")
                            env.reset()
                            obs_list = env.get_obs()
                            state    = env.get_state()

            # Save best checkpoint
            if mean_cov > best_coverage:
                best_coverage = mean_cov
                _save_checkpoint(trainer, config, global_step, episode, best_coverage,
                                 os.path.join(log_dir, "checkpoints", "ckpt_best.pt"))

        # --- Periodic checkpoint ---
        if global_step % ckpt_every < n_steps:
            _save_checkpoint(trainer, config, global_step, episode, best_coverage,
                             os.path.join(log_dir, "checkpoints", "ckpt_latest.pt"))

        # --- Console progress ---
        if len(ep_rewards) >= 10:
            fps = global_step / (time.time() - t_start)
            mean_r   = float(np.mean(ep_rewards[-10:]))
            mean_cov = float(np.mean(ep_coverages[-10:]))
            pbar.set_postfix(
                reward=f"{mean_r:.2f}",
                coverage=f"{mean_cov:.2%}",
                stage=env.get_current_stage(),
                fps=f"{fps:.0f}",
            )

    pbar.close()
    writer.close()
    env.close()

    # Final checkpoint
    _save_checkpoint(trainer, config, global_step, episode, best_coverage,
                     os.path.join(log_dir, "checkpoints", "ckpt_final.pt"))
    print(f"\nTraining complete. Best coverage: {best_coverage:.3f}")
    print(f"Checkpoints saved to: {log_dir}/checkpoints/")


# =====================================================================
# Evaluation
# =====================================================================

def evaluate(model, env, n_episodes: int, device: torch.device, n_agents: int) -> dict:
    """Run deterministic evaluation episodes. Returns aggregate metrics."""
    coverages, collisions, lengths, rewards = [], [], [], []
    model.eval()

    for _ in range(n_episodes):
        env.reset()
        obs_list = env.get_obs()
        ep_reward = 0.0
        ep_col = 0
        ep_len = 0

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

        coverages.append(info.get("coverage_pct", 0.0))
        collisions.append(ep_col)
        lengths.append(ep_len)
        rewards.append(ep_reward)

    model.train()
    return {
        "coverage_pct_mean":    float(np.mean(coverages)),
        "coverage_pct_std":     float(np.std(coverages)),
        "collision_rate_mean":  float(np.mean(collisions)),
        "episode_length_mean":  float(np.mean(lengths)),
        "episode_reward_mean":  float(np.mean(rewards)),
    }


def _save_checkpoint(trainer, config, global_step, episode, best_coverage, path):
    torch.save({
        "model_state":     trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "config":          config,
        "global_step":     global_step,
        "episode":         episode,
        "best_coverage":   best_coverage,
    }, path)


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARL Drone Swarm Training")
    parser.add_argument("--config",  type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume",  type=str, default=None,  help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    resume = args.resume or config.get("training", {}).get("resume_from")

    train(config, resume_from=resume)
