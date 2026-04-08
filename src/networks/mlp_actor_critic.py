"""
mlp_actor_critic.py — MLP-based shared actor-critic for MAPPO.

Architecture:
  Actor:  obs → [hidden] → action_dim  (tanh output, continuous)
  Critic: global_state → [hidden] → 1  (centralized value function)

Both networks use LayerNorm for MARL stability.
Parameters are shared across all agents (implicit coordination).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


def _build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    use_layer_norm: bool = True,
    output_activation: nn.Module = None,
) -> nn.Sequential:
    """Build a fully-connected network with optional LayerNorm."""
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class MLPActor(nn.Module):
    """
    Decentralized actor — processes single agent observation.
    Output: mean of Gaussian policy (tanh-squashed to [-1, 1]).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], use_layer_norm: bool = True):
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dims, action_dim, use_layer_norm)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))  # start with std≈0.37, not 1.0

        # Orthogonal initialization (standard for PPO)
        self._init_weights()

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim) or (obs_dim,)

        Returns:
            action_mean: (batch, action_dim) — tanh-squashed to [-1, 1]
            action_log_std: (batch, action_dim) — clamped to [-5, 2]
        """
        mean = torch.tanh(self.net(obs))
        log_std = self.log_std.clamp(-2.0, 0.5).expand_as(mean)  # std in [0.135, 1.65] — prevents both collapse and explosion
        return mean, log_std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action.

        Returns:
            action: (batch, action_dim) clamped to [-1, 1]
            log_prob: (batch,) log probability of the action
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device)

        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()   # reparameterization trick
        action = torch.tanh(raw_action)

        # Log prob with tanh correction (change of variables)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions.

        Args:
            obs:     (batch, obs_dim)
            actions: (batch, action_dim) in [-1, 1]

        Returns:
            log_probs: (batch,)
            entropy:   (batch,)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        # Inverse tanh to get raw action
        actions_clamped = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        raw_actions = torch.atanh(actions_clamped)

        log_probs = dist.log_prob(raw_actions) - torch.log(1.0 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Output layer smaller init
        last_linear = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)


class MLPCritic(nn.Module):
    """
    Centralized critic — processes global state (all agents' obs + coverage grid).
    Returns a scalar value estimate.
    """

    def __init__(self, state_dim: int, hidden_dims: List[int], use_layer_norm: bool = True):
        super().__init__()
        self.net = _build_mlp(state_dim, hidden_dims, 1, use_layer_norm)
        self._init_weights()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) global state

        Returns:
            value: (batch, 1) value estimate
        """
        return self.net(state)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        last_linear = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=1.0)


class MLPActorCritic(nn.Module):
    """
    Combined actor-critic module. Shared actor across all agents.
    Provides a clean interface for the MAPPO trainer.
    """

    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, config: dict):
        super().__init__()
        net_cfg = config.get("network", {})
        actor_hidden = net_cfg.get("actor_hidden", [128, 128])
        critic_hidden = net_cfg.get("critic_hidden", [256, 256])
        use_ln = net_cfg.get("use_layer_norm", True)

        self.actor = MLPActor(obs_dim, action_dim, actor_hidden, use_ln)
        self.critic = MLPCritic(state_dim, critic_hidden, use_ln)

    def get_actions(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor.get_action(obs, deterministic)

    def evaluate(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            log_probs: (batch,)
            values:    (batch, 1)
            entropy:   (batch,)
        """
        log_probs, entropy = self.actor.evaluate_actions(obs, actions)
        values = self.critic(state)
        return log_probs, values, entropy

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)

    def actor_parameters(self):
        return self.actor.parameters()

    def critic_parameters(self):
        return self.critic.parameters()
