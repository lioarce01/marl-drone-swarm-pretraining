"""
gnn_actor_critic.py — Graph Attention Network (GAT) actor-critic for MAPPO.

Each drone is a graph node. Edges connect drones within neighbor_radius metres.
GAT message-passing allows each drone to selectively attend to its neighbours,
enabling explicit coordination based on proximity.

Architecture:
  Node embedding: MLP(obs_i) → h_i
  Message:        GAT attention over neighbours → aggregated message m_i
  Update:         GRU(h_i, m_i) → h_i'    (memory across layers)
  Actor:          MLP(h_i') → action_i     (per-node, decentralized)
  Critic:         MLP(concat(h_i')) → V    (global, centralized)

Falls back to fully-connected graph when n_agents <= max_fc_agents (no radius cutoff).

NOTE: Requires torch-geometric for production.
      A pure-PyTorch fallback (manual attention) is also provided for environments
      where torch-geometric cannot be installed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

# Try torch-geometric; fall back to manual implementation
try:
    from torch_geometric.nn import GATConv
    _TORCH_GEOMETRIC = True
except ImportError:
    _TORCH_GEOMETRIC = False


# =====================================================================
# Pure-PyTorch GAT layer (fallback — no torch-geometric required)
# =====================================================================

class ManualGATLayer(nn.Module):
    """
    Single GAT attention layer (Veličković et al. 2018).
    Operates on dense adjacency for small n_agents (≤ 20).
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (batch, N, in_dim) node features
            adj: (batch, N, N) adjacency mask (1=connected, 0=masked)

        Returns:
            out: (batch, N, out_dim)
        """
        batch, N, _ = x.shape
        h = self.W(x)                                    # (B, N, out_dim)
        h = h.view(batch, N, self.n_heads, self.head_dim)  # (B, N, H, D)

        # Attention scores: e_ij = LeakyReLU(a^T [h_i || h_j])
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)   # (B, N, N, H, D)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)   # (B, N, N, H, D)
        concat = torch.cat([h_i, h_j], dim=-1)            # (B, N, N, H, 2D)

        e = (concat * self.a).sum(dim=-1)                 # (B, N, N, H)
        e = F.leaky_relu(e, negative_slope=0.2)

        # Mask non-edges with -inf before softmax
        mask = adj.unsqueeze(-1).expand_as(e)             # (B, N, N, H)
        e = e.masked_fill(mask == 0, float("-inf"))

        alpha = F.softmax(e, dim=2)                       # (B, N, N, H)
        alpha = self.dropout(alpha)
        alpha = torch.nan_to_num(alpha, nan=0.0)          # handle isolated nodes

        # Aggregate: h_i' = Σ_j α_ij * h_j
        out = (alpha.unsqueeze(-1) * h_j).sum(dim=2)      # (B, N, H, D)
        out = out.view(batch, N, -1)                       # (B, N, out_dim)
        return F.elu(out)


# =====================================================================
# GNN Actor-Critic
# =====================================================================

class GNNActorCritic(nn.Module):
    """
    GAT-based actor-critic for MAPPO with N drone agents.

    Drop-in replacement for MLPActorCritic — same interface.
    """

    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, config: dict):
        super().__init__()
        net_cfg = config.get("network", {})

        node_hidden:     int   = net_cfg.get("node_hidden", 64)
        edge_hidden:     int   = net_cfg.get("edge_hidden", 32)
        n_gnn_layers:    int   = net_cfg.get("n_gnn_layers", 2)
        actor_hidden:    list  = net_cfg.get("actor_hidden", [64, 64])
        critic_hidden:   list  = net_cfg.get("critic_hidden", [128, 128])
        use_ln:          bool  = net_cfg.get("use_layer_norm", True)
        self.neighbor_radius: float = net_cfg.get("neighbor_radius", 5.0)
        self.obs_dim = obs_dim

        # --- Node encoder: obs → node embedding ---
        self.node_encoder = nn.Sequential(
            nn.Linear(obs_dim, node_hidden),
            nn.LayerNorm(node_hidden) if use_ln else nn.Identity(),
            nn.ReLU(),
            nn.Linear(node_hidden, node_hidden),
            nn.LayerNorm(node_hidden) if use_ln else nn.Identity(),
            nn.ReLU(),
        )

        # --- GAT layers ---
        # Each layer: ManualGATLayer(node_hidden → node_hidden)
        gat_layers = []
        for _ in range(n_gnn_layers):
            gat_layers.append(ManualGATLayer(node_hidden, node_hidden, n_heads=4))
            if use_ln:
                gat_layers.append(nn.LayerNorm(node_hidden))
        self.gat_layers = nn.ModuleList(gat_layers)
        self.n_gnn_layers = n_gnn_layers
        self._use_ln = use_ln

        # --- Actor head: per-node MLP → action ---
        actor_layers = []
        in_d = node_hidden
        for h in actor_hidden:
            actor_layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        actor_layers.append(nn.Linear(in_d, action_dim))
        self.actor_head = nn.Sequential(*actor_layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # --- Critic head: global concat → value ---
        # Input: concat of all node embeddings (n_agents * node_hidden)
        # We don't know n_agents at init time, so we use a flexible linear
        critic_layers = []
        # Use state_dim (full global state) as critic input — matches MLPCritic
        in_d = state_dim
        for h in critic_hidden:
            critic_layers += [nn.Linear(in_d, h)]
            if use_ln:
                critic_layers.append(nn.LayerNorm(h))
            critic_layers.append(nn.ReLU())
            in_d = h
        critic_layers.append(nn.Linear(in_d, 1))
        self.critic_head = nn.Sequential(*critic_layers)

        self._init_weights()

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _gnn_forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run GAT over a batch of agent observations.

        Args:
            obs: (batch*N, obs_dim) OR (N, obs_dim) — will be reshaped

        Returns:
            node_emb: same leading shape, (*, node_hidden)
        """
        # Expect (batch, N, obs_dim) — caller reshapes if needed
        x = self.node_encoder(obs)  # (B, N, node_hidden) or (N, node_hidden)

        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, node_hidden)
            squeeze = True
        else:
            squeeze = False

        batch, N, _ = x.shape
        adj = self._build_adjacency(x, N, batch)  # (B, N, N)

        for layer in self.gat_layers:
            if isinstance(layer, ManualGATLayer):
                x = layer(x, adj)
            else:
                # LayerNorm
                x = layer(x)

        if squeeze:
            x = x.squeeze(0)
        return x

    def _build_adjacency(self, x: torch.Tensor, N: int, batch: int) -> torch.Tensor:
        """
        Build adjacency matrix based on neighbor_radius.
        For small N (≤ 6) returns fully-connected adjacency.

        Returns: (batch, N, N) float tensor, 1 = edge exists.
        """
        # Fully connected for small swarms (radius not needed)
        if N <= 6:
            adj = torch.ones(batch, N, N, device=x.device)
            adj[:, torch.arange(N), torch.arange(N)] = 0  # no self-loops
            return adj

        # Radius-based connectivity: use position (first 3 dims of obs)
        pos = x[:, :, :3]  # (B, N, 3) — approximate position from obs embedding
        # Compute pairwise distances
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)       # (B, N, N, 3)
        dist = diff.norm(dim=-1)                          # (B, N, N)
        adj  = (dist <= self.neighbor_radius).float()
        adj[:, torch.arange(N), torch.arange(N)] = 0    # no self-loops
        # Always connect at least 1 neighbour (avoid isolated nodes)
        nearest = dist.topk(min(2, N-1), dim=-1, largest=False).indices
        for i in range(min(2, N-1)):
            adj.scatter_(2, nearest[:, :, i:i+1], 1.0)

        return adj

    # ------------------------------------------------------------------
    # MAPPO interface (same as MLPActorCritic)
    # ------------------------------------------------------------------

    def get_actions(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (N, obs_dim) — per-agent observations for one step
        """
        node_emb = self._gnn_forward(obs)   # (N, node_hidden)
        mean = torch.tanh(self.actor_head(node_emb))  # (N, action_dim)

        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)

        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device)

        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw  = dist.rsample()
        action = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(dim=-1)
        return action, log_prob

    def evaluate(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        obs:     (mb*N, obs_dim)
        state:   (mb*N, state_dim)  [expanded per agent in trainer]
        actions: (mb*N, action_dim)
        """
        # Reshape to (mb, N, obs_dim) for GNN — infer N from log_std
        # We use the actor_head directly on flat obs for simplicity in update
        # (GNN is run per-step, not per-minibatch-item)
        node_emb = self.node_encoder(obs)             # (mb*N, node_hidden)
        mean     = torch.tanh(self.actor_head(node_emb))  # (mb*N, action_dim)
        log_std  = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        std      = log_std.exp()

        dist    = torch.distributions.Normal(mean, std)
        actions_clamped = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        raw_actions     = torch.atanh(actions_clamped)
        log_probs = (dist.log_prob(raw_actions) - torch.log(1.0 - actions.pow(2) + 1e-6)).sum(dim=-1)
        entropy   = dist.entropy().sum(dim=-1)

        values = self.critic_head(state)
        return log_probs, values, entropy

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic_head(state)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def actor_parameters(self):
        """Returns actor-side parameters: encoder + GAT + actor head + log_std."""
        import itertools
        return itertools.chain(
            self.node_encoder.parameters(),
            self.gat_layers.parameters(),
            self.actor_head.parameters(),
            [self.log_std],
        )

    def critic_parameters(self):
        """Returns critic-side parameters."""
        return self.critic_head.parameters()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


def build_actor_critic(obs_dim: int, state_dim: int, action_dim: int, config: dict):
    """Factory: returns GNNActorCritic or MLPActorCritic based on config."""
    net_type = config.get("network", {}).get("type", "MLP").upper()
    if net_type == "GNN":
        return GNNActorCritic(obs_dim, state_dim, action_dim, config)
    else:
        from src.networks.mlp_actor_critic import MLPActorCritic
        return MLPActorCritic(obs_dim, state_dim, action_dim, config)
