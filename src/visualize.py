"""
visualize.py — Top-down coverage replay animation and static snapshots.

Usage:
    python src/visualize.py --checkpoint logs/mappo_mlp/checkpoints/ckpt_best.pt
    python src/visualize.py --checkpoint logs/mappo_mlp/checkpoints/ckpt_best.pt --episodes 3 --save-mp4

Outputs:
    - Per-episode PNG snapshots:  logs/<run>/viz/ep_XXX_step_YYY.png
    - MP4 animation (optional):   logs/<run>/viz/episode_XXX.mp4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import List

from src.train import load_config
from src.networks.gnn_actor_critic import build_actor_critic

try:
    from src.envs.swarm_coverage_env import SwarmCoverageEnv
    from src.envs.epymarl_wrapper import EPyMARLWrapper
    _ENV_AVAILABLE = True
except ImportError:
    _ENV_AVAILABLE = False


# Drone colours — distinct per agent
_DRONE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

# Coverage colormap: white (unexplored) → teal (explored)
_COV_CMAP = LinearSegmentedColormap.from_list("cov", ["#f8f9fa", "#00b4d8"], N=256)


def render_frame(
    coverage_grid: np.ndarray,
    drone_positions: List[np.ndarray],
    cell_size: float,
    step: int,
    coverage_pct: float,
    ax: plt.Axes,
) -> None:
    """Render one frame onto a matplotlib Axes."""
    ax.clear()
    grid_size = coverage_grid.shape[0]
    arena = grid_size * cell_size

    # Coverage heatmap
    ax.imshow(
        coverage_grid.T,           # transpose: x→col, y→row
        origin="lower",
        extent=[0, arena, 0, arena],
        cmap=_COV_CMAP,
        vmin=0, vmax=1,
        alpha=0.7,
        interpolation="nearest",
    )

    # Grid lines
    for i in range(grid_size + 1):
        v = i * cell_size
        ax.axhline(v, color="#dee2e6", linewidth=0.3, zorder=1)
        ax.axvline(v, color="#dee2e6", linewidth=0.3, zorder=1)

    # Drone positions
    for idx, pos in enumerate(drone_positions):
        color = _DRONE_COLORS[idx % len(_DRONE_COLORS)]
        # Sensor radius circle
        circle = plt.Circle(
            (pos[0], pos[1]), radius=1.5,
            color=color, fill=True, alpha=0.15, zorder=2
        )
        ax.add_patch(circle)
        # Drone marker
        ax.scatter(pos[0], pos[1], c=color, s=80, zorder=3, marker="^",
                   edgecolors="white", linewidths=0.8)
        ax.text(pos[0] + 0.2, pos[1] + 0.2, f"D{idx}", fontsize=6,
                color=color, zorder=4, fontweight="bold")

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=_DRONE_COLORS[i], label=f"Drone {i}")
        for i in range(len(drone_positions))
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=6, framealpha=0.8)

    ax.set_xlim(0, arena)
    ax.set_ylim(0, arena)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_title(f"Step {step:4d} | Coverage {coverage_pct:.1%}", fontsize=9)


def run_episode_visualize(
    model,
    env_raw: "SwarmCoverageEnv",
    env: "EPyMARLWrapper",
    device: torch.device,
    episode_idx: int,
    save_dir: str,
    save_mp4: bool = False,
    snapshot_every: int = 50,
) -> List[np.ndarray]:
    """Run one episode, saving frames. Returns list of frame arrays."""
    env.reset()
    obs_list = env.get_obs()
    n_agents = env.n_agents
    frames = []

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    done = False
    step = 0
    while not done:
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
        with torch.no_grad():
            actions_tensor, _ = model.get_actions(obs_tensor, deterministic=True)
        actions_np = actions_tensor.cpu().numpy()

        reward, done, info = env.step([actions_np[i] for i in range(n_agents)])
        obs_list = env.get_obs()
        step += 1

        # Snapshot every N steps or at episode end
        if step % snapshot_every == 0 or done:
            positions = env_raw._true_positions
            grid = env_raw._coverage_map.get_grid()
            cov  = info.get("coverage_pct", 0.0)

            render_frame(grid, [positions[i] for i in range(n_agents)],
                         env_raw.cell_size, step, cov, ax)

            # Save PNG
            png_path = os.path.join(save_dir, f"ep{episode_idx:03d}_step{step:04d}.png")
            fig.tight_layout()
            fig.savefig(png_path, dpi=100, bbox_inches="tight")

            # Capture frame array for MP4
            if save_mp4:
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                frames.append(buf.reshape(h, w, 3))

    plt.close(fig)
    return frames


def save_mp4(frames: List[np.ndarray], path: str, fps: int = 10) -> None:
    """Save list of RGB frame arrays as MP4 using imageio."""
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, codec="libx264")
        print(f"  Saved MP4: {path}")
    except ImportError:
        print("  imageio not installed — skipping MP4. Run: pip install imageio imageio-ffmpeg")


def visualize(checkpoint_path: str, n_episodes: int = 3, save_mp4_flag: bool = False) -> None:
    if not _ENV_AVAILABLE:
        raise ImportError("gym-pybullet-drones not installed.")

    ckpt   = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_raw = SwarmCoverageEnv(config)
    env     = EPyMARLWrapper(env_raw)
    env.reset()

    obs_dim    = env.get_obs_size()
    state_dim  = env.get_state_size()
    action_dim = env.get_total_actions()

    model = build_actor_critic(obs_dim, state_dim, action_dim, config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    save_dir = os.path.join(os.path.dirname(checkpoint_path), "..", "viz")
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(n_episodes):
        print(f"Rendering episode {ep + 1}/{n_episodes}...")
        frames = run_episode_visualize(
            model, env_raw, env, device,
            episode_idx=ep,
            save_dir=save_dir,
            save_mp4=save_mp4_flag,
        )
        if save_mp4_flag and frames:
            mp4_path = os.path.join(save_dir, f"episode_{ep:03d}.mp4")
            save_mp4(frames, mp4_path)

    env.close()
    print(f"\nFrames saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained MARL policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes",   type=int, default=3)
    parser.add_argument("--save-mp4",   action="store_true", default=False)
    parser.add_argument("--snap-every", type=int, default=50,
                        help="Save a PNG snapshot every N steps")
    args = parser.parse_args()

    visualize(args.checkpoint, args.episodes, args.save_mp4)
