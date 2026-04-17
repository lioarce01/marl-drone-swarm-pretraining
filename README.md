# swarm-ml-rl
Multi-agent reinforcement learning (MARL) model for cooperative quadcopter swarm control in simulation. The goal is a transferable actor policy that can be deployed on real drones.

## Task

**Area Coverage** — a swarm of N drones must collectively cover a 2D grid as fast as possible, avoiding collisions and revisiting as few cells as possible.

## Stack

| Component | Choice |
|-----------|--------|
| Simulator | gym-pybullet-drones (PyBullet, cross-platform) |
| Algorithm | MAPPO (centralized critic, shared actor, on-policy) |
| Framework | Custom training loop (PyTorch) |
| Obs/Act | 21-dim per-agent vector / continuous 3D velocity setpoints |

## Project Structure

```
config/          YAML configs (hyperparameters, curriculum, reward weights)
src/
  envs/          PettingZoo ParallelEnv wrapper + coverage map + domain randomization
  networks/      MLP actor-critic (Phase 2), GNN (Phase 5)
  rewards/       Coverage reward function
  train.py       Training entry point
  evaluate.py    Deterministic evaluation + metrics
  visualize.py   Top-down coverage replay animation
scripts/         Windows setup and launch scripts
tests/           Env smoke tests and reward unit tests
logs/            Checkpoints + TensorBoard runs
TESTED.md        Experiment log — what has been tried and must not be repeated
```

## Phases

1. **Env** — PettingZoo wrapper over gym-pybullet-drones
2. **Baseline** — MAPPO + MLP, 3 drones, 10×10 grid
3. **Curriculum** — 5 stages: 10×10 → 20×20, 3 → 6 drones *(completed in v11)*
4. **Domain Randomization** — GPS noise, wind, mass, motor noise *(pending)*
5. **GNN Upgrade** — replace MLP with Graph Attention Network *(pending)*
6. **Eval** — 100-episode deterministic report, final model artifact

## Training

```bash
python src/train.py --config config/mappo_mlp.yaml
```

Resume from checkpoint:
```bash
python src/train.py --config config/mappo_mlp.yaml --resume logs/mappo_mlp_v11/best.pt
```

Evaluate:
```bash
python src/evaluate.py --checkpoint logs/mappo_mlp_v11/best.pt
```

See `TESTED.md` for full experiment history and known failure modes.
