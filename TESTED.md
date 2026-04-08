# What We've Tested — Do Not Repeat

## Hyperparameters

| Setting | Outcome | Verdict |
|---|---|---|
| `entropy_coef=0.01` | Entropy → 8.8, policy collapsed to random | DO NOT USE |
| `entropy_coef=0.05` | Entropy → 9.0 immediately | DO NOT USE |
| `entropy_coef=0.001` | Entropy → -7.5, policy collapsed to deterministic | DO NOT USE |
| `entropy_coef=0.003` | Entropy → -8.1, same collapse | DO NOT USE |
| `entropy_coef=0.005` fixed | Entropy stable early, drifts to 2.25 by 1.3M as coverage reward weakens | DO NOT USE fixed |
| `entropy_coef 0.005→0.0005` over 10M steps | Too slow — only 14% drop at 2M, entropy still drifted | DO NOT USE |
| `entropy_coef 0.005→0.001` over 2M steps | Entropy peaks ~1.8 at 400k then declines to 0.1 by 2.5M — collapses too far | Too aggressive |
| `entropy_coef 0.005→0.002` over 2M steps | Floor insufficient — entropy decays 1.35→0.85 by 3M; floor too weak to hold exploration | DO NOT USE |
| `entropy_coef 0.005→0.003` over 2M steps | Entropy stable at ~0.95–1.0 through stage 5; maintained exploration across all curriculum stages | **CURRENT** |
| `n_minibatches=8` | Major deviation from official MAPPO; unstable training | DO NOT USE |
| `n_minibatches=1` | Official MAPPO default | **CORRECT** |
| `value_loss_coef=0.5` | Wrong for separate actor/critic optimizers | DO NOT USE |
| `value_loss_coef=1.0` | Correct for separate optimizers | **CORRECT** |
| `lr_actor=3e-4, lr_critic=1e-3` | Asymmetric; contributed to instability | DO NOT USE |
| `lr_actor=5e-4, lr_critic=5e-4` | Official MAPPO default | **CORRECT** |

## Reward Weights

| Setting | Outcome | Verdict |
|---|---|---|
| `w_cov=0.5` | Coverage reward ~+0.2/ep, buried by jerk penalty -40/ep | DO NOT USE |
| `w_cov=3.0` | Coverage signal dominant | **CURRENT** |
| `w_prox=0.1` | Eval penalty 27x larger than training (deterministic vs stochastic) → -1100 eval reward | DO NOT USE |
| `w_prox=0.0` | Removed; navigation handled by `nearest_uncovered_direction` in obs | **CORRECT** |
| `w_jerk=0.01` | Dominant negative term (-40/ep), buried coverage signal | DO NOT USE |
| `w_jerk=0.001` | Minimal smoothness penalty | **CURRENT** |
| `w_col=5.0` | Too soft; collisions not strongly penalized | DO NOT USE |
| `w_col=15.0` | Collision strongly penalized | **CURRENT** |

## Network / Architecture

| Setting | Outcome | Verdict |
|---|---|---|
| `log_std` init = `zeros` (std=1.0) | Contributed to entropy explosion | DO NOT USE |
| `log_std` init = `-1.0` (std≈0.37) | Better starting distribution | **CURRENT** |
| `log_std.clamp(-5, 2)` | Too wide; entropy still oscillated | DO NOT USE |
| `log_std.clamp(-2, 0.5)` | Bounds std to [0.135, 1.65]; physically prevents entropy explosion/collapse | **CURRENT** |

## Entropy Schedule — Lessons Learned

The entropy bonus creates persistent pressure toward higher entropy. As coverage reward weakens late in training (fewer uncovered cells), the entropy bonus dominates and policy drifts toward random. Key findings:

- **Fixed entropy_coef**: always drifts or collapses — needs a schedule
- **Schedule too slow (over 10M steps)**: entropy still rises in first 2M steps
- **Schedule too fast (→0.001 by 2M)**: entropy collapses to 0.1 by 2.5M — policy becomes near-deterministic
- **Floor 0.002 insufficient**: entropy continued decaying to 0.85 by 3M even with floor held flat. Stage 3 threshold 0.45 was never met because the policy converged before reaching it.
- **Floor 0.003 working**: entropy holds ~0.95–1.0 through v11; policy successfully advanced through all 5 curriculum stages.
- **Entropy near-zero before curriculum advancement**: policy memorizes a fixed trajectory, cannot adapt to new stage conditions. Coverage temporarily drops 30 points on identical environment after stage transition.
- **Entropy bounce on curriculum transition**: when entropy was ~0.1 and curriculum advanced, entropy spiked back to 1.45 due to new obs patterns (empty coverage map). This is a useful but unreliable recovery mechanism.
- **Recommended floor: 0.003** — 0.002 is too weak; 0.003 maintains sufficient exploration pressure through late stages.

## Curriculum Learning — Lessons Learned

- **Stage 1 threshold 0.80**: unreachable — coverage plateaued at 0.55. DO NOT use thresholds above 0.60 for stage 1.
- **Stage 1 threshold 0.55**: reached at ~2.5M steps in v9 (train 0.55, eval 0.53 → advanced).
- **Stage 1 threshold 0.52**: better target — policy already past this, advances sooner.
- **Stage 3 threshold 0.45**: never met in v10 — train coverage was 0.38–0.40, entropy collapsed before reaching it. Lowered to 0.38.
- **Stage 3 threshold 0.38**: working — policy advanced in v11.
- **Curriculum uses BOTH eval AND train coverage**: both `mean_cov >= threshold` (eval) AND `np.mean(ep_coverages[-20:]) >= threshold` (training) must be met. Eval coverage is usually lower than train coverage — threshold it accordingly.
- **Stage 2 obstacles not implemented**: `n_obstacles` in stage config is silently ignored by `_apply_stage_cfg()`. Stage 2 is currently identical to stage 1 physically. Coverage still dropped ~30 points at stage transition due to entropy collapse state of the policy, not obstacle difficulty.
- **All 5 stages completed**: v11 (resuming v10 best checkpoint) advanced through all stages within 400k additional steps. Stage 5 (20×20, 6 agents) reached with coverage 0.32–0.35 train / 0.41 eval at 2M steps.
- **Actor preservation on grid resize**: when `grid_size` changes (e.g. stage 2→3, 10×10→15×15), `state_dim` changes, triggering `trainer.build_model()`. Must save and restore `actor.state_dict()` whenever `obs_dim` is unchanged — even when `n_agents` changes (per-agent obs_dim is identical regardless of swarm size). Do NOT gate on `n_agents` equality.
- **Working thresholds**: stage 1: 0.52, stage 2: 0.50, stage 3: 0.38, stage 4: 0.40, stage 5: null (final).

## Crash Spiral — Lessons Learned

- **Crash spiral in eval (v6)**: deterministic eval policy locked into downward trajectory. Drones crashed to ground (alt < 0.05m), PyBullet didn't terminate, `-15/step` accumulated silently for 1000+ steps → eval reward -1400. Training unaffected (stochastic policy naturally flew back up).
- **Crash spiral in training (v7 at 700k)**: same pattern eventually hit training — reward -370.
- **Fix**: terminate episode immediately when any drone altitude < 0.15m (`collisions.any()` in done condition). Single-step crash penalty (-15) instead of multi-step spiral.
- **Altitude threshold**: 0.05m was too low (drones could bounce above it). 0.15m catches dangerous low-altitude before ground contact.

## Bugs Fixed — Do Not Reintroduce

- **Return normalization index mismatch**: normalizing returns with a separate `randperm` outside `get_minibatches` → mismatched (state, return) pairs → corrupt value function → value_loss spikes → training crash. Always normalize `buffer.returns` in-place before calling `get_minibatches`.
- **Stale obs after eval**: not resetting env + re-collecting obs after evaluation caused recurring value_loss spikes at each eval checkpoint. Always call `env.reset()` and refresh `obs_list/state` after eval.
- **Missing `"collision"` key in info dict**: `info.get("collision", False)` silently returned False every step — collision penalty never triggered. Must be explicitly set in `epymarl_wrapper.py`.
- **Collision reporting only checked drone_0**: `infos.get(agents[0])` in the wrapper only read the first agent's info. Drone_1/2 crashes accumulated `-15/step` in the reward silently, making eval reward plummet to -1400 while `collision_rate_mean` logged 0. Fixed to `any(infos[a]["collision"] for a in agents)`.
- **No episode termination on crash**: When a drone hit the ground, `raw_term` from PyBullet stayed False so the episode continued with `collision=True` every remaining step. Fixed by adding `or collisions.any()` to done condition.
- **`w_prox` at eval time**: potential-based shaping creates huge negative rewards under deterministic eval. Keep `w_prox=0.0`.
- **`drone_mass_multiplier` never applied**: sampled every episode but never passed to PyBullet. Fixed via `pybullet.changeDynamics` after reset. Only active when `domain_randomization.enabled: true`.
- **DR ranges missing from YAML**: all DR ranges defaulted to `[0.0, 0.0]` — enabling DR would randomize nothing. Ranges now in config under `domain_randomization`.
- **`apply_motor_noise()` dead code**: method operated on RPMs but env uses `ActionType.VEL`. Noise is applied directly on velocity commands in `step()`. Method removed.
- **Actor destroyed on grid_size curriculum transition**: when `grid_size` changes (e.g. stage 2→3, 10×10→15×15), `state_dim` changes (163→288), triggering `trainer.build_model()` which rebuilt the ENTIRE model from random weights — wiping the actor. Fix: save `actor.state_dict()` before rebuild, restore after if `obs_dim` is unchanged (`preserve_actor = new_obs_dim == obs_dim`). Only the critic needs rebuilding for grid resize. Do NOT gate on `n_agents` — per-agent obs_dim is the same regardless of swarm size, actor weights are fully reusable when going 3→6 agents.

## Observation Space

- Final obs_dim = **21** (not 22 or 19 from earlier iterations).
- Added `nearest_uncovered_direction` (2D unit vector) — critical navigation signal. Without it, drones had no compass to unexplored cells.
- Removing any obs component will break the network input shape; always update `_obs_dim()` to match.

## Domain Randomization (Phase 4) — Readiness Status

| Component | Status |
|---|---|
| GPS noise | Ready |
| Wind force | Ready |
| Start position noise | Ready |
| Motor noise (velocity commands) | Ready |
| Mass randomization | Ready (fixed) |
| Obstacle jitter | Not ready — obstacles not implemented in env |

Enable only after: curriculum stage ≥ 3, eval coverage consistently > 0.55, `coverage_pct_std > 0.05` in eval (policy not memorizing one trajectory).

## Training Runs Summary

| Run | Steps | Peak Coverage | Notes |
|---|---|---|---|
| v1–v3 | various | ~0.20 | Obs dim bugs, reward bugs |
| v4 | ~2.5M | ~0.35 | Entropy exploded at 2M |
| v5 | ~5.9M | **0.58** at ~800k | Entropy exploded ~1M, never recovered |
| v6 | 700k (stopped) | 0.34 eval ~450k | Crash spiral at 600k — reward -370 in training |
| v7 | 1.3M (stopped) | 0.42 train | Crash fix worked; entropy drifted 1.65→2.25 |
| v8 | 2M (stopped) | 0.44 plateau | Entropy schedule too slow (→0.0005 over 10M) |
| v9 | 3M (stopped) | **0.55 train / 0.53 eval** | Stages 1→2→3 in 500k steps; actor destroyed at stage 3 (grid resize bug) |
| v10 | 3M (stopped) | 0.44 eval stage 3 | entropy floor 0.002 insufficient — entropy decayed to 0.85 by 3M; stage 3 threshold 0.45 never met; actor preservation fix first confirmed working (coverage held at 0.35, not 0.15) |
| v11 | 2M+ (running) | **0.41 eval stage 5** | Resumed v10 best ckpt; entropy floor raised to 0.003; stage 3 threshold lowered to 0.38; all 5 curriculum stages completed within 400k steps; coverage 0.32–0.35 train / 0.41 eval at 2M on stage 5 (20×20, 6 agents) |
