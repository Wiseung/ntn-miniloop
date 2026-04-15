# NTN Joint Scheduling Closed Loop

This repo now contains a minimal but complete NTN control loop:

`scenario -> link state -> access decision -> beam selection -> power split -> bandwidth split -> queue / throughput / delay metrics`

The control stack is no longer limited to handover heuristics. It includes:

- `proportional_fair` scheduling with an EMA throughput state.
- `lyapunov` scheduling driven by queue backlog.
- same-satellite inter-beam interference from an explicit beam pattern.
- `joint_*` policies that choose satellite-user pairs with an ILP-based finite-horizon MPC.
- minimum-service filtering and an unserved-user penalty inside the joint controller.

## Files

- `configs/default.json`: default experiment setup.
- `configs/synthetic_multiplane_realistic.json`: synthetic multi-plane scenario with more realistic user geography.
- `configs/tle_demo.json`: TLE-driven demo setup.
- `configs/tle_starlink_large.json`: larger-scale Starlink TLE scenario.
- `configs/tle_oneweb_large.json`: larger-scale OneWeb TLE scenario.
- `configs/tle_starlink_oneweb_mixed.json`: mixed-constellation TLE scenario using multiple source files.
- `data/tle/sample_demo.tle`: small bundled TLE example.
- `data/tle/snapshots/*.tle`: downloaded real snapshots.
- `data/tle/snapshots/*_latest.tle`: auto-refreshed latest aliases.
- `data/tle/snapshots/latest_snapshot.json`: manifest describing the current latest snapshot tag.
- `tools/fetch_tle_snapshots.py`: refresh Starlink/OneWeb snapshots from official CelesTrak endpoints.
- `ntn_miniloop/core.py`: simulation, schedulers, interference model, reporting.
- `ntn_miniloop/env.py`: Gym-style RL environment wrapper.
- `ntn_miniloop/spaces.py`: minimal space classes for action/observation specs.
- `run_closed_loop.py`: CLI entry point.

## Install

Base install:

```bash
pip install -e .
```

Install with RL extras:

```bash
pip install -e .[rl]
```

## Constellation Modes

- `synthetic`: multi-plane circular constellation with inclination, Earth rotation, WGS84 ground geometry, and optional J2 secular drift.
- `tle`: satellite positions propagated from real TLE snapshots using local `sgp4`, including multi-file mixed-constellation selection.

## User Mobility

- `user_mobility_mode="static"` keeps terminals fixed.
- `user_mobility_mode="linear"` moves each user with per-user speed and heading.
- `user_mobility_mode="waypoint"` follows explicit waypoint lists.
- `user_mobility_mode="road"` synthesizes a piecewise-turn ground route from speed, heading, and turn pattern.
- `user_mobility_mode="flight_corridor"` synthesizes a smoother corridor-style airborne route.
- `user_altitudes_m` and `user_target_altitudes_m` add altitude profiles, so ground and airborne users can coexist in one scenario.
- airborne trajectories can use dynamic propagation with acceleration, bank-angle, turn-rate, climb-rate, and descent-rate constraints.
- airborne runs now emit path-following diagnostics such as lateral path error and turn-anticipation distance.
- The same mobility model is used for both synthetic and TLE constellation scenes.

## Mixed-Constellation Stats

- TLE runs now emit `source_label` per satellite into the timeseries output.
- `summary_<policy>.json` contains `source_stats` for each source label.
- `report.md` includes a `Source Breakdown` section for mixed TLE scenes.
- mixed TLE outputs also include `Source x Platform Breakdown` and source/platform timeseries figures.

## Policies

- `sticky_equal`
- `sticky_proportional_fair`
- `sticky_lyapunov`
- `joint_proportional_fair`
- `joint_lyapunov`

`sticky_*` keeps access and resource allocation separated.

`joint_*` couples access and resource control in one policy.

## Run

```bash
python run_closed_loop.py
```

Run the bundled TLE demo:

```bash
python run_closed_loop.py --config configs/tle_demo.json --policies sticky_equal joint_proportional_fair
```

Run the larger Starlink TLE scenario:

```bash
python run_closed_loop.py --config configs/tle_starlink_large.json --policies sticky_equal joint_proportional_fair --output-dir outputs_starlink_large
```

Run the mixed Starlink + OneWeb TLE scenario:

```bash
python run_closed_loop.py --config configs/tle_starlink_oneweb_mixed.json --policies sticky_equal joint_proportional_fair --output-dir outputs_mixed_tle
```

Refresh official snapshots:

```bash
python tools/fetch_tle_snapshots.py --tag 2026-04-11
```

Print the currently active latest tag:

```bash
python tools/fetch_tle_snapshots.py --print-latest
```

Reuse the tag already recorded in `latest_snapshot.json`:

```bash
python tools/fetch_tle_snapshots.py --use-latest-tag
```

After this runs, the script automatically refreshes:

- `data/tle/snapshots/starlink_latest.tle`
- `data/tle/snapshots/oneweb_latest.tle`
- `data/tle/snapshots/starlink_oneweb_latest.tle`

The large TLE configs already point to these `latest` aliases, so no manual JSON edits are needed after each snapshot refresh.

GitHub Actions also includes a scheduled workflow that refreshes these latest aliases and opens a PR automatically.
If your repository does not allow the default `GITHUB_TOKEN` to create PRs, add a `CODEX_PR_TOKEN` Actions secret and the workflow will use it automatically.

Run a subset:

```bash
python run_closed_loop.py --policies joint_proportional_fair joint_lyapunov
```

Use a custom config:

```bash
python run_closed_loop.py --config configs/default.json --output-dir outputs
```

## Outputs

- `outputs/report.md`
- `outputs/policy_comparison.png`
- `outputs/source_breakdown.png` for mixed TLE scenes
- `outputs/source_platform_breakdown.png` and `outputs/source_platform_timeseries.png` for mixed TLE scenes
- `outputs/source_platform_service_timeseries.png` for mixed TLE throughput / SINR / beam-grant linkage
- `outputs/user0_trace.png`
- `outputs/summary_<policy>.json`
- `outputs/timeseries_<policy>.csv`

## RL Environment

The package now exposes a Gym-style environment without requiring `gymnasium`:

```python
from ntn_miniloop import NTNClosedLoopEnv, SimulationConfig

env = NTNClosedLoopEnv(SimulationConfig())
obs, info = env.reset(seed=123)
action = env.action_from_policy("joint_lyapunov")
obs, reward, terminated, truncated, info = env.step(action)
```

Main methods:

- `reset(seed=None, options=None)`
- `step(action)`
- `action_from_policy(policy_name)` for heuristic rollout baselines
- `get_episode_frame()`
- `episode_summary()`

Action structure:

- `satellite_choices`
- `beam_priority`
- `bandwidth_weight`
- `power_weight`

## Gymnasium Wrapper

For direct SB3 / CleanRL integration, use the Gymnasium-compatible wrapper:

```python
from ntn_miniloop import GymnasiumNTNEnv, SimulationConfig

env = GymnasiumNTNEnv(
    SimulationConfig(),
    reward_template="throughput_first",
)
obs, info = env.reset(seed=123)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

Available reward templates:

- `throughput_first`
- `delay_first`
- `handover_penalty`

You can also override them with a custom `RewardWeights`.

## Flattened Observation Wrapper

For the simplest MLP baselines, flatten the dict observation into a single vector:

```python
from ntn_miniloop import GymnasiumNTNEnv, SingleAgentFlattenObservationWrapper, SimulationConfig

env = SingleAgentFlattenObservationWrapper(
    GymnasiumNTNEnv(SimulationConfig())
)
obs, info = env.reset(seed=123)
print(obs.shape)
```

## Flattened Action Wrapper

For the simplest SB3 MLP policies, flatten the dict action into a single continuous vector:

```python
from ntn_miniloop import (
    GymnasiumNTNEnv,
    SingleAgentFlattenActionWrapper,
    SingleAgentFlattenObservationWrapper,
    SimulationConfig,
)

env = SingleAgentFlattenActionWrapper(
    SingleAgentFlattenObservationWrapper(
        GymnasiumNTNEnv(SimulationConfig())
    )
)
obs, info = env.reset(seed=123)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

The wrapper maps:

- the first block of the flat action to quantized `satellite_choices`
- the remaining blocks to `beam_priority`, `bandwidth_weight`, and `power_weight`

## Vector Environment Factories

Gymnasium vector env:

```python
from ntn_miniloop import SimulationConfig, make_gymnasium_vector_env

vec_env = make_gymnasium_vector_env(
    SimulationConfig(),
    num_envs=4,
    flatten_observation=True,
    flatten_action=True,
    asynchronous=False,
)
obs, info = vec_env.reset(seed=123)
```

For manual vector rollouts with dict actions, stack per-env actions first:

```python
from ntn_miniloop import stack_dict_actions

actions = [vec_env.single_action_space.sample() for _ in range(vec_env.num_envs)]
batched_actions = stack_dict_actions(actions)
obs, rewards, terminations, truncations, infos = vec_env.step(batched_actions)
```

SB3-ready env factory:

```python
from ntn_miniloop import SimulationConfig, make_sb3_vec_env

vec_env = make_sb3_vec_env(
    SimulationConfig(),
    num_envs=4,
    flatten_observation=True,
    flatten_action=True,
    use_subprocess=True,
    monitor=True,
    log_dir="logs/sb3_run",
)
```

If SB3 is not installed, `make_sb3_vec_env()` raises a clear error and you can still use `make_vec_env_fns()` or `make_gymnasium_vector_env()`.

## Key Metrics

- `delivery_ratio`
- `p95_queue_delay_s`
- `beam_block_ratio`
- `handover_block_ratio`
- `jain_fairness`
- `mean_sinr_loss_db`

## Extension Points

- Replace the current lightweight MPC forecast with explicit backlog-state coupling or terminal-cost shaping.
- Replace the simple beam pattern with measured antenna patterns or beam footprints.
- Add multi-plane constellations or imported TLE data.
- Add learning-based schedulers on top of the same closed loop.
