from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any

VENDOR_PATH = Path(__file__).resolve().parent.parent / "_vendor"
if VENDOR_PATH.exists() and str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))

import numpy as np
import pandas as pd

from .core import (
    SimulationConfig,
    beam_pattern_attenuation_db,
    build_time_grid,
    compute_control_objective,
    compute_link_state,
    compute_satellite_positions,
    compute_user_state,
    dbm_to_mw,
    get_satellite_metadata,
    linear_to_db,
    parse_policy_name,
    select_assignments_for_slot,
)
from .spaces import Box, Dict, MultiDiscrete

REWARD_TEMPLATES: dict[str, dict[str, float]] = {
    "throughput_first": {
        "throughput_mbps": 1.0,
        "queue_delay_s": 0.01,
        "outage_users": 1.0,
        "beam_blocked_users": 0.5,
        "handover_events": 0.25,
    },
    "delay_first": {
        "throughput_mbps": 0.6,
        "queue_delay_s": 0.05,
        "outage_users": 1.5,
        "beam_blocked_users": 0.8,
        "handover_events": 0.3,
    },
    "handover_penalty": {
        "throughput_mbps": 0.9,
        "queue_delay_s": 0.02,
        "outage_users": 1.0,
        "beam_blocked_users": 0.5,
        "handover_events": 1.5,
    },
}


@dataclass(slots=True)
class RewardWeights:
    throughput_mbps: float = 1.0
    queue_delay_s: float = 0.02
    outage_users: float = 1.5
    beam_blocked_users: float = 0.75
    handover_events: float = 0.5

    @classmethod
    def from_template(cls, name: str) -> "RewardWeights":
        if name not in REWARD_TEMPLATES:
            raise ValueError(f"Unknown reward template '{name}'. Available: {', '.join(sorted(REWARD_TEMPLATES))}")
        return cls(**REWARD_TEMPLATES[name])

    def to_dict(self) -> dict[str, float]:
        return {
            "throughput_mbps": float(self.throughput_mbps),
            "queue_delay_s": float(self.queue_delay_s),
            "outage_users": float(self.outage_users),
            "beam_blocked_users": float(self.beam_blocked_users),
            "handover_events": float(self.handover_events),
        }


class NTNClosedLoopEnv:
    """Gym-style environment for the NTN closed loop.

    Action semantics:
    - `satellite_choices[i]` picks the requested satellite for user i, with `num_satellites` as "no assignment".
    - `beam_priority[i]` ranks users competing for beams on the same satellite.
    - `bandwidth_weight[i]` and `power_weight[i]` control the split among granted users on a satellite.
    """

    metadata = {"render_modes": ["human"], "name": "NTNClosedLoopEnv"}

    def __init__(
        self,
        config: SimulationConfig,
        reward_weights: RewardWeights | None = None,
        reward_template: str = "throughput_first",
    ) -> None:
        self.base_config = config
        self.reward_template = reward_template
        self.reward_weights = reward_weights or RewardWeights.from_template(reward_template)
        self.np_random = np.random.default_rng(config.random_seed)
        self._setup_static_context(seed=config.random_seed)
        self._initialize_episode_state()

    def _setup_static_context(self, seed: int | None = None) -> None:
        config = replace(self.base_config, random_seed=seed if seed is not None else self.base_config.random_seed)
        self.config = config
        self.times_s = build_time_grid(config)
        self.sat_positions_m = compute_satellite_positions(config, self.times_s)
        self.user_state = compute_user_state(config, self.times_s)
        self.user_positions_m = self.user_state["positions_m"]
        self.link_state = compute_link_state(config, self.sat_positions_m, self.user_positions_m)
        self.num_users = config.num_users
        self.num_steps = len(self.times_s)
        self.num_satellites = self.sat_positions_m.shape[1]
        self.offered_load_bps = config.offered_load_bps
        self.pf_beta = min(1.0, config.dt_s / config.pf_time_constant_s)
        self.satellite_names, self.source_labels = get_satellite_metadata(config=config, num_satellites=self.num_satellites)

        num_users = self.num_users
        num_satellites = self.num_satellites
        self.action_space = Dict(
            {
                "satellite_choices": MultiDiscrete(np.full(num_users, num_satellites + 1, dtype=np.int64)),
                "beam_priority": Box(
                    low=np.zeros(num_users, dtype=np.float32),
                    high=np.ones(num_users, dtype=np.float32),
                    shape=(num_users,),
                    dtype=np.float32,
                ),
                "bandwidth_weight": Box(
                    low=np.zeros(num_users, dtype=np.float32),
                    high=np.ones(num_users, dtype=np.float32),
                    shape=(num_users,),
                    dtype=np.float32,
                ),
                "power_weight": Box(
                    low=np.zeros(num_users, dtype=np.float32),
                    high=np.ones(num_users, dtype=np.float32),
                    shape=(num_users,),
                    dtype=np.float32,
                ),
            }
        )
        self.observation_space = Dict(
            {
                "time_index": Box(np.asarray([0], dtype=np.int32), np.asarray([self.num_steps], dtype=np.int32), shape=(1,), dtype=np.int32),
                "backlog_mbits": Box(
                    low=np.zeros(num_users, dtype=np.float32),
                    high=np.full(num_users, 1e6, dtype=np.float32),
                    shape=(num_users,),
                    dtype=np.float32,
                ),
                "handover_timer_s": Box(
                    low=np.zeros(num_users, dtype=np.float32),
                    high=np.full(num_users, config.handover_penalty_s, dtype=np.float32),
                    shape=(num_users,),
                    dtype=np.float32,
                ),
                "average_rate_mbps": Box(
                    low=np.zeros(num_users, dtype=np.float32),
                    high=np.full(num_users, 1e4, dtype=np.float32),
                    shape=(num_users,),
                    dtype=np.float32,
                ),
                "offered_mbps": Box(
                    low=np.zeros(num_users, dtype=np.float32),
                    high=np.full(num_users, 1e4, dtype=np.float32),
                    shape=(num_users,),
                    dtype=np.float32,
                ),
                "user_altitude_m": Box(
                    low=np.zeros(num_users, dtype=np.float32),
                    high=np.full(num_users, 2e4, dtype=np.float32),
                    shape=(num_users,),
                    dtype=np.float32,
                ),
                "platform_airborne": Box(
                    low=np.zeros(num_users, dtype=np.float32),
                    high=np.ones(num_users, dtype=np.float32),
                    shape=(num_users,),
                    dtype=np.float32,
                ),
                "visible_mask": Box(
                    low=np.zeros((num_users, num_satellites), dtype=np.float32),
                    high=np.ones((num_users, num_satellites), dtype=np.float32),
                    shape=(num_users, num_satellites),
                    dtype=np.float32,
                ),
                "snr_db": Box(
                    low=np.full((num_users, num_satellites), -40.0, dtype=np.float32),
                    high=np.full((num_users, num_satellites), 40.0, dtype=np.float32),
                    shape=(num_users, num_satellites),
                    dtype=np.float32,
                ),
                "elevation_deg": Box(
                    low=np.full((num_users, num_satellites), -90.0, dtype=np.float32),
                    high=np.full((num_users, num_satellites), 90.0, dtype=np.float32),
                    shape=(num_users, num_satellites),
                    dtype=np.float32,
                ),
                "current_satellite": Box(
                    low=np.full(num_users, -1, dtype=np.int32),
                    high=np.full(num_users, num_satellites, dtype=np.int32),
                    shape=(num_users,),
                    dtype=np.int32,
                ),
            }
        )
    def _initialize_episode_state(self) -> None:
        self.time_index = 0
        self.attachment_assignments = np.full(self.num_users, -1, dtype=int)
        self.handover_timers_s = np.zeros(self.num_users, dtype=float)
        self.backlog_bits = np.zeros(self.num_users, dtype=float)
        self.average_rate_bps = np.full(self.num_users, self.config.pf_rate_floor_mbps * 1e6, dtype=float)
        self.total_served_bits = np.zeros(self.num_users, dtype=float)
        self.handover_counts = np.zeros(self.num_users, dtype=int)
        self.history: list[dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self._setup_static_context(seed=seed)
        self._initialize_episode_state()
        observation = self._build_observation()
        info = {
            "time_s": float(self.times_s[self.time_index]) if self.num_steps else 0.0,
            "num_satellites": self.num_satellites,
            "num_users": self.num_users,
            "reward_template": self.reward_template,
            "reward_weights": self.reward_weights.to_dict(),
        }
        return observation, info

    def _current_slot_tensors(self, time_index: int | None = None) -> dict[str, np.ndarray]:
        slot = self.time_index if time_index is None else time_index
        return {
            "visible": self.link_state["visible"][slot],
            "snr_db": self.link_state["snr_fullband_db"][slot],
            "link_capacity_bps": self.link_state["link_capacity_bps"][slot],
            "rx_power_dbm": self.link_state["rx_power_full_dbm"][slot],
            "look_vectors": self.link_state["sat_to_user_unit_vectors"][slot],
            "elevation_deg": self.link_state["elevation_deg"][slot],
            "propagation_delay_ms": self.link_state["propagation_delay_ms"][slot],
        }

    def _build_observation(self, time_index: int | None = None) -> dict[str, np.ndarray]:
        slot = self.time_index if time_index is None else time_index
        tensors = self._current_slot_tensors(time_index=slot)
        platform_airborne = (
            self.user_state["platform_type"][slot] == "airborne"
        ).astype(np.float32)
        snr_db = np.nan_to_num(tensors["snr_db"], nan=-40.0, neginf=-40.0, posinf=40.0)
        elevation = np.nan_to_num(tensors["elevation_deg"], nan=-90.0)
        return {
            "time_index": np.asarray([slot], dtype=np.int32),
            "backlog_mbits": (self.backlog_bits / 1e6).astype(np.float32),
            "handover_timer_s": self.handover_timers_s.astype(np.float32),
            "average_rate_mbps": (self.average_rate_bps / 1e6).astype(np.float32),
            "offered_mbps": (self.offered_load_bps / 1e6).astype(np.float32),
            "user_altitude_m": self.user_state["altitudes_m"][slot].astype(np.float32),
            "platform_airborne": platform_airborne.astype(np.float32),
            "visible_mask": tensors["visible"].astype(np.float32),
            "snr_db": np.clip(snr_db, -40.0, 40.0).astype(np.float32),
            "elevation_deg": np.clip(elevation, -90.0, 90.0).astype(np.float32),
            "current_satellite": self.attachment_assignments.astype(np.int32),
        }

    @staticmethod
    def _normalize_positive_weights(values: np.ndarray) -> np.ndarray:
        clipped = np.maximum(values, 1e-6)
        return clipped / clipped.sum()

    def _allocate_with_action(
        self,
        user_indices: np.ndarray,
        beam_priority: np.ndarray,
        bandwidth_weight: np.ndarray,
        power_weight: np.ndarray,
        rx_power_full_dbm: np.ndarray,
        look_vectors: np.ndarray,
    ) -> dict[str, np.ndarray]:
        num_candidates = user_indices.size
        result = {
            "beam_granted": np.zeros(num_candidates, dtype=bool),
            "allocated_bandwidth_hz": np.zeros(num_candidates, dtype=float),
            "allocated_power_share": np.zeros(num_candidates, dtype=float),
            "allocated_eirp_dbm": np.full(num_candidates, np.nan, dtype=float),
            "assigned_sinr_db": np.full(num_candidates, np.nan, dtype=float),
            "interference_dbm": np.full(num_candidates, np.nan, dtype=float),
            "service_capacity_bps": np.zeros(num_candidates, dtype=float),
        }
        if num_candidates == 0:
            return result

        active_count = min(self.config.max_beams_per_satellite, num_candidates)
        active_local_indices = np.argsort(beam_priority)[::-1][:active_count]
        active_rx_power_dbm = rx_power_full_dbm[active_local_indices]
        active_look_vectors = look_vectors[active_local_indices]

        bandwidth_share = self._normalize_positive_weights(bandwidth_weight[active_local_indices])
        power_share = self._normalize_positive_weights(power_weight[active_local_indices])
        allocated_bandwidth_hz = self.config.total_bandwidth_hz * bandwidth_share
        desired_signal_mw = dbm_to_mw(active_rx_power_dbm) * power_share

        attenuation_db = beam_pattern_attenuation_db(
            config=self.config,
            boresight_vectors=active_look_vectors,
            target_vectors=active_look_vectors,
        )
        bandwidth_overlap = np.minimum.outer(allocated_bandwidth_hz, allocated_bandwidth_hz) / np.maximum(
            allocated_bandwidth_hz[:, None],
            1.0,
        )
        np.fill_diagonal(bandwidth_overlap, 0.0)
        interference_dbm_matrix = (
            active_rx_power_dbm[None, :]
            + linear_to_db(power_share)[None, :]
            - attenuation_db
        )
        interference_mw_matrix = dbm_to_mw(interference_dbm_matrix) * bandwidth_overlap.T
        np.fill_diagonal(interference_mw_matrix, 0.0)
        interference_mw = interference_mw_matrix.sum(axis=1)

        noise_dbm = (
            -174.0
            + 10.0 * np.log10(allocated_bandwidth_hz)
            + self.config.noise_figure_db
        )
        noise_mw = dbm_to_mw(noise_dbm)
        sinr_linear = desired_signal_mw / (noise_mw + interference_mw)
        sinr_db = linear_to_db(sinr_linear)
        spectral_efficiency = np.clip(np.log2(1.0 + sinr_linear), 0.0, 6.0)
        service_capacity_bps = allocated_bandwidth_hz * spectral_efficiency

        result["beam_granted"][active_local_indices] = True
        result["allocated_bandwidth_hz"][active_local_indices] = allocated_bandwidth_hz
        result["allocated_power_share"][active_local_indices] = power_share
        result["allocated_eirp_dbm"][active_local_indices] = self.config.eirp_dbm + linear_to_db(power_share)
        result["assigned_sinr_db"][active_local_indices] = sinr_db
        result["interference_dbm"][active_local_indices] = linear_to_db(np.maximum(interference_mw, 1e-12))
        result["service_capacity_bps"][active_local_indices] = service_capacity_bps
        return result

    def _coerce_action(self, action: dict[str, Any]) -> dict[str, np.ndarray]:
        if not self.action_space.contains(action):
            raise ValueError("Action does not match action_space")
        return {
            key: np.asarray(value)
            for key, value in action.items()
        }

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = self._coerce_action(action)
        if self.time_index >= self.num_steps:
            raise RuntimeError("Episode already finished. Call reset().")

        tensors = self._current_slot_tensors()
        self.backlog_bits += self.offered_load_bps * self.config.dt_s

        requested_assignments = np.full(self.num_users, -1, dtype=int)
        for user_idx in range(self.num_users):
            choice = int(action["satellite_choices"][user_idx])
            if 0 <= choice < self.num_satellites and tensors["visible"][user_idx, choice]:
                requested_assignments[user_idx] = choice

        previous_assignments = self.attachment_assignments.copy()
        handover_event = np.zeros(self.num_users, dtype=bool)
        for user_idx in range(self.num_users):
            if requested_assignments[user_idx] >= 0 and previous_assignments[user_idx] >= 0:
                if requested_assignments[user_idx] != previous_assignments[user_idx]:
                    self.handover_counts[user_idx] += 1
                    self.handover_timers_s[user_idx] = self.config.handover_penalty_s
                    handover_event[user_idx] = True
        self.attachment_assignments = requested_assignments

        handover_blocked = self.handover_timers_s > 0.0
        beam_granted = np.zeros(self.num_users, dtype=bool)
        beam_blocked = np.zeros(self.num_users, dtype=bool)
        service_capacity_bps = np.zeros(self.num_users, dtype=float)
        assigned_sinr_db = np.full(self.num_users, np.nan, dtype=float)

        for sat_idx in range(self.num_satellites):
            user_indices = np.flatnonzero((self.attachment_assignments == sat_idx) & (~handover_blocked))
            if user_indices.size == 0:
                continue
            allocations = self._allocate_with_action(
                user_indices=user_indices,
                beam_priority=action["beam_priority"][user_indices],
                bandwidth_weight=action["bandwidth_weight"][user_indices],
                power_weight=action["power_weight"][user_indices],
                rx_power_full_dbm=tensors["rx_power_dbm"][user_indices, sat_idx],
                look_vectors=tensors["look_vectors"][user_indices, sat_idx],
            )
            beam_granted[user_indices] = allocations["beam_granted"]
            service_capacity_bps[user_indices] = allocations["service_capacity_bps"]
            assigned_sinr_db[user_indices] = allocations["assigned_sinr_db"]

        coverage_outage = self.attachment_assignments < 0
        beam_blocked = (self.attachment_assignments >= 0) & (~handover_blocked) & (~beam_granted)
        service_blocked = coverage_outage | handover_blocked | (~beam_granted)
        service_bits = np.where(
            service_blocked,
            0.0,
            np.minimum(self.backlog_bits, service_capacity_bps * self.config.dt_s),
        )
        self.backlog_bits -= service_bits
        self.total_served_bits += service_bits

        queue_delay_s = np.divide(
            self.backlog_bits,
            self.offered_load_bps,
            out=np.zeros_like(self.backlog_bits),
            where=self.offered_load_bps > 0.0,
        )
        served_rate_bps = service_bits / self.config.dt_s
        self.average_rate_bps = (1.0 - self.pf_beta) * self.average_rate_bps + self.pf_beta * np.maximum(
            served_rate_bps,
            self.config.pf_rate_floor_mbps * 1e6,
        )
        reward = (
            self.reward_weights.throughput_mbps * float(served_rate_bps.sum() / 1e6)
            - self.reward_weights.queue_delay_s * float(queue_delay_s.sum())
            - self.reward_weights.outage_users * float(np.sum(coverage_outage))
            - self.reward_weights.beam_blocked_users * float(np.sum(beam_blocked))
            - self.reward_weights.handover_events * float(np.sum(handover_event))
        )

        for user_idx in range(self.num_users):
            self.history.append(
                {
                    "time_s": float(self.times_s[self.time_index]),
                    "user_id": int(user_idx),
                    "satellite_id": int(self.attachment_assignments[user_idx]),
                    "beam_granted": bool(beam_granted[user_idx]),
                    "coverage_outage": bool(coverage_outage[user_idx]),
                    "beam_blocked": bool(beam_blocked[user_idx]),
                    "handover_event": bool(handover_event[user_idx]),
                    "served_mbps": float(served_rate_bps[user_idx] / 1e6),
                    "queue_delay_s": float(queue_delay_s[user_idx]),
                    "assigned_sinr_db": float(assigned_sinr_db[user_idx]) if not np.isnan(assigned_sinr_db[user_idx]) else np.nan,
                    "platform_type": str(self.user_state["platform_type"][self.time_index, user_idx]),
                    "user_altitude_m": float(self.user_state["altitudes_m"][self.time_index, user_idx]),
                }
            )

        self.handover_timers_s = np.maximum(0.0, self.handover_timers_s - self.config.dt_s)
        self.time_index += 1
        terminated = False
        truncated = self.time_index >= self.num_steps
        observation = self._build_observation() if not truncated else self._terminal_observation()
        info = {
            "reward_terms": {
                "throughput_mbps": float(served_rate_bps.sum() / 1e6),
                "queue_delay_sum_s": float(queue_delay_s.sum()),
                "outage_users": int(np.sum(coverage_outage)),
                "beam_blocked_users": int(np.sum(beam_blocked)),
                "handover_events": int(np.sum(handover_event)),
            },
            "episode_summary": self.episode_summary() if truncated else None,
        }
        return observation, reward, terminated, truncated, info

    def _terminal_observation(self) -> dict[str, np.ndarray]:
        slot = max(0, self.num_steps - 1)
        observation = self._build_observation(time_index=slot)
        observation["time_index"] = np.asarray([self.num_steps], dtype=np.int32)
        return observation

    def action_from_policy(self, policy: str) -> dict[str, np.ndarray]:
        normalized_policy, access_policy, resource_policy = parse_policy_name(policy)
        tensors = self._current_slot_tensors()
        horizon_end = min(self.time_index + self.config.mpc_horizon_steps, self.num_steps)
        assignments = select_assignments_for_slot(
            config=self.config,
            access_policy=access_policy,
            resource_policy=resource_policy,
            current_assignments=self.attachment_assignments,
            visible_t=tensors["visible"],
            snr_t=tensors["snr_db"],
            backlog_bits=self.backlog_bits,
            offered_load_bps=self.offered_load_bps,
            average_rate_bps=self.average_rate_bps,
            base_link_capacity_t=tensors["link_capacity_bps"],
            rx_power_t=tensors["rx_power_dbm"],
            look_vectors_t=tensors["look_vectors"],
            visible_horizon=self.link_state["visible"][self.time_index:horizon_end],
            base_link_capacity_horizon=self.link_state["link_capacity_bps"][self.time_index:horizon_end],
            rx_power_horizon=self.link_state["rx_power_full_dbm"][self.time_index:horizon_end],
            look_vectors_horizon=self.link_state["sat_to_user_unit_vectors"][self.time_index:horizon_end],
        )
        satellite_choices = np.where(assignments >= 0, assignments, self.num_satellites)
        demand_bits = self.backlog_bits + self.offered_load_bps * self.config.dt_s
        beam_priority = np.zeros(self.num_users, dtype=np.float32)
        bandwidth_weight = np.ones(self.num_users, dtype=np.float32)
        power_weight = np.ones(self.num_users, dtype=np.float32)

        for user_idx in range(self.num_users):
            sat_idx = assignments[user_idx]
            if sat_idx < 0:
                continue
            objective = compute_control_objective(
                resource_policy=resource_policy,
                rate_bps=float(tensors["link_capacity_bps"][user_idx, sat_idx]),
                demand_bits=float(demand_bits[user_idx]),
                average_rate_bps=float(self.average_rate_bps[user_idx]),
                config=self.config,
            )
            beam_priority[user_idx] = float(objective)
            bandwidth_weight[user_idx] = max(float(objective), 1.0)
            power_weight[user_idx] = max(float(np.sqrt(objective + 1e-6)), 1.0)

        if resource_policy == "equal":
            beam_priority[:] = np.where(assignments >= 0, 1.0, 0.0)
            bandwidth_weight[:] = 1.0
            power_weight[:] = 1.0
        return {
            "satellite_choices": satellite_choices.astype(np.int64),
            "beam_priority": beam_priority.astype(np.float32),
            "bandwidth_weight": bandwidth_weight.astype(np.float32),
            "power_weight": power_weight.astype(np.float32),
        }

    def episode_summary(self) -> dict[str, Any]:
        frame = self.get_episode_frame()
        if frame.empty:
            return {
                "system_throughput_mbps": 0.0,
                "total_handovers": 0,
                "mean_queue_delay_s": 0.0,
            }
        served_mbps = float(frame["served_mbps"].sum() / max(self.num_steps, 1))
        return {
            "system_throughput_mbps": round(served_mbps, 4),
            "total_handovers": int(frame["handover_event"].sum()),
            "mean_queue_delay_s": round(float(frame["queue_delay_s"].mean()), 4),
        }

    def get_episode_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.history)

    def render(self) -> str:
        summary = self.episode_summary()
        return (
            f"time_index={self.time_index}/{self.num_steps} "
            f"throughput={summary['system_throughput_mbps']:.2f}Mbps "
            f"handover_events={summary['total_handovers']} "
            f"mean_queue_delay={summary['mean_queue_delay_s']:.2f}s"
        )
