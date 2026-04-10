from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import json
import math
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

VENDOR_PATH = Path(__file__).resolve().parent.parent / "_vendor"
if VENDOR_PATH.exists() and str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))

try:
    from sgp4.api import Satrec
except Exception:
    Satrec = None

EARTH_RADIUS_M = 6_371_000.0
EARTH_MU = 3.986_004_418e14
LIGHT_SPEED_MPS = 299_792_458.0
BOLTZMANN_DBM_PER_HZ = -174.0
EARTH_ROTATION_RAD_S = 7.2921159e-5
J2 = 1.08262668e-3
WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
SUPPORTED_ACCESS_POLICIES = ("greedy", "sticky", "joint")
SUPPORTED_RESOURCE_POLICIES = ("equal", "proportional_fair", "lyapunov")
CANONICAL_POLICIES = tuple(
    f"{access}_{resource}"
    for access in SUPPORTED_ACCESS_POLICIES
    for resource in SUPPORTED_RESOURCE_POLICIES
)
SUPPORTED_POLICIES = ("greedy", "sticky") + CANONICAL_POLICIES


@dataclass(slots=True)
class SimulationConfig:
    duration_s: int = 1800
    dt_s: float = 1.0
    constellation_mode: str = "synthetic"
    num_satellites: int = 48
    num_orbit_planes: int = 8
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    synthetic_j2_enabled: bool = True
    use_wgs84_earth: bool = True
    carrier_frequency_hz: float = 2e9
    total_bandwidth_hz: float = 15e6
    eirp_dbm: float = 65.0
    noise_figure_db: float = 5.0
    elevation_mask_deg: float = 5.0
    hysteresis_db: float = 4.0
    handover_penalty_s: float = 5.0
    orbit_longitude_offset_deg: float = 0.0
    shadowing_sigma_db: float = 2.0
    random_seed: int = 7
    max_beams_per_satellite: int = 2
    pf_time_constant_s: float = 120.0
    pf_rate_floor_mbps: float = 1.0
    lyapunov_control_weight: float = 6.0
    beam_3db_width_deg: float = 6.0
    beam_max_attenuation_db: float = 22.0
    joint_switch_penalty: float = 0.75
    joint_min_service_mbps: float = 4.0
    joint_unserved_penalty_scale: float = 1.2
    mpc_horizon_steps: int = 3
    mpc_discount: float = 0.85
    earth_rotation_enabled: bool = True
    user_mobility_mode: str = "static"
    user_speeds_kmph: tuple[float, ...] = ()
    user_initial_speeds_kmph: tuple[float, ...] = ()
    user_headings_deg: tuple[float, ...] = ()
    user_altitudes_m: tuple[float, ...] = ()
    user_target_altitudes_m: tuple[float, ...] = ()
    user_waypoints_deg: tuple[Any, ...] = ()
    user_waypoint_times_s: tuple[Any, ...] = ()
    road_turn_pattern_deg: tuple[float, ...] = (35.0, -45.0, 25.0)
    road_segment_fraction: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    flight_corridor_length_km: float = 1200.0
    flight_corridor_bend_deg: float = 8.0
    flight_climb_fraction: float = 0.2
    flight_descent_fraction: float = 0.2
    airborne_altitude_threshold_m: float = 1000.0
    airborne_dynamic_enabled: bool = True
    airborne_max_turn_rate_deg_s: float = 3.0
    airborne_max_climb_rate_mps: float = 20.0
    airborne_max_descent_rate_mps: float = 15.0
    airborne_max_accel_mps2: float = 2.5
    airborne_max_decel_mps2: float = 3.5
    airborne_bank_angle_deg: float = 25.0
    airborne_min_turn_radius_m: float = 4000.0
    airborne_waypoint_capture_radius_m: float = 15000.0
    airborne_path_lookahead_m: float = 30000.0
    tle_file: str | None = None
    tle_files: tuple[str, ...] = ()
    tle_selection_mode: str = "round_robin"
    tle_start_offset_s: float = 0.0
    tle_max_satellites: int = 64
    tle_source_max_satellites: tuple[int, ...] = ()
    user_latitudes_deg: tuple[float, ...] = (35.0, 28.0, 18.0, 0.0, -12.0, 22.0, 32.0)
    user_longitudes_deg: tuple[float, ...] = (-7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0)
    offered_load_mbps: tuple[float, ...] = (15.0, 19.0, 23.0, 27.0, 23.0, 19.0, 15.0)

    def __post_init__(self) -> None:
        if not self.user_speeds_kmph:
            self.user_speeds_kmph = tuple(0.0 for _ in self.user_longitudes_deg)
        if not self.user_initial_speeds_kmph:
            self.user_initial_speeds_kmph = tuple(value for value in self.user_speeds_kmph)
        if not self.user_headings_deg:
            self.user_headings_deg = tuple(0.0 for _ in self.user_longitudes_deg)
        if not self.user_altitudes_m:
            self.user_altitudes_m = tuple(0.0 for _ in self.user_longitudes_deg)
        if not self.user_target_altitudes_m:
            self.user_target_altitudes_m = tuple(value for value in self.user_altitudes_m)

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "SimulationConfig":
        if path is None:
            config = cls()
        else:
            with Path(path).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if "user_longitudes_deg" in payload:
                payload["user_longitudes_deg"] = tuple(payload["user_longitudes_deg"])
            if "user_latitudes_deg" in payload:
                payload["user_latitudes_deg"] = tuple(payload["user_latitudes_deg"])
            if "user_speeds_kmph" in payload:
                payload["user_speeds_kmph"] = tuple(payload["user_speeds_kmph"])
            if "user_initial_speeds_kmph" in payload:
                payload["user_initial_speeds_kmph"] = tuple(payload["user_initial_speeds_kmph"])
            if "user_headings_deg" in payload:
                payload["user_headings_deg"] = tuple(payload["user_headings_deg"])
            if "user_altitudes_m" in payload:
                payload["user_altitudes_m"] = tuple(payload["user_altitudes_m"])
            if "user_target_altitudes_m" in payload:
                payload["user_target_altitudes_m"] = tuple(payload["user_target_altitudes_m"])
            if "user_waypoints_deg" in payload:
                payload["user_waypoints_deg"] = tuple(
                    tuple(tuple(point) for point in track)
                    for track in payload["user_waypoints_deg"]
                )
            if "user_waypoint_times_s" in payload:
                payload["user_waypoint_times_s"] = tuple(
                    tuple(track_times) for track_times in payload["user_waypoint_times_s"]
                )
            if "offered_load_mbps" in payload:
                payload["offered_load_mbps"] = tuple(payload["offered_load_mbps"])
            if "tle_files" in payload:
                payload["tle_files"] = tuple(payload["tle_files"])
            if "tle_source_max_satellites" in payload:
                payload["tle_source_max_satellites"] = tuple(payload["tle_source_max_satellites"])
            if "road_turn_pattern_deg" in payload:
                payload["road_turn_pattern_deg"] = tuple(payload["road_turn_pattern_deg"])
            if "road_segment_fraction" in payload:
                payload["road_segment_fraction"] = tuple(payload["road_segment_fraction"])
            config = cls(**payload)
        if not config.user_speeds_kmph:
            config.user_speeds_kmph = tuple(0.0 for _ in config.user_longitudes_deg)
        if not config.user_initial_speeds_kmph:
            config.user_initial_speeds_kmph = tuple(value for value in config.user_speeds_kmph)
        if not config.user_headings_deg:
            config.user_headings_deg = tuple(0.0 for _ in config.user_longitudes_deg)
        if not config.user_altitudes_m:
            config.user_altitudes_m = tuple(0.0 for _ in config.user_longitudes_deg)
        if not config.user_target_altitudes_m:
            config.user_target_altitudes_m = tuple(value for value in config.user_altitudes_m)
        config.validate()
        return config

    @property
    def num_users(self) -> int:
        return len(self.user_longitudes_deg)

    @property
    def offered_load_bps(self) -> np.ndarray:
        return np.asarray(self.offered_load_mbps, dtype=float) * 1e6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if self.constellation_mode not in ("synthetic", "tle"):
            raise ValueError("constellation_mode must be 'synthetic' or 'tle'")
        if self.constellation_mode == "synthetic":
            if self.num_satellites < 2:
                raise ValueError("num_satellites must be at least 2")
            if self.num_orbit_planes <= 0:
                raise ValueError("num_orbit_planes must be positive")
            if self.num_satellites % self.num_orbit_planes != 0:
                raise ValueError("num_satellites must be divisible by num_orbit_planes")
            if not (0.0 <= self.inclination_deg <= 180.0):
                raise ValueError("inclination_deg must be in [0, 180]")
        if self.total_bandwidth_hz <= 0:
            raise ValueError("total_bandwidth_hz must be positive")
        if self.shadowing_sigma_db < 0:
            raise ValueError("shadowing_sigma_db cannot be negative")
        if self.max_beams_per_satellite <= 0:
            raise ValueError("max_beams_per_satellite must be positive")
        if self.pf_time_constant_s <= 0:
            raise ValueError("pf_time_constant_s must be positive")
        if self.pf_rate_floor_mbps <= 0:
            raise ValueError("pf_rate_floor_mbps must be positive")
        if self.beam_3db_width_deg <= 0:
            raise ValueError("beam_3db_width_deg must be positive")
        if self.beam_max_attenuation_db <= 0:
            raise ValueError("beam_max_attenuation_db must be positive")
        if self.joint_switch_penalty < 0:
            raise ValueError("joint_switch_penalty cannot be negative")
        if self.joint_min_service_mbps <= 0:
            raise ValueError("joint_min_service_mbps must be positive")
        if self.joint_unserved_penalty_scale < 0:
            raise ValueError("joint_unserved_penalty_scale cannot be negative")
        if self.mpc_horizon_steps <= 0:
            raise ValueError("mpc_horizon_steps must be positive")
        if not (0 < self.mpc_discount <= 1):
            raise ValueError("mpc_discount must be in (0, 1]")
        if self.tle_max_satellites <= 0:
            raise ValueError("tle_max_satellites must be positive")
        if self.tle_selection_mode not in ("round_robin", "concat"):
            raise ValueError("tle_selection_mode must be 'round_robin' or 'concat'")
        if self.user_mobility_mode not in ("static", "linear", "waypoint", "road", "flight_corridor"):
            raise ValueError("user_mobility_mode must be one of static/linear/waypoint/road/flight_corridor")
        if len(self.user_longitudes_deg) == 0:
            raise ValueError("user_longitudes_deg cannot be empty")
        if len(self.user_latitudes_deg) != len(self.user_longitudes_deg):
            raise ValueError("user_latitudes_deg and user_longitudes_deg must have the same length")
        if len(self.user_speeds_kmph) != len(self.user_longitudes_deg):
            raise ValueError("user_speeds_kmph must have the same length as user_longitudes_deg")
        if len(self.user_initial_speeds_kmph) != len(self.user_longitudes_deg):
            raise ValueError("user_initial_speeds_kmph must have the same length as user_longitudes_deg")
        if len(self.user_headings_deg) != len(self.user_longitudes_deg):
            raise ValueError("user_headings_deg must have the same length as user_longitudes_deg")
        if len(self.user_altitudes_m) != len(self.user_longitudes_deg):
            raise ValueError("user_altitudes_m must have the same length as user_longitudes_deg")
        if len(self.user_target_altitudes_m) != len(self.user_longitudes_deg):
            raise ValueError("user_target_altitudes_m must have the same length as user_longitudes_deg")
        if len(self.user_longitudes_deg) != len(self.offered_load_mbps):
            raise ValueError("user_longitudes_deg and offered_load_mbps must have the same length")
        if any(load < 0 for load in self.offered_load_mbps):
            raise ValueError("offered_load_mbps cannot contain negative values")
        if any(speed < 0 for speed in self.user_speeds_kmph):
            raise ValueError("user_speeds_kmph cannot contain negative values")
        if any(speed < 0 for speed in self.user_initial_speeds_kmph):
            raise ValueError("user_initial_speeds_kmph cannot contain negative values")
        if any(altitude < 0 for altitude in self.user_altitudes_m):
            raise ValueError("user_altitudes_m cannot contain negative values")
        if any(altitude < 0 for altitude in self.user_target_altitudes_m):
            raise ValueError("user_target_altitudes_m cannot contain negative values")
        if self.road_segment_fraction and sorted(self.road_segment_fraction) != list(self.road_segment_fraction):
            raise ValueError("road_segment_fraction must be sorted ascending")
        if self.road_segment_fraction and self.road_segment_fraction[-1] != 1.0:
            raise ValueError("road_segment_fraction must end at 1.0")
        if self.flight_corridor_length_km <= 0:
            raise ValueError("flight_corridor_length_km must be positive")
        if self.flight_corridor_bend_deg < 0:
            raise ValueError("flight_corridor_bend_deg cannot be negative")
        if self.flight_climb_fraction < 0 or self.flight_descent_fraction < 0:
            raise ValueError("flight climb/descent fractions cannot be negative")
        if self.flight_climb_fraction + self.flight_descent_fraction >= 1.0:
            raise ValueError("flight climb/descent fractions must sum to less than 1")
        if self.airborne_altitude_threshold_m < 0:
            raise ValueError("airborne_altitude_threshold_m cannot be negative")
        if self.airborne_max_turn_rate_deg_s <= 0:
            raise ValueError("airborne_max_turn_rate_deg_s must be positive")
        if self.airborne_max_climb_rate_mps <= 0:
            raise ValueError("airborne_max_climb_rate_mps must be positive")
        if self.airborne_max_descent_rate_mps <= 0:
            raise ValueError("airborne_max_descent_rate_mps must be positive")
        if self.airborne_max_accel_mps2 <= 0:
            raise ValueError("airborne_max_accel_mps2 must be positive")
        if self.airborne_max_decel_mps2 <= 0:
            raise ValueError("airborne_max_decel_mps2 must be positive")
        if self.airborne_bank_angle_deg <= 0:
            raise ValueError("airborne_bank_angle_deg must be positive")
        if self.airborne_min_turn_radius_m <= 0:
            raise ValueError("airborne_min_turn_radius_m must be positive")
        if self.airborne_waypoint_capture_radius_m <= 0:
            raise ValueError("airborne_waypoint_capture_radius_m must be positive")
        if self.airborne_path_lookahead_m <= 0:
            raise ValueError("airborne_path_lookahead_m must be positive")
        if self.user_waypoints_deg and len(self.user_waypoints_deg) != len(self.user_longitudes_deg):
            raise ValueError("user_waypoints_deg must match user count when provided")
        if self.user_waypoint_times_s and len(self.user_waypoint_times_s) != len(self.user_longitudes_deg):
            raise ValueError("user_waypoint_times_s must match user count when provided")
        if self.user_waypoint_times_s and not self.user_waypoints_deg:
            raise ValueError("user_waypoint_times_s requires user_waypoints_deg")
        for track in self.user_waypoints_deg:
            if len(track) < 2:
                raise ValueError("Each user waypoint track must have at least 2 points")
        for track_index, time_track in enumerate(self.user_waypoint_times_s):
            if len(time_track) != len(self.user_waypoints_deg[track_index]):
                raise ValueError("user_waypoint_times_s entries must align with user_waypoints_deg")
            if any(time_track[i] > time_track[i + 1] for i in range(len(time_track) - 1)):
                raise ValueError("user_waypoint_times_s must be nondecreasing")
        if self.constellation_mode == "tle":
            if not (self.tle_file or self.tle_files):
                raise ValueError("tle_file or tle_files is required when constellation_mode='tle'")
            if self.tle_source_max_satellites and self.tle_files and len(self.tle_source_max_satellites) != len(self.tle_files):
                raise ValueError("tle_source_max_satellites must match tle_files length")


def normalize_policy_name(policy: str) -> str:
    if policy in ("greedy", "sticky"):
        return f"{policy}_equal"
    if policy in CANONICAL_POLICIES:
        return policy
    raise ValueError(
        f"Unsupported policy '{policy}'. Supported policies: {', '.join(SUPPORTED_POLICIES)}"
    )


def parse_policy_name(policy: str) -> tuple[str, str, str]:
    normalized = normalize_policy_name(policy)
    access_policy, resource_policy = normalized.split("_", 1)
    return normalized, access_policy, resource_policy


def db_to_linear(value_db: np.ndarray | float) -> np.ndarray | float:
    return np.power(10.0, np.asarray(value_db, dtype=float) / 10.0)


def linear_to_db(value_linear: np.ndarray | float) -> np.ndarray | float:
    clipped = np.maximum(np.asarray(value_linear, dtype=float), 1e-12)
    result = 10.0 * np.log10(clipped)
    if np.isscalar(value_linear):
        return float(result)
    return result


def dbm_to_mw(value_dbm: np.ndarray | float) -> np.ndarray | float:
    return db_to_linear(value_dbm)


def mw_to_dbm(value_mw: np.ndarray | float) -> np.ndarray | float:
    return linear_to_db(value_mw)


def orbital_angular_velocity_rad_s(config: SimulationConfig) -> float:
    orbital_radius_m = EARTH_RADIUS_M + config.altitude_km * 1e3
    return math.sqrt(EARTH_MU / orbital_radius_m**3)


def build_time_grid(config: SimulationConfig) -> np.ndarray:
    steps = int(config.duration_s / config.dt_s)
    return np.arange(steps, dtype=float) * config.dt_s


def _rotation_matrix_z(angle_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.cos(angle_rad), np.sin(angle_rad)


def compute_synthetic_satellite_positions(config: SimulationConfig, times_s: np.ndarray) -> np.ndarray:
    orbital_radius_m = WGS84_A_M + config.altitude_km * 1e3
    satellites_per_plane = config.num_satellites // config.num_orbit_planes
    inclination_rad = math.radians(config.inclination_deg)
    angular_velocity = math.sqrt(EARTH_MU / orbital_radius_m**3)
    sin_i = math.sin(inclination_rad)
    cos_i = math.cos(inclination_rad)
    j2_factor = J2 * (WGS84_A_M / orbital_radius_m) ** 2
    if config.synthetic_j2_enabled:
        raan_rate = -1.5 * j2_factor * angular_velocity * cos_i
        argument_rate = angular_velocity * (1.0 + 0.75 * j2_factor * (2.0 - 3.0 * sin_i**2))
    else:
        raan_rate = 0.0
        argument_rate = angular_velocity
    raan_values = np.linspace(0.0, 2.0 * math.pi, config.num_orbit_planes, endpoint=False)
    raan_values += math.radians(config.orbit_longitude_offset_deg)

    positions = np.zeros((times_s.size, config.num_satellites, 3), dtype=float)
    sat_index = 0
    for plane_idx, raan in enumerate(raan_values):
        for sat_in_plane in range(satellites_per_plane):
            base_phase = 2.0 * math.pi * sat_in_plane / satellites_per_plane
            plane_phase = 2.0 * math.pi * plane_idx / config.num_satellites
            argument_of_latitude = argument_rate * times_s + base_phase + plane_phase
            current_raan = raan + raan_rate * times_s
            cos_u = np.cos(argument_of_latitude)
            sin_u = np.sin(argument_of_latitude)
            cos_raan = np.cos(current_raan)
            sin_raan = np.sin(current_raan)
            positions[:, sat_index, 0] = orbital_radius_m * (
                cos_raan * cos_u - sin_raan * sin_u * cos_i
            )
            positions[:, sat_index, 1] = orbital_radius_m * (
                sin_raan * cos_u + cos_raan * sin_u * cos_i
            )
            positions[:, sat_index, 2] = orbital_radius_m * sin_u * sin_i
            sat_index += 1
    return positions


def infer_tle_source_label(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith("starlink"):
        return "starlink"
    if stem.startswith("oneweb"):
        return "oneweb"
    return stem


def parse_tle_records_from_path(path: Path) -> list[tuple[str, str, Any]]:
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_label = infer_tle_source_label(path)
    records: list[tuple[str, str, Any]] = []
    index = 0
    while index < len(raw_lines):
        line = raw_lines[index]
        if line.startswith("1 ") and index + 1 < len(raw_lines):
            name = f"sat_{len(records):03d}"
            line1 = raw_lines[index]
            line2 = raw_lines[index + 1]
            index += 2
        elif index + 2 < len(raw_lines) and raw_lines[index + 1].startswith("1 ") and raw_lines[index + 2].startswith("2 "):
            name = raw_lines[index]
            line1 = raw_lines[index + 1]
            line2 = raw_lines[index + 2]
            index += 3
        else:
            raise ValueError(f"Invalid TLE format near line: {line}")
        sat = Satrec.twoline2rv(line1, line2)
        records.append((source_label, name, sat))
    return records


def load_tle_records(config: SimulationConfig) -> list[tuple[str, str, Any]]:
    if Satrec is None:
        raise RuntimeError("sgp4 is not available. Install it into the repo vendor directory first.")
    source_paths = [Path(path).resolve() for path in config.tle_files] if config.tle_files else [Path(config.tle_file).resolve()]
    source_records: list[list[tuple[str, str, Any]]] = []
    for source_path in source_paths:
        parsed = parse_tle_records_from_path(source_path)
        source_records.append(
            [
                (source_label, f"{source_path.stem}:{name}", sat)
                for source_label, name, sat in parsed
            ]
        )
    if not source_records or not any(source_records):
        raise ValueError("No TLE records were loaded")

    source_limits = (
        list(config.tle_source_max_satellites)
        if config.tle_source_max_satellites
        else [config.tle_max_satellites for _ in source_paths]
    )
    selected: list[tuple[str, str, Any]] = []
    if config.tle_selection_mode == "concat":
        for source_index, records in enumerate(source_records):
            limit = min(source_limits[source_index], len(records))
            selected.extend(records[:limit])
            if len(selected) >= config.tle_max_satellites:
                break
        return selected[: config.tle_max_satellites]

    source_offsets = [0 for _ in source_paths]
    selected_per_source = [0 for _ in source_paths]
    while len(selected) < config.tle_max_satellites:
        progress = False
        for source_index, records in enumerate(source_records):
            if selected_per_source[source_index] >= source_limits[source_index]:
                continue
            if source_offsets[source_index] >= len(records):
                continue
            selected.append(records[source_offsets[source_index]])
            source_offsets[source_index] += 1
            selected_per_source[source_index] += 1
            progress = True
            if len(selected) >= config.tle_max_satellites:
                break
        if not progress:
            break
    return selected


def compute_tle_satellite_positions(config: SimulationConfig, times_s: np.ndarray) -> np.ndarray:
    records = load_tle_records(config)
    reference_jd = records[0][2].jdsatepoch + records[0][2].jdsatepochF + config.tle_start_offset_s / 86400.0
    positions = np.zeros((times_s.size, len(records), 3), dtype=float)
    for sat_index, (_, _, sat) in enumerate(records):
        for time_index, time_s in enumerate(times_s):
            current_jd = reference_jd + time_s / 86400.0
            jd_int = math.floor(current_jd)
            jd_frac = current_jd - jd_int
            error_code, position_km, _ = sat.sgp4(jd_int, jd_frac)
            if error_code != 0:
                raise RuntimeError(f"SGP4 propagation failed for satellite {sat_index} with code {error_code}")
            positions[time_index, sat_index] = np.asarray(position_km, dtype=float) * 1e3
    return positions


def compute_satellite_positions(config: SimulationConfig, times_s: np.ndarray) -> np.ndarray:
    if config.constellation_mode == "tle":
        return compute_tle_satellite_positions(config=config, times_s=times_s)
    return compute_synthetic_satellite_positions(config=config, times_s=times_s)


def get_satellite_metadata(config: SimulationConfig, num_satellites: int) -> tuple[np.ndarray, np.ndarray]:
    if config.constellation_mode == "tle":
        records = load_tle_records(config)[:num_satellites]
        source_labels = np.asarray([source_label for source_label, _, _ in records], dtype=object)
        satellite_names = np.asarray([name for _, name, _ in records], dtype=object)
        return satellite_names, source_labels

    satellites_per_plane = num_satellites // config.num_orbit_planes
    source_labels: list[str] = []
    satellite_names: list[str] = []
    sat_index = 0
    for plane_idx in range(config.num_orbit_planes):
        for sat_in_plane in range(satellites_per_plane):
            source_labels.append(f"plane_{plane_idx:02d}")
            satellite_names.append(f"synthetic_{plane_idx:02d}_{sat_in_plane:02d}")
            sat_index += 1
    return np.asarray(satellite_names, dtype=object), np.asarray(source_labels, dtype=object)


def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    radius_m = WGS84_A_M
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    bearing = math.radians(bearing_deg)
    angular_distance = distance_m / radius_m
    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    sin_ad = math.sin(angular_distance)
    cos_ad = math.cos(angular_distance)
    lat2 = math.asin(sin_lat1 * cos_ad + cos_lat1 * sin_ad * math.cos(bearing))
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * sin_ad * cos_lat1,
        cos_ad - sin_lat1 * math.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi
    return math.degrees(lat2), math.degrees(lon2)


def initial_bearing_deg(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    delta_lon = math.radians(lon2_deg - lon1_deg)
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def wrap_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def great_circle_distance_m(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(lon2_deg - lon1_deg)
    hav = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * WGS84_A_M * math.asin(min(1.0, math.sqrt(hav)))


def local_tangent_offset_m(
    ref_lat_deg: float,
    ref_lon_deg: float,
    lat_deg: float,
    lon_deg: float,
) -> tuple[float, float]:
    delta_lat_deg = lat_deg - ref_lat_deg
    delta_lon_deg = wrap_angle_deg(lon_deg - ref_lon_deg)
    north_m = math.radians(delta_lat_deg) * WGS84_A_M
    east_m = math.radians(delta_lon_deg) * WGS84_A_M * math.cos(math.radians(ref_lat_deg))
    return east_m, north_m


def segment_guidance_heading_deg(
    current_lat_deg: float,
    current_lon_deg: float,
    segment_start_lat_deg: float,
    segment_start_lon_deg: float,
    segment_end_lat_deg: float,
    segment_end_lon_deg: float,
    lookahead_m: float,
) -> tuple[float, float, float]:
    start_e, start_n = 0.0, 0.0
    end_e, end_n = local_tangent_offset_m(
        segment_start_lat_deg,
        segment_start_lon_deg,
        segment_end_lat_deg,
        segment_end_lon_deg,
    )
    cur_e, cur_n = local_tangent_offset_m(
        segment_start_lat_deg,
        segment_start_lon_deg,
        current_lat_deg,
        current_lon_deg,
    )
    seg = np.asarray([end_e - start_e, end_n - start_n], dtype=float)
    pos = np.asarray([cur_e - start_e, cur_n - start_n], dtype=float)
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1.0:
        return (
            initial_bearing_deg(current_lat_deg, current_lon_deg, segment_end_lat_deg, segment_end_lon_deg),
            0.0,
            0.0,
            0.0,
            0.0,
        )
    seg_unit = seg / seg_len
    along_m = float(np.clip(np.dot(pos, seg_unit), 0.0, seg_len))
    closest = seg_unit * along_m
    error = pos - closest
    cross_track_m = float(seg_unit[0] * pos[1] - seg_unit[1] * pos[0])
    segment_heading_deg = (math.degrees(math.atan2(seg[0], seg[1])) + 360.0) % 360.0
    correction_deg = math.degrees(math.atan2(-cross_track_m, max(lookahead_m, 1.0)))
    desired_heading_deg = (segment_heading_deg + correction_deg + 360.0) % 360.0
    distance_to_end_m = float(np.linalg.norm(pos - seg))
    return desired_heading_deg, distance_to_end_m, along_m, cross_track_m, segment_heading_deg


def interpolate_waypoint_track(
    times_s: np.ndarray,
    waypoint_times_s: np.ndarray,
    waypoint_lats_deg: np.ndarray,
    waypoint_lons_deg: np.ndarray,
    waypoint_alts_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unwrapped_lons = np.unwrap(np.radians(waypoint_lons_deg))
    latitudes = np.interp(times_s, waypoint_times_s, waypoint_lats_deg)
    longitudes = np.interp(times_s, waypoint_times_s, np.degrees(unwrapped_lons))
    altitudes = np.interp(times_s, waypoint_times_s, waypoint_alts_m)
    longitudes = (longitudes + 180.0) % 360.0 - 180.0
    return latitudes, longitudes, altitudes


def propagate_airborne_track(
    config: SimulationConfig,
    times_s: np.ndarray,
    initial_heading_deg: float,
    waypoint_times_s: np.ndarray,
    waypoint_lats_deg: np.ndarray,
    waypoint_lons_deg: np.ndarray,
    waypoint_alts_m: np.ndarray,
    initial_speed_mps: float,
    target_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    latitudes = np.zeros(times_s.size, dtype=float)
    longitudes = np.zeros(times_s.size, dtype=float)
    altitudes = np.zeros(times_s.size, dtype=float)
    lateral_path_error_m = np.zeros(times_s.size, dtype=float)
    heading_error_deg_series = np.zeros(times_s.size, dtype=float)
    turn_anticipation_distance_m = np.zeros(times_s.size, dtype=float)
    turn_switch_event = np.zeros(times_s.size, dtype=bool)
    latitudes[0] = float(waypoint_lats_deg[0])
    longitudes[0] = float(waypoint_lons_deg[0])
    altitudes[0] = float(waypoint_alts_m[0])
    heading_deg = float(initial_heading_deg)
    speed_mps = float(initial_speed_mps)
    segment_index = 0

    for time_index in range(1, times_s.size):
        dt = float(times_s[time_index] - times_s[time_index - 1])
        current_time = float(times_s[time_index - 1])
        next_time = np.asarray([times_s[time_index]], dtype=float)
        target_lat, target_lon, target_alt = interpolate_waypoint_track(
            times_s=next_time,
            waypoint_times_s=waypoint_times_s,
            waypoint_lats_deg=waypoint_lats_deg,
            waypoint_lons_deg=waypoint_lons_deg,
            waypoint_alts_m=waypoint_alts_m,
        )
        target_lat = float(target_lat[0])
        target_lon = float(target_lon[0])
        target_alt = float(target_alt[0])
        while segment_index < waypoint_lats_deg.size - 2:
            _, distance_to_end_m, along_m, _, _ = segment_guidance_heading_deg(
                current_lat_deg=latitudes[time_index - 1],
                current_lon_deg=longitudes[time_index - 1],
                segment_start_lat_deg=float(waypoint_lats_deg[segment_index]),
                segment_start_lon_deg=float(waypoint_lons_deg[segment_index]),
                segment_end_lat_deg=float(waypoint_lats_deg[segment_index + 1]),
                segment_end_lon_deg=float(waypoint_lons_deg[segment_index + 1]),
                lookahead_m=config.airborne_path_lookahead_m,
            )
            segment_length_m = great_circle_distance_m(
                float(waypoint_lats_deg[segment_index]),
                float(waypoint_lons_deg[segment_index]),
                float(waypoint_lats_deg[segment_index + 1]),
                float(waypoint_lons_deg[segment_index + 1]),
            )
            if (
                distance_to_end_m <= config.airborne_waypoint_capture_radius_m
                or along_m >= max(0.0, segment_length_m - config.airborne_waypoint_capture_radius_m)
                or current_time >= float(waypoint_times_s[segment_index + 1])
            ):
                turn_switch_event[time_index] = True
                turn_anticipation_distance_m[time_index] = distance_to_end_m
                segment_index += 1
            else:
                break

        if segment_index < waypoint_lats_deg.size - 1:
            desired_heading_deg, _, _, cross_track_m, segment_heading_deg = segment_guidance_heading_deg(
                current_lat_deg=latitudes[time_index - 1],
                current_lon_deg=longitudes[time_index - 1],
                segment_start_lat_deg=float(waypoint_lats_deg[segment_index]),
                segment_start_lon_deg=float(waypoint_lons_deg[segment_index]),
                segment_end_lat_deg=float(waypoint_lats_deg[segment_index + 1]),
                segment_end_lon_deg=float(waypoint_lons_deg[segment_index + 1]),
                lookahead_m=config.airborne_path_lookahead_m,
            )
        else:
            desired_heading_deg = initial_bearing_deg(
                latitudes[time_index - 1],
                longitudes[time_index - 1],
                target_lat,
                target_lon,
            )
            cross_track_m = 0.0
            segment_heading_deg = desired_heading_deg
        heading_error_deg = wrap_angle_deg(desired_heading_deg - heading_deg)
        lateral_path_error_m[time_index] = abs(float(cross_track_m))
        heading_error_deg_series[time_index] = abs(float(wrap_angle_deg(segment_heading_deg - heading_deg)))
        turn_rate_bank_deg_s = math.degrees(
            9.80665 * math.tan(math.radians(config.airborne_bank_angle_deg)) / max(speed_mps, 1.0)
        )
        turn_rate_radius_deg_s = math.degrees(speed_mps / max(config.airborne_min_turn_radius_m, 1.0))
        max_turn_deg = min(
            config.airborne_max_turn_rate_deg_s,
            turn_rate_bank_deg_s,
            turn_rate_radius_deg_s if speed_mps > 0 else config.airborne_max_turn_rate_deg_s,
        ) * dt
        heading_deg += float(np.clip(heading_error_deg, -max_turn_deg, max_turn_deg))

        remaining_distance_m = great_circle_distance_m(
            latitudes[time_index - 1],
            longitudes[time_index - 1],
            float(waypoint_lats_deg[-1]),
            float(waypoint_lons_deg[-1]),
        )
        braking_speed_mps = math.sqrt(max(0.0, 2.0 * config.airborne_max_decel_mps2 * remaining_distance_m))
        desired_speed_mps = min(target_speed_mps, braking_speed_mps if braking_speed_mps > 0 else target_speed_mps)
        speed_error_mps = desired_speed_mps - speed_mps
        if speed_error_mps >= 0.0:
            speed_step = min(speed_error_mps, config.airborne_max_accel_mps2 * dt)
        else:
            speed_step = max(speed_error_mps, -config.airborne_max_decel_mps2 * dt)
        speed_mps += speed_step

        altitude_error_m = target_alt - altitudes[time_index - 1]
        if altitude_error_m >= 0.0:
            altitude_step = min(altitude_error_m, config.airborne_max_climb_rate_mps * dt)
        else:
            altitude_step = max(altitude_error_m, -config.airborne_max_descent_rate_mps * dt)
        altitudes[time_index] = altitudes[time_index - 1] + altitude_step

        if speed_mps > 0.0:
            step_distance_m = speed_mps * dt
            next_lat, next_lon = destination_point(
                latitudes[time_index - 1],
                longitudes[time_index - 1],
                heading_deg,
                step_distance_m,
            )
        else:
            next_lat, next_lon = latitudes[time_index - 1], longitudes[time_index - 1]
        latitudes[time_index] = next_lat
        longitudes[time_index] = next_lon

    return latitudes, longitudes, altitudes, {
        "lateral_path_error_m": lateral_path_error_m,
        "heading_error_deg": heading_error_deg_series,
        "turn_anticipation_distance_m": turn_anticipation_distance_m,
        "turn_switch_event": turn_switch_event,
    }


def generate_user_waypoint_track(
    config: SimulationConfig,
    user_idx: int,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start_lat = float(config.user_latitudes_deg[user_idx])
    start_lon = float(config.user_longitudes_deg[user_idx])
    start_alt = float(config.user_altitudes_m[user_idx])
    target_alt = float(config.user_target_altitudes_m[user_idx])
    speed_mps = float(config.user_speeds_kmph[user_idx]) / 3.6
    heading_deg = float(config.user_headings_deg[user_idx])

    if config.user_waypoints_deg:
        waypoint_track = np.asarray(config.user_waypoints_deg[user_idx], dtype=float)
        if config.user_waypoint_times_s:
            waypoint_times_s = np.asarray(config.user_waypoint_times_s[user_idx], dtype=float)
        else:
            waypoint_times_s = np.linspace(0.0, duration_s, waypoint_track.shape[0])
        if waypoint_track.shape[1] >= 3:
            waypoint_alts = waypoint_track[:, 2]
        else:
            waypoint_alts = np.linspace(start_alt, target_alt, waypoint_track.shape[0])
        return waypoint_times_s, waypoint_track[:, 0], waypoint_track[:, 1], waypoint_alts

    if config.user_mobility_mode == "static":
        return (
            np.asarray([0.0, duration_s], dtype=float),
            np.asarray([start_lat, start_lat], dtype=float),
            np.asarray([start_lon, start_lon], dtype=float),
            np.asarray([start_alt, target_alt], dtype=float),
        )

    if config.user_mobility_mode == "linear":
        total_distance_m = speed_mps * duration_s
        end_lat, end_lon = destination_point(start_lat, start_lon, heading_deg, total_distance_m)
        return (
            np.asarray([0.0, duration_s], dtype=float),
            np.asarray([start_lat, end_lat], dtype=float),
            np.asarray([start_lon, end_lon], dtype=float),
            np.asarray([start_alt, target_alt], dtype=float),
        )

    if config.user_mobility_mode == "road":
        waypoint_times = [0.0]
        waypoint_lats = [start_lat]
        waypoint_lons = [start_lon]
        waypoint_alts = [start_alt]
        current_lat, current_lon = start_lat, start_lon
        current_heading = heading_deg
        previous_fraction = 0.0
        for segment_index, segment_fraction in enumerate(config.road_segment_fraction):
            segment_duration = duration_s * (segment_fraction - previous_fraction)
            segment_distance = speed_mps * segment_duration
            current_lat, current_lon = destination_point(
                current_lat,
                current_lon,
                current_heading,
                segment_distance,
            )
            waypoint_times.append(duration_s * segment_fraction)
            waypoint_lats.append(current_lat)
            waypoint_lons.append(current_lon)
            waypoint_alts.append(start_alt + (target_alt - start_alt) * segment_fraction)
            previous_fraction = segment_fraction
            if segment_index < len(config.road_turn_pattern_deg):
                current_heading += float(config.road_turn_pattern_deg[segment_index])
        return (
            np.asarray(waypoint_times, dtype=float),
            np.asarray(waypoint_lats, dtype=float),
            np.asarray(waypoint_lons, dtype=float),
            np.asarray(waypoint_alts, dtype=float),
        )

    total_distance_m = speed_mps * duration_s if speed_mps > 0 else config.flight_corridor_length_km * 1000.0
    half_distance_m = total_distance_m / 2.0
    bend_sign = -1.0 if user_idx % 2 else 1.0
    mid_lat, mid_lon = destination_point(
        start_lat,
        start_lon,
        heading_deg + bend_sign * config.flight_corridor_bend_deg,
        half_distance_m,
    )
    end_lat, end_lon = destination_point(start_lat, start_lon, heading_deg, total_distance_m)
    climb_time = duration_s * config.flight_climb_fraction
    descent_time = duration_s * config.flight_descent_fraction
    cruise_end = duration_s - descent_time
    return (
        np.asarray([0.0, climb_time, cruise_end, duration_s], dtype=float),
        np.asarray([start_lat, mid_lat, mid_lat, end_lat], dtype=float),
        np.asarray([start_lon, mid_lon, mid_lon, end_lon], dtype=float),
        np.asarray([start_alt, target_alt, target_alt, start_alt], dtype=float),
    )

def compute_user_state(config: SimulationConfig, times_s: np.ndarray) -> dict[str, np.ndarray]:
    latitudes_deg = np.zeros((times_s.size, config.num_users), dtype=float)
    longitudes_deg = np.zeros((times_s.size, config.num_users), dtype=float)
    altitudes_m = np.zeros((times_s.size, config.num_users), dtype=float)
    lateral_path_error_m = np.zeros((times_s.size, config.num_users), dtype=float)
    heading_error_deg = np.zeros((times_s.size, config.num_users), dtype=float)
    turn_anticipation_distance_m = np.zeros((times_s.size, config.num_users), dtype=float)
    turn_switch_event = np.zeros((times_s.size, config.num_users), dtype=bool)
    duration_s = float(times_s[-1]) if times_s.size > 1 else float(config.duration_s)
    for user_idx in range(config.num_users):
        waypoint_times, waypoint_lats, waypoint_lons, waypoint_alts = generate_user_waypoint_track(
            config=config,
            user_idx=user_idx,
            duration_s=duration_s,
        )
        is_airborne = max(np.max(waypoint_alts), float(config.user_target_altitudes_m[user_idx])) >= config.airborne_altitude_threshold_m
        if config.airborne_dynamic_enabled and is_airborne and config.user_mobility_mode in ("linear", "waypoint", "flight_corridor"):
            track_lat, track_lon, track_alt, diagnostics = propagate_airborne_track(
                config=config,
                times_s=times_s,
                initial_heading_deg=float(config.user_headings_deg[user_idx]),
                waypoint_times_s=waypoint_times,
                waypoint_lats_deg=waypoint_lats,
                waypoint_lons_deg=waypoint_lons,
                waypoint_alts_m=waypoint_alts,
                initial_speed_mps=float(config.user_initial_speeds_kmph[user_idx]) / 3.6,
                target_speed_mps=float(config.user_speeds_kmph[user_idx]) / 3.6,
            )
            lateral_path_error_m[:, user_idx] = diagnostics["lateral_path_error_m"]
            heading_error_deg[:, user_idx] = diagnostics["heading_error_deg"]
            turn_anticipation_distance_m[:, user_idx] = diagnostics["turn_anticipation_distance_m"]
            turn_switch_event[:, user_idx] = diagnostics["turn_switch_event"]
        else:
            track_lat, track_lon, track_alt = interpolate_waypoint_track(
                times_s=times_s,
                waypoint_times_s=waypoint_times,
                waypoint_lats_deg=waypoint_lats,
                waypoint_lons_deg=waypoint_lons,
                waypoint_alts_m=waypoint_alts,
            )
        latitudes_deg[:, user_idx] = track_lat
        longitudes_deg[:, user_idx] = track_lon
        altitudes_m[:, user_idx] = track_alt

    latitudes_rad = np.radians(latitudes_deg)
    longitudes_rad = np.radians(longitudes_deg)
    rotation = EARTH_ROTATION_RAD_S * times_s[:, None] if config.earth_rotation_enabled else 0.0
    longitude_t = longitudes_rad + rotation
    cos_lat = np.cos(latitudes_rad)
    sin_lat = np.sin(latitudes_rad)
    if config.use_wgs84_earth:
        prime_vertical = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.square(sin_lat))
        x0 = (prime_vertical + altitudes_m) * cos_lat
        z0 = (prime_vertical * (1.0 - WGS84_E2) + altitudes_m) * sin_lat
    else:
        x0 = (EARTH_RADIUS_M + altitudes_m) * cos_lat
        z0 = (EARTH_RADIUS_M + altitudes_m) * sin_lat
    x = x0 * np.cos(longitude_t)
    y = x0 * np.sin(longitude_t)
    z = z0
    platform_type = np.where(altitudes_m >= config.airborne_altitude_threshold_m, "airborne", "ground")
    return {
        "positions_m": np.stack([x, y, z], axis=-1),
        "latitudes_deg": latitudes_deg,
        "longitudes_deg": longitudes_deg,
        "altitudes_m": altitudes_m,
        "lateral_path_error_m": lateral_path_error_m,
        "heading_error_deg": heading_error_deg,
        "turn_anticipation_distance_m": turn_anticipation_distance_m,
        "turn_switch_event": turn_switch_event,
        "platform_type": platform_type,
    }


def compute_user_positions(config: SimulationConfig, times_s: np.ndarray) -> np.ndarray:
    return compute_user_state(config=config, times_s=times_s)["positions_m"]


def compute_link_state(
    config: SimulationConfig,
    sat_positions_m: np.ndarray,
    user_positions_m: np.ndarray,
) -> dict[str, np.ndarray]:
    los_vectors = sat_positions_m[:, None, :, :] - user_positions_m[:, :, None, :]
    ranges_m = np.linalg.norm(los_vectors, axis=-1)
    user_normals = user_positions_m / np.linalg.norm(user_positions_m, axis=-1, keepdims=True)
    elevation_sin = np.sum(los_vectors * user_normals[:, :, None, :], axis=-1) / ranges_m
    elevation_deg = np.degrees(np.arcsin(np.clip(elevation_sin, -1.0, 1.0)))
    visible = elevation_deg >= config.elevation_mask_deg

    fspl_db = 20.0 * np.log10(4.0 * math.pi * ranges_m * config.carrier_frequency_hz / LIGHT_SPEED_MPS)
    rx_power_full_dbm = config.eirp_dbm - fspl_db
    if config.shadowing_sigma_db > 0.0:
        rng = np.random.default_rng(config.random_seed)
        rx_power_full_dbm += rng.normal(
            loc=0.0,
            scale=config.shadowing_sigma_db,
            size=rx_power_full_dbm.shape,
        )

    full_noise_dbm = (
        BOLTZMANN_DBM_PER_HZ
        + 10.0 * np.log10(config.total_bandwidth_hz)
        + config.noise_figure_db
    )
    snr_fullband_db = rx_power_full_dbm - full_noise_dbm
    spectral_efficiency = np.clip(
        np.log2(1.0 + db_to_linear(np.clip(snr_fullband_db, -20.0, 30.0))),
        0.0,
        6.0,
    )
    link_capacity_bps = config.total_bandwidth_hz * spectral_efficiency
    propagation_delay_ms = ranges_m / LIGHT_SPEED_MPS * 1e3
    sat_to_user_unit_vectors = (
        user_positions_m[:, :, None, :] - sat_positions_m[:, None, :, :]
    ) / np.maximum(ranges_m[..., None], 1e-12)

    rx_power_full_dbm = np.where(visible, rx_power_full_dbm, -np.inf)
    snr_fullband_db = np.where(visible, snr_fullband_db, -np.inf)
    link_capacity_bps = np.where(visible, link_capacity_bps, 0.0)
    propagation_delay_ms = np.where(visible, propagation_delay_ms, np.nan)

    return {
        "visible": visible,
        "elevation_deg": elevation_deg,
        "rx_power_full_dbm": rx_power_full_dbm,
        "snr_fullband_db": snr_fullband_db,
        "link_capacity_bps": link_capacity_bps,
        "propagation_delay_ms": propagation_delay_ms,
        "sat_to_user_unit_vectors": sat_to_user_unit_vectors,
    }


def select_satellite(
    access_policy: str,
    current_sat: int,
    snr_db_row: np.ndarray,
    visible_row: np.ndarray,
    hysteresis_db: float,
) -> int:
    visible_indices = np.flatnonzero(visible_row)
    if visible_indices.size == 0:
        return -1

    best_sat = int(visible_indices[np.argmax(snr_db_row[visible_indices])])
    if access_policy == "greedy":
        return best_sat

    if current_sat < 0 or not visible_row[current_sat]:
        return best_sat

    current_snr = float(snr_db_row[current_sat])
    best_snr = float(snr_db_row[best_sat])
    if best_snr >= current_snr + hysteresis_db:
        return best_sat
    return current_sat


def jain_fairness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    denominator = values.size * np.square(values).sum()
    if denominator <= 0.0:
        return 0.0
    return float(np.square(values.sum()) / denominator)


def compute_control_objective(
    resource_policy: str,
    rate_bps: float,
    demand_bits: float,
    average_rate_bps: float,
    config: SimulationConfig,
) -> float:
    rate_mbps = max(rate_bps, 0.0) / 1e6
    demand_mbits = max(demand_bits, 0.0) / 1e6
    average_rate_mbps = max(average_rate_bps, config.pf_rate_floor_mbps * 1e6) / 1e6
    if resource_policy == "equal":
        return rate_mbps
    if resource_policy == "proportional_fair":
        return rate_mbps / average_rate_mbps
    return (demand_mbits + config.lyapunov_control_weight) * rate_mbps


def compute_joint_service_reward(
    config: SimulationConfig,
    resource_policy: str,
    demand_bits: float,
    average_rate_bps: float,
) -> float:
    baseline_rate_bps = config.joint_min_service_mbps * 1e6
    return config.joint_unserved_penalty_scale * compute_control_objective(
        resource_policy=resource_policy,
        rate_bps=baseline_rate_bps,
        demand_bits=demand_bits,
        average_rate_bps=average_rate_bps,
        config=config,
    )


def _beam_selection_scores(
    config: SimulationConfig,
    resource_policy: str,
    demand_bits: np.ndarray,
    average_rate_bps: np.ndarray,
    base_link_capacity_bps: np.ndarray,
) -> np.ndarray:
    scores = np.zeros_like(base_link_capacity_bps, dtype=float)
    for index in range(base_link_capacity_bps.size):
        scores[index] = compute_control_objective(
            resource_policy=resource_policy,
            rate_bps=float(base_link_capacity_bps[index]),
            demand_bits=float(demand_bits[index]),
            average_rate_bps=float(average_rate_bps[index]),
            config=config,
        )
    return scores


def _resource_weights(
    config: SimulationConfig,
    resource_policy: str,
    demand_bits: np.ndarray,
    average_rate_bps: np.ndarray,
    base_link_capacity_bps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if resource_policy == "equal":
        size = demand_bits.size
        return np.ones(size, dtype=float), np.ones(size, dtype=float)

    base_objective = _beam_selection_scores(
        config=config,
        resource_policy=resource_policy,
        demand_bits=demand_bits,
        average_rate_bps=average_rate_bps,
        base_link_capacity_bps=base_link_capacity_bps,
    )
    bandwidth_weights = np.maximum(base_objective, 1e-6)
    if resource_policy == "proportional_fair":
        power_weights = np.maximum(np.sqrt(base_objective), 1e-6)
    else:
        power_weights = np.maximum(
            np.sqrt(demand_bits / 1e6 + config.lyapunov_control_weight),
            1e-6,
        )
    return bandwidth_weights, power_weights


def beam_pattern_attenuation_db(
    config: SimulationConfig,
    boresight_vectors: np.ndarray,
    target_vectors: np.ndarray,
) -> np.ndarray:
    normalized_boresight = boresight_vectors / np.maximum(
        np.linalg.norm(boresight_vectors, axis=1, keepdims=True),
        1e-12,
    )
    normalized_targets = target_vectors / np.maximum(
        np.linalg.norm(target_vectors, axis=1, keepdims=True),
        1e-12,
    )
    cosine = np.clip(normalized_boresight @ normalized_targets.T, -1.0, 1.0)
    off_axis_angle_deg = np.degrees(np.arccos(cosine))
    attenuation_db = np.minimum(
        12.0 * np.square(off_axis_angle_deg / config.beam_3db_width_deg),
        config.beam_max_attenuation_db,
    )
    if attenuation_db.shape[0] == attenuation_db.shape[1]:
        attenuation_db = attenuation_db.copy()
        np.fill_diagonal(attenuation_db, 0.0)
    return attenuation_db


def allocate_satellite_resources(
    config: SimulationConfig,
    resource_policy: str,
    user_indices: np.ndarray,
    backlog_bits: np.ndarray,
    offered_load_bps: np.ndarray,
    average_rate_bps: np.ndarray,
    base_link_capacity_bps: np.ndarray,
    rx_power_full_dbm: np.ndarray,
    look_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    user_indices = np.asarray(user_indices, dtype=int)
    num_candidates = user_indices.size
    allocations = {
        "user_indices": user_indices,
        "beam_granted": np.zeros(num_candidates, dtype=bool),
        "beam_score": np.zeros(num_candidates, dtype=float),
        "allocated_bandwidth_hz": np.zeros(num_candidates, dtype=float),
        "allocated_power_share": np.zeros(num_candidates, dtype=float),
        "allocated_eirp_dbm": np.full(num_candidates, np.nan, dtype=float),
        "pre_interference_snr_db": np.full(num_candidates, np.nan, dtype=float),
        "assigned_sinr_db": np.full(num_candidates, np.nan, dtype=float),
        "interference_dbm": np.full(num_candidates, np.nan, dtype=float),
        "sinr_loss_db": np.zeros(num_candidates, dtype=float),
        "service_capacity_bps": np.zeros(num_candidates, dtype=float),
    }
    if num_candidates == 0:
        return allocations

    demand_bits = backlog_bits[user_indices] + offered_load_bps[user_indices] * config.dt_s
    candidate_average_rates = average_rate_bps[user_indices]
    scores = _beam_selection_scores(
        config=config,
        resource_policy=resource_policy,
        demand_bits=demand_bits,
        average_rate_bps=candidate_average_rates,
        base_link_capacity_bps=base_link_capacity_bps,
    )
    active_count = min(config.max_beams_per_satellite, num_candidates)
    active_local_indices = np.argsort(scores)[::-1][:active_count]
    active_rx_power_dbm = rx_power_full_dbm[active_local_indices]
    active_demand_bits = demand_bits[active_local_indices]
    active_average_rates = candidate_average_rates[active_local_indices]
    active_base_capacity_bps = base_link_capacity_bps[active_local_indices]
    active_look_vectors = look_vectors[active_local_indices]

    bandwidth_weights, power_weights = _resource_weights(
        config=config,
        resource_policy=resource_policy,
        demand_bits=active_demand_bits,
        average_rate_bps=active_average_rates,
        base_link_capacity_bps=active_base_capacity_bps,
    )
    bandwidth_weights = bandwidth_weights / bandwidth_weights.sum()
    power_weights = power_weights / power_weights.sum()

    allocated_bandwidth_hz = config.total_bandwidth_hz * bandwidth_weights
    allocated_power_share = power_weights
    desired_signal_mw = dbm_to_mw(active_rx_power_dbm) * allocated_power_share
    off_axis_attenuation_db = beam_pattern_attenuation_db(
        config=config,
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
        + linear_to_db(allocated_power_share)[None, :]
        - off_axis_attenuation_db
    )
    interference_mw_matrix = dbm_to_mw(interference_dbm_matrix) * bandwidth_overlap.T
    np.fill_diagonal(interference_mw_matrix, 0.0)
    interference_mw = interference_mw_matrix.sum(axis=1)
    noise_dbm = (
        BOLTZMANN_DBM_PER_HZ
        + 10.0 * np.log10(allocated_bandwidth_hz)
        + config.noise_figure_db
    )
    noise_mw = dbm_to_mw(noise_dbm)
    pre_interference_snr_linear = desired_signal_mw / noise_mw
    assigned_sinr_linear = desired_signal_mw / (noise_mw + interference_mw)
    pre_interference_snr_db = linear_to_db(pre_interference_snr_linear)
    assigned_sinr_db = linear_to_db(assigned_sinr_linear)
    spectral_efficiency = np.clip(
        np.log2(1.0 + assigned_sinr_linear),
        0.0,
        6.0,
    )
    service_capacity_bps = allocated_bandwidth_hz * spectral_efficiency
    sinr_loss_db = pre_interference_snr_db - assigned_sinr_db

    allocations["beam_granted"][active_local_indices] = True
    allocations["beam_score"] = scores
    allocations["allocated_bandwidth_hz"][active_local_indices] = allocated_bandwidth_hz
    allocations["allocated_power_share"][active_local_indices] = allocated_power_share
    allocations["allocated_eirp_dbm"][active_local_indices] = config.eirp_dbm + linear_to_db(
        allocated_power_share
    )
    allocations["pre_interference_snr_db"][active_local_indices] = pre_interference_snr_db
    allocations["assigned_sinr_db"][active_local_indices] = assigned_sinr_db
    allocations["interference_dbm"][active_local_indices] = mw_to_dbm(interference_mw)
    allocations["sinr_loss_db"][active_local_indices] = sinr_loss_db
    allocations["service_capacity_bps"][active_local_indices] = service_capacity_bps
    return allocations


def enumerate_joint_configurations(
    config: SimulationConfig,
    resource_policy: str,
    slot_index: int,
    sat_idx: int,
    candidate_users: np.ndarray,
    predicted_backlog_bits: np.ndarray,
    offered_load_bps: np.ndarray,
    average_rate_bps: np.ndarray,
    base_link_capacity_t: np.ndarray,
    rx_power_t: np.ndarray,
    look_vectors_t: np.ndarray,
) -> list[dict[str, Any]]:
    candidate_users = np.asarray(candidate_users, dtype=int)
    configs: list[dict[str, Any]] = []
    if candidate_users.size == 0:
        return configs

    max_size = min(config.max_beams_per_satellite, candidate_users.size)
    min_service_bps = config.joint_min_service_mbps * 1e6
    for size in range(1, max_size + 1):
        for combo in combinations(candidate_users.tolist(), size):
            combo_users = np.asarray(combo, dtype=int)
            allocations = allocate_satellite_resources(
                config=config,
                resource_policy=resource_policy,
                user_indices=combo_users,
                backlog_bits=predicted_backlog_bits,
                offered_load_bps=offered_load_bps,
                average_rate_bps=average_rate_bps,
                base_link_capacity_bps=base_link_capacity_t[combo_users, sat_idx],
                rx_power_full_dbm=rx_power_t[combo_users, sat_idx],
                look_vectors=look_vectors_t[combo_users, sat_idx],
            )
            if np.any(allocations["service_capacity_bps"] < min_service_bps):
                continue
            utility = 0.0
            for local_index, user_idx in enumerate(combo_users):
                rate_bps = float(allocations["service_capacity_bps"][local_index])
                demand_bits = predicted_backlog_bits[user_idx] + offered_load_bps[user_idx] * config.dt_s
                utility += compute_control_objective(
                    resource_policy=resource_policy,
                    rate_bps=rate_bps,
                    demand_bits=float(demand_bits),
                    average_rate_bps=float(average_rate_bps[user_idx]),
                    config=config,
                )
                utility += compute_joint_service_reward(
                    config=config,
                    resource_policy=resource_policy,
                    demand_bits=float(demand_bits),
                    average_rate_bps=float(average_rate_bps[user_idx]),
                )
            configs.append(
                {
                    "slot_index": slot_index,
                    "sat_idx": sat_idx,
                    "users": tuple(int(v) for v in combo_users),
                    "utility": float(utility),
                }
            )
    return configs


def joint_select_assignments(
    config: SimulationConfig,
    resource_policy: str,
    current_assignments: np.ndarray,
    visible_horizon: np.ndarray,
    backlog_bits: np.ndarray,
    offered_load_bps: np.ndarray,
    average_rate_bps: np.ndarray,
    base_link_capacity_horizon: np.ndarray,
    rx_power_horizon: np.ndarray,
    look_vectors_horizon: np.ndarray,
) -> np.ndarray:
    horizon = min(config.mpc_horizon_steps, visible_horizon.shape[0])
    num_users, num_sats = visible_horizon.shape[1], visible_horizon.shape[2]
    configs: list[dict[str, Any]] = []

    for slot_index in range(horizon):
        predicted_backlog_bits = backlog_bits + offered_load_bps * config.dt_s * slot_index
        visible_t = visible_horizon[slot_index]
        base_link_capacity_t = base_link_capacity_horizon[slot_index]
        rx_power_t = rx_power_horizon[slot_index]
        look_vectors_t = look_vectors_horizon[slot_index]
        slot_discount = config.mpc_discount**slot_index

        for sat_idx in range(num_sats):
            candidate_users = np.flatnonzero(visible_t[:, sat_idx])
            sat_configs = enumerate_joint_configurations(
                config=config,
                resource_policy=resource_policy,
                slot_index=slot_index,
                sat_idx=int(sat_idx),
                candidate_users=candidate_users,
                predicted_backlog_bits=predicted_backlog_bits,
                offered_load_bps=offered_load_bps,
                average_rate_bps=average_rate_bps,
                base_link_capacity_t=base_link_capacity_t,
                rx_power_t=rx_power_t,
                look_vectors_t=look_vectors_t,
            )
            for config_item in sat_configs:
                penalty = 0.0
                if slot_index == 0:
                    for user_idx in config_item["users"]:
                        current_sat = int(current_assignments[user_idx])
                        if current_sat >= 0 and current_sat != sat_idx and visible_t[user_idx, current_sat]:
                            demand_bits = predicted_backlog_bits[user_idx] + offered_load_bps[user_idx] * config.dt_s
                            penalty += config.joint_switch_penalty * compute_control_objective(
                                resource_policy=resource_policy,
                                rate_bps=float(offered_load_bps[user_idx]),
                                demand_bits=float(demand_bits),
                                average_rate_bps=float(average_rate_bps[user_idx]),
                                config=config,
                            )
                config_item["utility"] = slot_discount * (config_item["utility"] - penalty)
                configs.append(config_item)

    assignments = np.full(num_users, -1, dtype=int)
    if not configs:
        return assignments

    num_configs = len(configs)
    objective = -np.asarray([config_item["utility"] for config_item in configs], dtype=float)
    integrality = np.ones(num_configs, dtype=int)
    lower_bounds = np.zeros(num_configs, dtype=float)
    upper_bounds = np.ones(num_configs, dtype=float)
    rows = []
    rhs = []

    for slot_index in range(horizon):
        for sat_idx in range(num_sats):
            rows.append(
                [
                    1.0
                    if config_item["slot_index"] == slot_index and config_item["sat_idx"] == sat_idx
                    else 0.0
                    for config_item in configs
                ]
            )
            rhs.append(1.0)
        for user_idx in range(num_users):
            rows.append(
                [
                    1.0
                    if config_item["slot_index"] == slot_index and user_idx in config_item["users"]
                    else 0.0
                    for config_item in configs
                ]
            )
            rhs.append(1.0)

    constraints = LinearConstraint(np.asarray(rows, dtype=float), -np.inf, np.asarray(rhs, dtype=float))
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Joint ILP failed: {result.message}")

    chosen = np.flatnonzero(result.x > 0.5)
    for config_index in chosen:
        config_item = configs[int(config_index)]
        if config_item["slot_index"] == 0:
            for user_idx in config_item["users"]:
                assignments[user_idx] = int(config_item["sat_idx"])
    return assignments


def select_assignments_for_slot(
    config: SimulationConfig,
    access_policy: str,
    resource_policy: str,
    current_assignments: np.ndarray,
    visible_t: np.ndarray,
    snr_t: np.ndarray,
    backlog_bits: np.ndarray,
    offered_load_bps: np.ndarray,
    average_rate_bps: np.ndarray,
    base_link_capacity_t: np.ndarray,
    rx_power_t: np.ndarray,
    look_vectors_t: np.ndarray,
    visible_horizon: np.ndarray | None = None,
    base_link_capacity_horizon: np.ndarray | None = None,
    rx_power_horizon: np.ndarray | None = None,
    look_vectors_horizon: np.ndarray | None = None,
) -> np.ndarray:
    if access_policy == "joint":
        if (
            visible_horizon is None
            or base_link_capacity_horizon is None
            or rx_power_horizon is None
            or look_vectors_horizon is None
        ):
            raise ValueError("joint access requires horizon tensors")
        return joint_select_assignments(
            config=config,
            resource_policy=resource_policy,
            current_assignments=current_assignments,
            visible_horizon=visible_horizon,
            backlog_bits=backlog_bits,
            offered_load_bps=offered_load_bps,
            average_rate_bps=average_rate_bps,
            base_link_capacity_horizon=base_link_capacity_horizon,
            rx_power_horizon=rx_power_horizon,
            look_vectors_horizon=look_vectors_horizon,
        )

    assignments = np.full(visible_t.shape[0], -1, dtype=int)
    for user_idx in range(visible_t.shape[0]):
        assignments[user_idx] = select_satellite(
            access_policy=access_policy,
            current_sat=int(current_assignments[user_idx]),
            snr_db_row=snr_t[user_idx],
            visible_row=visible_t[user_idx],
            hysteresis_db=config.hysteresis_db,
        )
    return assignments


def simulate_policy(config: SimulationConfig, policy: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized_policy, access_policy, resource_policy = parse_policy_name(policy)

    times_s = build_time_grid(config)
    sat_positions_m = compute_satellite_positions(config, times_s)
    user_state = compute_user_state(config, times_s)
    user_positions_m = user_state["positions_m"]
    link_state = compute_link_state(config, sat_positions_m, user_positions_m)

    num_users = config.num_users
    num_sats = sat_positions_m.shape[1]
    num_steps = len(times_s)
    dt = config.dt_s
    offered_load_bps = config.offered_load_bps
    pf_beta = min(1.0, dt / config.pf_time_constant_s)
    satellite_names, source_labels = get_satellite_metadata(config=config, num_satellites=num_sats)

    attachment_assignments = np.full(num_users, -1, dtype=int)
    handover_timers_s = np.zeros(num_users, dtype=float)
    backlog_bits = np.zeros(num_users, dtype=float)
    total_served_bits = np.zeros(num_users, dtype=float)
    handover_counts = np.zeros(num_users, dtype=int)
    average_rate_bps = np.full(num_users, config.pf_rate_floor_mbps * 1e6, dtype=float)
    total_granted_beams = 0
    records: list[dict[str, Any]] = []
    total_user_time_samples = num_users * num_steps

    visible = link_state["visible"]
    rx_power_full_dbm = link_state["rx_power_full_dbm"]
    snr_fullband_db = link_state["snr_fullband_db"]
    link_capacity_bps = link_state["link_capacity_bps"]
    propagation_delay_ms = link_state["propagation_delay_ms"]
    elevation_deg = link_state["elevation_deg"]
    look_vectors = link_state["sat_to_user_unit_vectors"]
    unique_sources = sorted({str(label) for label in source_labels})
    unique_platforms = sorted({str(label) for label in np.unique(user_state["platform_type"])})
    source_accumulators: dict[str, dict[str, float]] = {
        source_label: {
            "satellite_count": float(np.sum(source_labels == source_label)),
            "visible_samples": 0.0,
            "attached_samples": 0.0,
            "beam_granted_samples": 0.0,
            "served_bits": 0.0,
            "queue_delay_sum": 0.0,
            "queue_delay_count": 0.0,
            "handover_events": 0.0,
            "sinr_loss_sum": 0.0,
            "sinr_loss_count": 0.0,
        }
        for source_label in unique_sources
    }
    platform_time_samples: dict[str, float] = {platform: 0.0 for platform in unique_platforms}
    source_platform_accumulators: dict[str, dict[str, dict[str, float]]] = {
        source_label: {
            platform: {
                "visible_samples": 0.0,
                "attached_samples": 0.0,
                "beam_granted_samples": 0.0,
                "served_bits": 0.0,
                "queue_delay_sum": 0.0,
                "queue_delay_count": 0.0,
                "handover_events": 0.0,
                "sinr_loss_sum": 0.0,
                "sinr_loss_count": 0.0,
            }
            for platform in unique_platforms
        }
        for source_label in unique_sources
    }

    for time_index, time_s in enumerate(times_s):
        visible_t = visible[time_index]
        snr_t = snr_fullband_db[time_index]
        base_link_capacity_t = link_capacity_bps[time_index]
        rx_power_t = rx_power_full_dbm[time_index]
        look_vectors_t = look_vectors[time_index]
        visible_satellites = visible_t.sum(axis=1)
        platform_types_t = user_state["platform_type"][time_index]
        for platform in unique_platforms:
            platform_time_samples[platform] += float(np.sum(platform_types_t == platform))
        for source_label in unique_sources:
            source_mask = source_labels == source_label
            source_visible = np.any(visible_t[:, source_mask], axis=1)
            source_accumulators[source_label]["visible_samples"] += float(np.sum(source_visible))
            for platform in unique_platforms:
                platform_mask = platform_types_t == platform
                source_platform_accumulators[source_label][platform]["visible_samples"] += float(
                    np.sum(source_visible & platform_mask)
                )
        horizon_end = min(time_index + config.mpc_horizon_steps, num_steps)

        backlog_bits += offered_load_bps * dt
        proposed_assignments = select_assignments_for_slot(
            config=config,
            access_policy=access_policy,
            resource_policy=resource_policy,
            current_assignments=attachment_assignments,
            visible_t=visible_t,
            snr_t=snr_t,
            backlog_bits=backlog_bits,
            offered_load_bps=offered_load_bps,
            average_rate_bps=average_rate_bps,
            base_link_capacity_t=base_link_capacity_t,
            rx_power_t=rx_power_t,
            look_vectors_t=look_vectors_t,
            visible_horizon=visible[time_index:horizon_end],
            base_link_capacity_horizon=link_capacity_bps[time_index:horizon_end],
            rx_power_horizon=rx_power_full_dbm[time_index:horizon_end],
            look_vectors_horizon=look_vectors[time_index:horizon_end],
        )

        previous_assignments = attachment_assignments.copy()
        handover_event = np.zeros(num_users, dtype=bool)
        for user_idx in range(num_users):
            previous_sat = int(previous_assignments[user_idx])
            proposed_sat = int(proposed_assignments[user_idx])
            if proposed_sat >= 0 and previous_sat >= 0 and proposed_sat != previous_sat:
                handover_counts[user_idx] += 1
                handover_timers_s[user_idx] = config.handover_penalty_s
                handover_event[user_idx] = True

        next_attachments = attachment_assignments.copy()
        for user_idx in range(num_users):
            proposed_sat = int(proposed_assignments[user_idx])
            previous_sat = int(previous_assignments[user_idx])
            if proposed_sat >= 0:
                next_attachments[user_idx] = proposed_sat
            elif visible_satellites[user_idx] == 0:
                next_attachments[user_idx] = -1
            elif previous_sat >= 0 and visible_t[user_idx, previous_sat]:
                next_attachments[user_idx] = previous_sat
            else:
                next_attachments[user_idx] = -1
        attachment_assignments = next_attachments

        handover_blocked = handover_timers_s > 0.0
        beam_granted = np.zeros(num_users, dtype=bool)
        beam_score = np.zeros(num_users, dtype=float)
        allocated_bandwidth_hz = np.zeros(num_users, dtype=float)
        allocated_power_share = np.zeros(num_users, dtype=float)
        allocated_eirp_dbm = np.full(num_users, np.nan, dtype=float)
        pre_interference_snr_db = np.full(num_users, np.nan, dtype=float)
        assigned_sinr_db = np.full(num_users, np.nan, dtype=float)
        interference_dbm = np.full(num_users, np.nan, dtype=float)
        sinr_loss_db = np.zeros(num_users, dtype=float)
        service_capacity_bps = np.zeros(num_users, dtype=float)
        assigned_delay_ms = np.full(num_users, np.nan, dtype=float)
        assigned_elevation_deg = np.full(num_users, np.nan, dtype=float)
        contention_count = np.zeros(num_users, dtype=int)

        candidate_assignments = proposed_assignments if access_policy == "joint" else attachment_assignments
        for sat_idx in range(num_sats):
            sat_user_indices = np.flatnonzero(candidate_assignments == sat_idx)
            if sat_user_indices.size > 0:
                contention_count[sat_user_indices] = sat_user_indices.size

            eligible_user_indices = np.flatnonzero((candidate_assignments == sat_idx) & (~handover_blocked))
            if eligible_user_indices.size == 0:
                continue

            sat_allocations = allocate_satellite_resources(
                config=config,
                resource_policy=resource_policy,
                user_indices=eligible_user_indices,
                backlog_bits=backlog_bits,
                offered_load_bps=offered_load_bps,
                average_rate_bps=average_rate_bps,
                base_link_capacity_bps=base_link_capacity_t[eligible_user_indices, sat_idx],
                rx_power_full_dbm=rx_power_t[eligible_user_indices, sat_idx],
                look_vectors=look_vectors_t[eligible_user_indices, sat_idx],
            )
            local_indices = sat_allocations["user_indices"]
            beam_granted[local_indices] = sat_allocations["beam_granted"]
            beam_score[local_indices] = sat_allocations["beam_score"]
            allocated_bandwidth_hz[local_indices] = sat_allocations["allocated_bandwidth_hz"]
            allocated_power_share[local_indices] = sat_allocations["allocated_power_share"]
            allocated_eirp_dbm[local_indices] = sat_allocations["allocated_eirp_dbm"]
            pre_interference_snr_db[local_indices] = sat_allocations["pre_interference_snr_db"]
            assigned_sinr_db[local_indices] = sat_allocations["assigned_sinr_db"]
            interference_dbm[local_indices] = sat_allocations["interference_dbm"]
            sinr_loss_db[local_indices] = sat_allocations["sinr_loss_db"]
            service_capacity_bps[local_indices] = sat_allocations["service_capacity_bps"]
            assigned_delay_ms[local_indices] = propagation_delay_ms[time_index, local_indices, sat_idx]
            assigned_elevation_deg[local_indices] = elevation_deg[time_index, local_indices, sat_idx]

        coverage_outage = visible_satellites == 0
        beam_blocked = (visible_satellites > 0) & (~handover_blocked) & (~beam_granted)
        service_blocked = coverage_outage | handover_blocked | (~beam_granted)
        service_bits = np.where(
            service_blocked,
            0.0,
            np.minimum(backlog_bits, service_capacity_bps * dt),
        )
        backlog_bits -= service_bits
        total_served_bits += service_bits
        total_granted_beams += int(beam_granted.sum())

        queue_delay_s = np.divide(
            backlog_bits,
            offered_load_bps,
            out=np.zeros_like(backlog_bits),
            where=offered_load_bps > 0.0,
        )
        served_rate_bps = service_bits / dt
        average_rate_bps = (1.0 - pf_beta) * average_rate_bps + pf_beta * np.maximum(
            served_rate_bps,
            config.pf_rate_floor_mbps * 1e6,
        )
        best_link_bps = base_link_capacity_t.max(axis=1)

        for user_idx in range(num_users):
            if attachment_assignments[user_idx] >= 0:
                source_label = str(source_labels[attachment_assignments[user_idx]])
                platform_type = str(platform_types_t[user_idx])
                source_accumulators[source_label]["attached_samples"] += 1.0
                source_accumulators[source_label]["served_bits"] += float(service_bits[user_idx])
                source_accumulators[source_label]["queue_delay_sum"] += float(queue_delay_s[user_idx])
                source_accumulators[source_label]["queue_delay_count"] += 1.0
                if handover_event[user_idx]:
                    source_accumulators[source_label]["handover_events"] += 1.0
                if beam_granted[user_idx]:
                    source_accumulators[source_label]["beam_granted_samples"] += 1.0
                    source_accumulators[source_label]["sinr_loss_sum"] += float(sinr_loss_db[user_idx])
                    source_accumulators[source_label]["sinr_loss_count"] += 1.0
                platform_accumulator = source_platform_accumulators[source_label][platform_type]
                platform_accumulator["attached_samples"] += 1.0
                platform_accumulator["served_bits"] += float(service_bits[user_idx])
                platform_accumulator["queue_delay_sum"] += float(queue_delay_s[user_idx])
                platform_accumulator["queue_delay_count"] += 1.0
                if handover_event[user_idx]:
                    platform_accumulator["handover_events"] += 1.0
                if beam_granted[user_idx]:
                    platform_accumulator["beam_granted_samples"] += 1.0
                    platform_accumulator["sinr_loss_sum"] += float(sinr_loss_db[user_idx])
                    platform_accumulator["sinr_loss_count"] += 1.0
            records.append(
                {
                    "time_s": float(time_s),
                    "user_id": int(user_idx),
                    "policy": normalized_policy,
                    "access_policy": access_policy,
                    "resource_policy": resource_policy,
                    "satellite_id": int(attachment_assignments[user_idx]),
                    "satellite_name": (
                        str(satellite_names[attachment_assignments[user_idx]])
                        if attachment_assignments[user_idx] >= 0
                        else "none"
                    ),
                    "source_label": (
                        str(source_labels[attachment_assignments[user_idx]])
                        if attachment_assignments[user_idx] >= 0
                        else "none"
                    ),
                    "scheduled_satellite_id": int(proposed_assignments[user_idx]),
                    "contention_count": int(contention_count[user_idx]),
                    "visible_satellites": int(visible_satellites[user_idx]),
                    "beam_granted": bool(beam_granted[user_idx]),
                    "beam_blocked": bool(beam_blocked[user_idx]),
                    "beam_score": float(beam_score[user_idx]),
                    "allocated_bandwidth_mhz": float(allocated_bandwidth_hz[user_idx] / 1e6),
                    "allocated_power_share": float(allocated_power_share[user_idx]),
                    "allocated_eirp_dbm": float(allocated_eirp_dbm[user_idx]) if not np.isnan(allocated_eirp_dbm[user_idx]) else np.nan,
                    "pre_interference_snr_db": float(pre_interference_snr_db[user_idx]) if not np.isnan(pre_interference_snr_db[user_idx]) else np.nan,
                    "assigned_sinr_db": float(assigned_sinr_db[user_idx]) if not np.isnan(assigned_sinr_db[user_idx]) else np.nan,
                    "interference_dbm": float(interference_dbm[user_idx]) if not np.isnan(interference_dbm[user_idx]) else np.nan,
                    "sinr_loss_db": float(sinr_loss_db[user_idx]),
                    "assigned_elevation_deg": float(assigned_elevation_deg[user_idx]) if not np.isnan(assigned_elevation_deg[user_idx]) else np.nan,
                    "assigned_propagation_delay_ms": float(assigned_delay_ms[user_idx]) if not np.isnan(assigned_delay_ms[user_idx]) else np.nan,
                    "best_link_capacity_mbps": float(best_link_bps[user_idx] / 1e6),
                    "service_capacity_mbps": float(service_capacity_bps[user_idx] / 1e6),
                    "served_mbps": float(served_rate_bps[user_idx] / 1e6),
                    "offered_mbps": float(offered_load_bps[user_idx] / 1e6),
                    "average_rate_mbps": float(average_rate_bps[user_idx] / 1e6),
                    "user_latitude_deg": float(user_state["latitudes_deg"][time_index, user_idx]),
                    "user_longitude_deg": float(user_state["longitudes_deg"][time_index, user_idx]),
                    "user_altitude_m": float(user_state["altitudes_m"][time_index, user_idx]),
                    "lateral_path_error_m": float(user_state["lateral_path_error_m"][time_index, user_idx]),
                    "heading_error_deg": float(user_state["heading_error_deg"][time_index, user_idx]),
                    "turn_anticipation_distance_m": float(user_state["turn_anticipation_distance_m"][time_index, user_idx]),
                    "turn_switch_event": bool(user_state["turn_switch_event"][time_index, user_idx]),
                    "platform_type": str(user_state["platform_type"][time_index, user_idx]),
                    "backlog_mbits": float(backlog_bits[user_idx] / 1e6),
                    "queue_delay_s": float(queue_delay_s[user_idx]),
                    "coverage_outage": bool(coverage_outage[user_idx]),
                    "handover_blocked": bool(handover_blocked[user_idx]),
                    "handover_event": bool(handover_event[user_idx]),
                    "handover_count_so_far": int(handover_counts[user_idx]),
                }
            )

        handover_timers_s = np.maximum(0.0, handover_timers_s - dt)

    frame = pd.DataFrame.from_records(records)
    total_offered_bits = offered_load_bps.sum() * config.duration_s
    user_throughput_mbps = total_served_bits / config.duration_s / 1e6
    system_throughput_mbps = user_throughput_mbps.sum()
    delivery_ratio = total_served_bits.sum() / total_offered_bits if total_offered_bits > 0 else 0.0
    granted_rows = frame[frame["beam_granted"]]
    mean_bandwidth_mhz = (
        float(granted_rows["allocated_bandwidth_mhz"].mean()) if not granted_rows.empty else 0.0
    )
    mean_power_share = (
        float(granted_rows["allocated_power_share"].mean()) if not granted_rows.empty else 0.0
    )
    mean_sinr_loss_db = (
        float(granted_rows["sinr_loss_db"].mean()) if not granted_rows.empty else 0.0
    )
    mean_interference_dbm = (
        float(granted_rows["interference_dbm"].mean()) if not granted_rows.empty else float("nan")
    )
    beam_utilization_ratio = total_granted_beams / (
        num_steps * config.num_satellites * config.max_beams_per_satellite
    )
    initial_platforms = user_state["platform_type"][0]
    platform_user_counts = {
        str(platform_type): int(np.sum(initial_platforms == platform_type))
        for platform_type in np.unique(initial_platforms)
    }
    airborne_rows = frame[frame["platform_type"] == "airborne"]
    path_following_stats = {
        "mean_lateral_path_error_m": round(float(airborne_rows["lateral_path_error_m"].mean()), 4) if not airborne_rows.empty else 0.0,
        "p95_lateral_path_error_m": round(float(airborne_rows["lateral_path_error_m"].quantile(0.95)), 4) if not airborne_rows.empty else 0.0,
        "mean_heading_error_deg": round(float(airborne_rows["heading_error_deg"].mean()), 4) if not airborne_rows.empty else 0.0,
        "turn_switch_events": int(airborne_rows["turn_switch_event"].sum()) if not airborne_rows.empty else 0,
        "mean_turn_anticipation_distance_m": round(float(airborne_rows.loc[airborne_rows["turn_switch_event"], "turn_anticipation_distance_m"].mean()), 4) if not airborne_rows.empty and airborne_rows["turn_switch_event"].any() else 0.0,
    }
    source_stats: dict[str, dict[str, Any]] = {}
    for source_label, accumulator in source_accumulators.items():
        sinr_loss_mean = (
            accumulator["sinr_loss_sum"] / accumulator["sinr_loss_count"]
            if accumulator["sinr_loss_count"] > 0
            else 0.0
        )
        queue_delay_mean = (
            accumulator["queue_delay_sum"] / accumulator["queue_delay_count"]
            if accumulator["queue_delay_count"] > 0
            else 0.0
        )
        visibility_share = accumulator["visible_samples"] / total_user_time_samples
        source_stats[source_label] = {
            "satellite_count": int(accumulator["satellite_count"]),
            "visibility_share": round(visibility_share, 4),
            "outage_share": round(1.0 - visibility_share, 4),
            "attachment_share": round(accumulator["attached_samples"] / total_user_time_samples, 4),
            "beam_grant_share": round(accumulator["beam_granted_samples"] / total_user_time_samples, 4),
            "throughput_mbps": round(accumulator["served_bits"] / config.duration_s / 1e6, 4),
            "mean_queue_delay_s": round(queue_delay_mean, 4),
            "handover_events": int(accumulator["handover_events"]),
            "mean_sinr_loss_db": round(sinr_loss_mean, 4),
        }
    source_platform_stats: dict[str, dict[str, dict[str, Any]]] = {}
    for source_label in unique_sources:
        source_platform_stats[source_label] = {}
        for platform in unique_platforms:
            accumulator = source_platform_accumulators[source_label][platform]
            platform_denominator = max(platform_time_samples[platform], 1.0)
            sinr_loss_mean = (
                accumulator["sinr_loss_sum"] / accumulator["sinr_loss_count"]
                if accumulator["sinr_loss_count"] > 0
                else 0.0
            )
            queue_delay_mean = (
                accumulator["queue_delay_sum"] / accumulator["queue_delay_count"]
                if accumulator["queue_delay_count"] > 0
                else 0.0
            )
            visibility_share = accumulator["visible_samples"] / platform_denominator
            source_platform_stats[source_label][platform] = {
                "visibility_share": round(visibility_share, 4),
                "outage_share": round(1.0 - visibility_share, 4),
                "attachment_share": round(accumulator["attached_samples"] / platform_denominator, 4),
                "beam_grant_share": round(accumulator["beam_granted_samples"] / platform_denominator, 4),
                "throughput_mbps": round(accumulator["served_bits"] / config.duration_s / 1e6, 4),
                "mean_queue_delay_s": round(queue_delay_mean, 4),
                "handover_events": int(accumulator["handover_events"]),
                "mean_sinr_loss_db": round(sinr_loss_mean, 4),
            }
    summary = {
        "policy": normalized_policy,
        "access_policy": access_policy,
        "resource_policy": resource_policy,
        "constellation_mode": config.constellation_mode,
        "duration_s": config.duration_s,
        "num_users": num_users,
        "num_satellites": num_sats,
        "max_beams_per_satellite": config.max_beams_per_satellite,
        "system_throughput_mbps": round(float(system_throughput_mbps), 4),
        "mean_user_throughput_mbps": round(float(system_throughput_mbps / num_users), 4),
        "delivery_ratio": round(float(delivery_ratio), 4),
        "avg_queue_delay_s": round(float(frame["queue_delay_s"].mean()), 4),
        "p95_queue_delay_s": round(float(frame["queue_delay_s"].quantile(0.95)), 4),
        "jain_fairness": round(jain_fairness(user_throughput_mbps), 4),
        "coverage_outage_ratio": round(float(frame["coverage_outage"].mean()), 4),
        "handover_block_ratio": round(float(frame["handover_blocked"].mean()), 4),
        "beam_block_ratio": round(float(frame["beam_blocked"].mean()), 4),
        "beam_utilization_ratio": round(float(beam_utilization_ratio), 4),
        "mean_visible_satellites": round(float(frame["visible_satellites"].mean()), 4),
        "mean_allocated_bandwidth_mhz": round(mean_bandwidth_mhz, 4),
        "mean_power_share": round(mean_power_share, 4),
        "mean_sinr_loss_db": round(mean_sinr_loss_db, 4),
        "mean_interference_dbm": round(mean_interference_dbm, 4) if not math.isnan(mean_interference_dbm) else None,
        "total_handovers": int(handover_counts.sum()),
        "per_user_handovers": handover_counts.tolist(),
        "platform_user_counts": platform_user_counts,
        "path_following_stats": path_following_stats,
        "source_stats": source_stats,
        "source_platform_stats": source_platform_stats,
    }
    return frame, summary


def write_markdown_report(
    config: SimulationConfig,
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    report_path = Path(output_dir) / "report.md"
    lines = [
        "# NTN joint scheduling closed loop report",
        "",
        "## Configuration",
        "",
        f"- Constellation mode: {config.constellation_mode}",
        f"- Duration: {config.duration_s}s",
        f"- Time step: {config.dt_s}s",
        f"- Satellites: {summaries[0]['num_satellites'] if summaries else config.num_satellites}",
        f"- Orbit planes: {config.num_orbit_planes}",
        f"- Altitude: {config.altitude_km} km",
        f"- Inclination: {config.inclination_deg:.1f} deg",
        f"- Synthetic J2 enabled: {config.synthetic_j2_enabled}",
        f"- WGS84 Earth enabled: {config.use_wgs84_earth}",
        f"- Carrier frequency: {config.carrier_frequency_hz / 1e9:.1f} GHz",
        f"- Total bandwidth per satellite: {config.total_bandwidth_hz / 1e6:.1f} MHz",
        f"- Total EIRP per satellite: {config.eirp_dbm:.1f} dBm",
        f"- Beam limit per satellite: {config.max_beams_per_satellite}",
        f"- Beam 3 dB width: {config.beam_3db_width_deg:.1f} deg",
        f"- Beam max attenuation: {config.beam_max_attenuation_db:.1f} dB",
        f"- PF time constant: {config.pf_time_constant_s:.1f}s",
        f"- Lyapunov control weight: {config.lyapunov_control_weight:.1f}",
        f"- Joint switch penalty: {config.joint_switch_penalty:.2f}",
        f"- Joint minimum service: {config.joint_min_service_mbps:.1f} Mbps",
        f"- Joint unserved penalty scale: {config.joint_unserved_penalty_scale:.2f}",
        f"- MPC horizon: {config.mpc_horizon_steps} slots",
        f"- MPC discount: {config.mpc_discount:.2f}",
        f"- Earth rotation enabled: {config.earth_rotation_enabled}",
        f"- TLE file: {config.tle_file if config.constellation_mode == 'tle' else 'n/a'}",
        f"- TLE files: {list(config.tle_files) if config.constellation_mode == 'tle' and config.tle_files else []}",
        f"- User mobility mode: {config.user_mobility_mode}",
        f"- Platform user counts: {summaries[0].get('platform_user_counts', {}) if summaries else {}}",
        f"- Path following stats: {summaries[0].get('path_following_stats', {}) if summaries else {}}",
        f"- Users (longitude deg): {list(config.user_longitudes_deg)}",
        f"- Users (latitude deg): {list(config.user_latitudes_deg)}",
        f"- Offered load (Mbps): {list(config.offered_load_mbps)}",
        "",
        "## Policy comparison",
        "",
        "| Policy | Throughput (Mbps) | Delivery ratio | P95 delay (s) | Beam block | Handover block | Fairness | Mean SINR loss (dB) | Handovers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for summary in summaries:
        lines.append(
            "| {policy} | {system_throughput_mbps:.2f} | {delivery_ratio:.3f} | {p95_queue_delay_s:.2f} | {beam_block_ratio:.3f} | {handover_block_ratio:.3f} | {jain_fairness:.3f} | {mean_sinr_loss_db:.2f} | {total_handovers} |".format(
                **summary
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `proportional_fair` uses an exponential moving average of achieved rate and prioritizes large instantaneous-rate / average-rate gain.",
            "- `lyapunov` increases scheduling weight for large queue backlog while still preferring strong service rate.",
            "- `joint_*` policies run a finite-horizon ILP/MPC with minimum-service filtering and an unserved-user penalty, so access and resource management are no longer decoupled.",
            "- `mean_sinr_loss_db` measures the penalty created by simultaneous same-satellite beams.",
            "",
            "## Next iteration",
            "",
            "- Add explicit backlog-state coupling or terminal-cost shaping to the MPC instead of using a lightweight open-loop forecast.",
            "- Add multi-plane or imported TLE geometry so joint scheduling is not limited to one orbital ring.",
            "- Replace the simple beam pattern with measured antenna patterns or beam footprints.",
        ]
    )
    layered_sources = sorted(
        {
            source_label
            for summary in summaries
            for source_label in summary.get("source_stats", {}).keys()
        }
    )
    if layered_sources and any(summary.get("constellation_mode") == "tle" for summary in summaries):
        lines.extend(
            [
                "",
                "## Source Breakdown",
                "",
            ]
        )
        for source_label in layered_sources:
            lines.append(f"### {source_label}")
            lines.append("")
            lines.append("| Policy | Throughput (Mbps) | Attachment Share | Outage Share | Mean Queue Delay (s) | Handover Events | Mean SINR Loss (dB) |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for summary in summaries:
                source_stat = summary.get("source_stats", {}).get(source_label)
                if source_stat is None:
                    continue
                lines.append(
                    "| {policy} | {throughput_mbps:.2f} | {attachment_share:.3f} | {outage_share:.3f} | {mean_queue_delay_s:.2f} | {handover_events} | {mean_sinr_loss_db:.2f} |".format(
                        policy=summary["policy"],
                        **source_stat,
                    )
                )
            lines.append("")
        layered_platforms = sorted(
            {
                platform
                for summary in summaries
                for source_entry in summary.get("source_platform_stats", {}).values()
                for platform in source_entry.keys()
            }
        )
        if layered_platforms:
            lines.extend(
                [
                    "",
                    "## Source x Platform Breakdown",
                    "",
                ]
            )
            for source_label in layered_sources:
                for platform in layered_platforms:
                    lines.append(f"### {source_label} / {platform}")
                    lines.append("")
                    lines.append("| Policy | Throughput (Mbps) | Attachment Share | Outage Share | Mean Queue Delay (s) | Handover Events |")
                    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
                    for summary in summaries:
                        platform_stat = summary.get("source_platform_stats", {}).get(source_label, {}).get(platform)
                        if platform_stat is None:
                            continue
                        lines.append(
                            "| {policy} | {throughput_mbps:.2f} | {attachment_share:.3f} | {outage_share:.3f} | {mean_queue_delay_s:.2f} | {handover_events} |".format(
                                policy=summary["policy"],
                                **platform_stat,
                            )
                        )
                    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def save_plots(
    frames: dict[str, pd.DataFrame],
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths: dict[str, Path] = {}

    policies = [summary["policy"] for summary in summaries]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(policies), 3)))
    throughput = [summary["system_throughput_mbps"] for summary in summaries]
    delay = [summary["p95_queue_delay_s"] for summary in summaries]
    beam_block = [summary["beam_block_ratio"] for summary in summaries]
    fairness = [summary["jain_fairness"] for summary in summaries]
    handovers = [summary["total_handovers"] for summary in summaries]
    sinr_loss = [summary["mean_sinr_loss_db"] for summary in summaries]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("NTN joint scheduling closed loop: policy comparison")
    axes = axes.flatten()

    axes[0].bar(policies, throughput, color=colors[: len(policies)])
    axes[0].set_title("System throughput")
    axes[0].set_ylabel("Mbps")

    axes[1].bar(policies, delay, color=colors[: len(policies)])
    axes[1].set_title("P95 queue delay")
    axes[1].set_ylabel("Seconds")

    axes[2].bar(policies, beam_block, color=colors[: len(policies)])
    axes[2].set_title("Beam block ratio")
    axes[2].set_ylabel("Ratio")

    axes[3].bar(policies, fairness, color=colors[: len(policies)])
    axes[3].set_title("Jain fairness")
    axes[3].set_ylabel("Score")
    axes[3].set_ylim(0.0, 1.05)

    axes[4].bar(policies, handovers, color=colors[: len(policies)])
    axes[4].set_title("Total handovers")
    axes[4].set_ylabel("Count")

    axes[5].bar(policies, sinr_loss, color=colors[: len(policies)])
    axes[5].set_title("Mean SINR loss")
    axes[5].set_ylabel("dB")

    for axis in axes:
        axis.grid(alpha=0.2, linestyle="--")
        axis.tick_params(axis="x", rotation=15)

    comparison_path = output_dir / "policy_comparison.png"
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["comparison"] = comparison_path

    fig, axes = plt.subplots(len(frames), 5, figsize=(22, 4 * len(frames)), sharex=True)
    if len(frames) == 1:
        axes = np.asarray([axes])
    representative_user = 0
    for row_index, policy in enumerate(policies):
        frame = frames[policy]
        user_frame = frame[frame["user_id"] == representative_user].copy()
        user_frame["satellite_id"] = user_frame["satellite_id"].replace(-1, np.nan)
        axes[row_index, 0].plot(user_frame["time_s"], user_frame["satellite_id"], linewidth=1.2)
        axes[row_index, 0].set_title(f"{policy}: user 0 satellite")
        axes[row_index, 0].set_ylabel("Satellite id")
        axes[row_index, 0].grid(alpha=0.2, linestyle="--")

        axes[row_index, 1].plot(user_frame["time_s"], user_frame["allocated_bandwidth_mhz"], linewidth=1.2)
        axes[row_index, 1].set_title(f"{policy}: user 0 bandwidth")
        axes[row_index, 1].set_ylabel("MHz")
        axes[row_index, 1].grid(alpha=0.2, linestyle="--")

        axes[row_index, 2].plot(user_frame["time_s"], user_frame["allocated_power_share"], linewidth=1.2)
        axes[row_index, 2].set_title(f"{policy}: user 0 power share")
        axes[row_index, 2].set_ylabel("Share")
        axes[row_index, 2].grid(alpha=0.2, linestyle="--")

        axes[row_index, 3].plot(user_frame["time_s"], user_frame["assigned_sinr_db"], linewidth=1.2)
        axes[row_index, 3].set_title(f"{policy}: user 0 SINR")
        axes[row_index, 3].set_ylabel("dB")
        axes[row_index, 3].grid(alpha=0.2, linestyle="--")

        axes[row_index, 4].plot(user_frame["time_s"], user_frame["queue_delay_s"], linewidth=1.2)
        axes[row_index, 4].set_title(f"{policy}: user 0 queue delay")
        axes[row_index, 4].set_ylabel("Seconds")
        axes[row_index, 4].grid(alpha=0.2, linestyle="--")

    for column in range(5):
        axes[-1, column].set_xlabel("Time (s)")

    trace_path = output_dir / "user0_trace.png"
    fig.tight_layout()
    fig.savefig(trace_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["trace"] = trace_path

    layered_sources = sorted(
        {
            source_label
            for summary in summaries
            for source_label in summary.get("source_stats", {}).keys()
        }
    )
    if layered_sources and any(summary.get("constellation_mode") == "tle" for summary in summaries):
        metric_specs = [
            ("throughput_mbps", "Throughput (Mbps)"),
            ("attachment_share", "Attachment Share"),
            ("outage_share", "Outage Share"),
            ("mean_queue_delay_s", "Mean Queue Delay (s)"),
            ("handover_events", "Handover Events"),
            ("mean_sinr_loss_db", "Mean SINR Loss (dB)"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()
        source_colors = plt.get_cmap("Set2")(np.linspace(0.0, 1.0, max(len(layered_sources), 3)))
        x = np.arange(len(policies))
        width = 0.8 / max(len(layered_sources), 1)
        for axis_index, (metric_key, metric_label) in enumerate(metric_specs):
            axis = axes[axis_index]
            for source_index, source_label in enumerate(layered_sources):
                values = [
                    summary.get("source_stats", {}).get(source_label, {}).get(metric_key, 0.0)
                    for summary in summaries
                ]
                offset = (source_index - (len(layered_sources) - 1) / 2.0) * width
                axis.bar(
                    x + offset,
                    values,
                    width=width,
                    label=source_label,
                    color=source_colors[source_index],
                )
            axis.set_title(metric_label)
            axis.set_xticks(x)
            axis.set_xticklabels(policies, rotation=15)
            axis.grid(alpha=0.2, linestyle="--")
        axes[0].legend()
        fig.suptitle("Mixed TLE source breakdown")
        source_path = output_dir / "source_breakdown.png"
        fig.tight_layout()
        fig.savefig(source_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths["source_breakdown"] = source_path

        layered_platforms = sorted(
            {
                platform
                for summary in summaries
                for source_entry in summary.get("source_platform_stats", {}).values()
                for platform in source_entry.keys()
            }
        )
        combos = [
            (source_label, platform)
            for source_label in layered_sources
            for platform in layered_platforms
        ]
        combo_metric_specs = [
            ("throughput_mbps", "Throughput (Mbps)"),
            ("attachment_share", "Attachment Share"),
            ("outage_share", "Outage Share"),
            ("mean_queue_delay_s", "Mean Queue Delay (s)"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(18, 8))
        axes = axes.flatten()
        combo_colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(combos), 4)))
        x = np.arange(len(policies))
        width = 0.8 / max(len(combos), 1)
        for axis_index, (metric_key, metric_label) in enumerate(combo_metric_specs):
            axis = axes[axis_index]
            for combo_index, (source_label, platform) in enumerate(combos):
                values = [
                    summary.get("source_platform_stats", {}).get(source_label, {}).get(platform, {}).get(metric_key, 0.0)
                    for summary in summaries
                ]
                offset = (combo_index - (len(combos) - 1) / 2.0) * width
                axis.bar(
                    x + offset,
                    values,
                    width=width,
                    label=f"{source_label}/{platform}",
                    color=combo_colors[combo_index],
                )
            axis.set_title(metric_label)
            axis.set_xticks(x)
            axis.set_xticklabels(policies, rotation=15)
            axis.grid(alpha=0.2, linestyle="--")
        axes[0].legend(fontsize=8)
        fig.suptitle("Mixed TLE source x platform breakdown")
        source_platform_path = output_dir / "source_platform_breakdown.png"
        fig.tight_layout()
        fig.savefig(source_platform_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths["source_platform_breakdown"] = source_platform_path

        fig, axes = plt.subplots(len(frames), 3, figsize=(18, 4 * len(frames)), sharex=True)
        if len(frames) == 1:
            axes = np.asarray([axes])
        combo_colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(combos), 4)))
        for row_index, policy in enumerate(policies):
            frame = frames[policy]
            grouped = frame.groupby(["time_s", "source_label", "platform_type"], as_index=False).agg(
                outage_share=("coverage_outage", "mean"),
                mean_queue_delay_s=("queue_delay_s", "mean"),
                handover_event_rate=("handover_event", "mean"),
                throughput_mbps=("served_mbps", "sum"),
                mean_sinr_db=("assigned_sinr_db", "mean"),
                beam_grant_share=("beam_granted", "mean"),
            )
            for combo_index, (source_label, platform) in enumerate(combos):
                combo_frame = grouped[
                    (grouped["source_label"] == source_label) & (grouped["platform_type"] == platform)
                ]
                label = f"{source_label}/{platform}"
                axes[row_index, 0].plot(
                    combo_frame["time_s"],
                    combo_frame["outage_share"],
                    label=label,
                    color=combo_colors[combo_index],
                    linewidth=1.1,
                )
                axes[row_index, 1].plot(
                    combo_frame["time_s"],
                    combo_frame["mean_queue_delay_s"],
                    label=label,
                    color=combo_colors[combo_index],
                    linewidth=1.1,
                )
                axes[row_index, 2].plot(
                    combo_frame["time_s"],
                    combo_frame["handover_event_rate"],
                    label=label,
                    color=combo_colors[combo_index],
                    linewidth=1.1,
                )
            axes[row_index, 0].set_title(f"{policy}: outage")
            axes[row_index, 0].set_ylabel("Share")
            axes[row_index, 1].set_title(f"{policy}: queue delay")
            axes[row_index, 1].set_ylabel("Seconds")
            axes[row_index, 2].set_title(f"{policy}: handover rate")
            axes[row_index, 2].set_ylabel("Share")
            for axis in axes[row_index]:
                axis.grid(alpha=0.2, linestyle="--")
            axes[row_index, 0].legend(fontsize=8)
        for column in range(3):
            axes[-1, column].set_xlabel("Time (s)")
        fig.suptitle("Mixed TLE source x platform timeseries")
        source_platform_timeseries_path = output_dir / "source_platform_timeseries.png"
        fig.tight_layout()
        fig.savefig(source_platform_timeseries_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths["source_platform_timeseries"] = source_platform_timeseries_path

        fig, axes = plt.subplots(len(frames), 3, figsize=(18, 4 * len(frames)), sharex=True)
        if len(frames) == 1:
            axes = np.asarray([axes])
        for row_index, policy in enumerate(policies):
            frame = frames[policy]
            grouped = frame.groupby(["time_s", "source_label", "platform_type"], as_index=False).agg(
                throughput_mbps=("served_mbps", "sum"),
                mean_sinr_db=("assigned_sinr_db", "mean"),
                beam_grant_share=("beam_granted", "mean"),
            )
            for combo_index, (source_label, platform) in enumerate(combos):
                combo_frame = grouped[
                    (grouped["source_label"] == source_label) & (grouped["platform_type"] == platform)
                ]
                label = f"{source_label}/{platform}"
                axes[row_index, 0].plot(
                    combo_frame["time_s"],
                    combo_frame["throughput_mbps"],
                    label=label,
                    color=combo_colors[combo_index],
                    linewidth=1.1,
                )
                axes[row_index, 1].plot(
                    combo_frame["time_s"],
                    combo_frame["mean_sinr_db"],
                    label=label,
                    color=combo_colors[combo_index],
                    linewidth=1.1,
                )
                axes[row_index, 2].plot(
                    combo_frame["time_s"],
                    combo_frame["beam_grant_share"],
                    label=label,
                    color=combo_colors[combo_index],
                    linewidth=1.1,
                )
            axes[row_index, 0].set_title(f"{policy}: throughput")
            axes[row_index, 0].set_ylabel("Mbps")
            axes[row_index, 1].set_title(f"{policy}: SINR")
            axes[row_index, 1].set_ylabel("dB")
            axes[row_index, 2].set_title(f"{policy}: beam grant")
            axes[row_index, 2].set_ylabel("Share")
            for axis in axes[row_index]:
                axis.grid(alpha=0.2, linestyle="--")
            axes[row_index, 0].legend(fontsize=8)
        for column in range(3):
            axes[-1, column].set_xlabel("Time (s)")
        fig.suptitle("Mixed TLE source x platform service timeseries")
        source_platform_service_timeseries_path = output_dir / "source_platform_service_timeseries.png"
        fig.tight_layout()
        fig.savefig(source_platform_service_timeseries_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths["source_platform_service_timeseries"] = source_platform_service_timeseries_path
    return paths


def run_experiment(
    config: SimulationConfig,
    policies: list[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    for policy in policies:
        frame, summary = simulate_policy(config=config, policy=policy)
        frame_path = output_dir / f"timeseries_{summary['policy']}.csv"
        summary_path = output_dir / f"summary_{summary['policy']}.json"
        frame.to_csv(frame_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        frames[summary["policy"]] = frame
        summaries.append(summary)

    report_path = write_markdown_report(config=config, summaries=summaries, output_dir=output_dir)
    plot_paths = save_plots(frames=frames, summaries=summaries, output_dir=output_dir)

    bundle = {
        "config": config.to_dict(),
        "summaries": summaries,
        "report_path": str(report_path),
        "plot_paths": {name: str(path) for name, path in plot_paths.items()},
    }
    (output_dir / "run_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle
