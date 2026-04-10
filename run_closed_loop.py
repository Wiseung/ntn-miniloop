from __future__ import annotations

import argparse
import json
from pathlib import Path

from ntn_miniloop import SimulationConfig, run_experiment
from ntn_miniloop.core import SUPPORTED_POLICIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an NTN closed loop with PF/Lyapunov scheduling, beam interference, and joint control."
    )
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="Path to a JSON configuration file.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "sticky_equal",
            "sticky_proportional_fair",
            "sticky_lyapunov",
            "joint_proportional_fair",
            "joint_lyapunov",
        ],
        help=f"Policies to run. Supported: {', '.join(SUPPORTED_POLICIES)}",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory used to store CSV, JSON, plots, and markdown outputs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    policies = []
    for policy in args.policies:
        if policy not in SUPPORTED_POLICIES:
            parser.error(f"Unsupported policy '{policy}'. Supported: {', '.join(SUPPORTED_POLICIES)}")
        policies.append(policy)

    config = SimulationConfig.from_json(args.config)
    result = run_experiment(config=config, policies=policies, output_dir=args.output_dir)

    print("NTN joint scheduling closed loop finished.")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
