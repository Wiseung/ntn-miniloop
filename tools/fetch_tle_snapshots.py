from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import urllib.request


OFFICIAL_URLS = {
    "starlink": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "oneweb": "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle",
}


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch official CelesTrak Starlink/OneWeb TLE snapshots.")
    parser.add_argument(
        "--output-dir",
        default="data/tle/snapshots",
        help="Directory where snapshots will be stored.",
    )
    parser.add_argument(
        "--tag",
        default=str(date.today()),
        help="Snapshot tag used in file names, for example 2026-04-10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for name, url in OFFICIAL_URLS.items():
        content = fetch_text(url)
        target_path = output_dir / f"{name}_{args.tag}.tle"
        target_path.write_text(content, encoding="utf-8")
        written_files.append(target_path)
        line_count = len([line for line in content.splitlines() if line.strip()])
        object_count = line_count // (3 if content.splitlines() and not content.splitlines()[0].startswith("1 ") else 2)
        print(f"{name}: {target_path} ({object_count} objects)")

    combined_path = output_dir / f"starlink_oneweb_{args.tag}.tle"
    combined_payload = "\n".join(path.read_text(encoding="utf-8").strip() for path in written_files) + "\n"
    combined_path.write_text(combined_payload, encoding="utf-8")
    print(f"combined: {combined_path}")


if __name__ == "__main__":
    main()
