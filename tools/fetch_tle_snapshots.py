from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import urllib.request
from urllib.error import URLError, HTTPError


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
    parser.add_argument(
        "--use-latest-tag",
        action="store_true",
        help="Reuse the tag recorded in latest_snapshot.json instead of today's date.",
    )
    parser.add_argument(
        "--print-latest",
        action="store_true",
        help="Print only the current latest snapshot tag from latest_snapshot.json and exit.",
    )
    parser.add_argument(
        "--no-cache-fallback",
        action="store_true",
        help="Fail instead of reusing an existing dated snapshot when network fetch fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_manifest_path = output_dir / "latest_snapshot.json"

    if args.print_latest:
        if not snapshot_manifest_path.exists():
            raise FileNotFoundError(f"Missing latest snapshot manifest: {snapshot_manifest_path}")
        manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        print(manifest["tag"])
        return

    if args.use_latest_tag:
        if not snapshot_manifest_path.exists():
            raise FileNotFoundError(f"Missing latest snapshot manifest: {snapshot_manifest_path}")
        manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        args.tag = manifest["tag"]

    written_files: list[Path] = []
    latest_files: list[Path] = []
    snapshot_manifest: dict[str, str] = {"tag": args.tag}
    for name, url in OFFICIAL_URLS.items():
        target_path = output_dir / f"{name}_{args.tag}.tle"
        try:
            content = fetch_text(url)
            source_mode = "downloaded"
        except (HTTPError, URLError, TimeoutError) as exc:
            if args.no_cache_fallback or not target_path.exists():
                raise
            content = target_path.read_text(encoding="utf-8")
            source_mode = f"cached ({exc})"
        target_path.write_text(content, encoding="utf-8")
        written_files.append(target_path)
        latest_path = output_dir / f"{name}_latest.tle"
        latest_path.write_text(content, encoding="utf-8")
        latest_files.append(latest_path)
        snapshot_manifest[f"{name}_dated"] = target_path.as_posix()
        snapshot_manifest[f"{name}_latest"] = latest_path.as_posix()
        line_count = len([line for line in content.splitlines() if line.strip()])
        object_count = line_count // (3 if content.splitlines() and not content.splitlines()[0].startswith("1 ") else 2)
        print(f"{name}: {target_path} ({object_count} objects, {source_mode})")
        print(f"{name}_latest: {latest_path}")

    combined_path = output_dir / f"starlink_oneweb_{args.tag}.tle"
    combined_payload = "\n".join(path.read_text(encoding="utf-8").strip() for path in written_files) + "\n"
    combined_path.write_text(combined_payload, encoding="utf-8")
    combined_latest_path = output_dir / "starlink_oneweb_latest.tle"
    combined_latest_path.write_text(combined_payload, encoding="utf-8")
    snapshot_manifest["combined_dated"] = combined_path.as_posix()
    snapshot_manifest["combined_latest"] = combined_latest_path.as_posix()
    snapshot_manifest_path.write_text(json.dumps(snapshot_manifest, indent=2), encoding="utf-8")
    print(f"combined: {combined_path}")
    print(f"combined_latest: {combined_latest_path}")
    print(f"manifest: {snapshot_manifest_path}")


if __name__ == "__main__":
    main()
