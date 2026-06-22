from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_generator import generate_benchmark_file


def _default_paths() -> list[Path]:
    upload_dir = PROJECT_ROOT / "data" / "uploads"
    if not upload_dir.exists():
        return []
    return sorted(upload_dir.glob("*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate semantic CSV intelligence benchmark cases.")
    parser.add_argument("paths", nargs="*", type=Path, help="CSV or Excel files to profile.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "generated_csv_benchmarks.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    paths = args.paths or _default_paths()
    cases = generate_benchmark_file(paths, args.output)
    print("Generated {0} benchmark cases".format(len(cases)))
    print("Output: {0}".format(args.output))


if __name__ == "__main__":
    main()
