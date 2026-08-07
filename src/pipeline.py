"""Pipeline orchestrator for RECCON dataset processing stages.

Run this module from the repository root with Python using either:
    python -m src.pipeline --stage all
or:
    python -m src.pipeline --stage download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.scripts.download_dataset import download_all_files
from src.scripts.embed_dataset import embed_dataset
from src.scripts.format_dataset import write_formatted_dataset
from src.scripts.select_turns import write_selected_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RECCON processing pipeline.")
    parser.add_argument(
        "--stage",
        choices=["download", "format", "select", "embed", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries for API calls during embedding.",
    )
    parser.add_argument(
        "--initial-delay",
        type=float,
        default=1.0,
        help="Initial backoff delay in seconds for retries.",
    )
    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=2.0,
        help="Backoff factor for exponential retry delays.",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=5,
        help="Maximum number of concurrent embedding requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_start = time.perf_counter()

    if args.stage in {"download", "all"}:
        print("Running download stage...")
        download_all_files()

    if args.stage in {"format", "all"}:
        print("Running format stage...")
        write_formatted_dataset()

    if args.stage in {"select", "all"}:
        print("Running select stage...")
        write_selected_dataset()

    if args.stage in {"embed", "all"}:
        print("Running embed stage...")
        embed_dataset(
            max_retries=args.max_retries,
            initial_delay=args.initial_delay,
            backoff_factor=args.backoff_factor,
            concurrent_requests=args.concurrent_requests,
        )

    pipeline_duration = time.perf_counter() - pipeline_start
    print(f"Pipeline completed in {pipeline_duration:.2f}s.")


if __name__ == "__main__":
    main()
