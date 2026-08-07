"""Download RECCON dataset files from the public GitHub repository.

This module is responsible for downloading the three split files and saving them
into the root data/ directory for subsequent pipeline stages.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests

RECCON_BASE_URL = (
    "https://raw.githubusercontent.com/declare-lab/RECCON/main/data/original_annotation/"
)
RECCON_FILES = [
    "dailydialog_train.json",
    "dailydialog_test.json",
    "dailydialog_valid.json",
]

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(filename: str) -> Path:
    """Download a single RECCON file and write it to the data directory."""
    ensure_data_dir()
    url = urljoin(RECCON_BASE_URL, filename)
    local_path = DATA_DIR / filename

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    local_path.write_bytes(response.content)
    return local_path


def download_all_files(filenames: Iterable[str] = RECCON_FILES) -> list[Path]:
    """Download all predefined RECCON files to the data directory."""
    downloaded_paths = []
    for filename in filenames:
        downloaded_paths.append(download_file(filename))
    return downloaded_paths


def main() -> None:
    """CLI entry point for downloading the dataset."""
    downloaded = download_all_files()
    print("Downloaded files:")
    for path in downloaded:
        print(f"- {path}")


if __name__ == "__main__":
    main()
