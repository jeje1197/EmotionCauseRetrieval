"""Remove derived RECCON artifacts while preserving reusable checkpoints.

By default this keeps the selected datasets and final semantic embedding outputs,
while deleting the raw downloads, formatted intermediates, and VAD-only outputs.
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


def _is_raw_download(path: Path) -> bool:
    return path.name.startswith("dailydialog_") and path.suffix == ".json"


def _is_formatted_dataset(path: Path) -> bool:
    return path.name.startswith("reccon_formatted") and path.suffix == ".json"


def _is_selected_dataset(path: Path) -> bool:
    return path.name.startswith("reccon_selected") and path.suffix == ".json"


def _is_semantic_output(path: Path) -> bool:
    return path.name.startswith("reccon") and path.name.endswith("_embedded_semantic.json")


def _is_vad_output(path: Path) -> bool:
    return path.name.startswith("reccon") and path.name.endswith("_embedded.json") and not _is_semantic_output(path)


def cleanup_dataset(
    data_dir: Path | None = None,
    *,
    keep_selected: bool = True,
    keep_semantic: bool = True,
    keep_vad_embedded: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Delete intermediate JSON artifacts from the data directory.

    Returns the paths that were deleted, or would be deleted in dry-run mode.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    removed: list[Path] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue

        if _is_raw_download(path) or _is_formatted_dataset(path):
            action = "Would delete" if dry_run else "Deleting"
            print(f"{action} intermediate file: {path}")
            if not dry_run:
                path.unlink()
            removed.append(path)
            continue

        if _is_semantic_output(path):
            if keep_semantic:
                continue
            action = "Would delete" if dry_run else "Deleting"
            print(f"{action} semantic output: {path}")
            if not dry_run:
                path.unlink()
            removed.append(path)
            continue

        if _is_selected_dataset(path):
            if keep_selected:
                continue
            action = "Would delete" if dry_run else "Deleting"
            print(f"{action} selected dataset: {path}")
            if not dry_run:
                path.unlink()
            removed.append(path)
            continue

        if _is_vad_output(path):
            if keep_vad_embedded:
                continue
            action = "Would delete" if dry_run else "Deleting"
            print(f"{action} VAD-only output: {path}")
            if not dry_run:
                path.unlink()
            removed.append(path)

    return removed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Remove derived RECCON artifacts while preserving reusable outputs.")
    parser.add_argument("--data-dir", dest="data_dir", default=None, help="Directory containing RECCON JSON artifacts.")
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print files that would be removed without deleting anything.",
    )
    parser.add_argument(
        "--keep-vad-embedded",
        dest="keep_vad_embedded",
        action="store_true",
        help="Preserve VAD-only embedded files such as reccon_train_embedded.json.",
    )
    parser.add_argument(
        "--delete-selected",
        dest="keep_selected",
        action="store_false",
        help="Also delete selected datasets instead of preserving them.",
    )
    parser.add_argument(
        "--delete-semantic",
        dest="keep_semantic",
        action="store_false",
        help="Also delete final semantic embedding outputs instead of preserving them.",
    )
    parser.set_defaults(keep_selected=True, keep_semantic=True)
    args = parser.parse_args()

    removed_paths = cleanup_dataset(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        keep_selected=args.keep_selected,
        keep_semantic=args.keep_semantic,
        keep_vad_embedded=args.keep_vad_embedded,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"Cleanup dry-run complete; {len(removed_paths)} file(s) would be removed.")
    else:
        print(f"Cleanup complete; removed {len(removed_paths)} file(s).")


if __name__ == "__main__":
    main()