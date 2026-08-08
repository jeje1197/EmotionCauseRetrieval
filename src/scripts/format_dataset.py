"""Format raw RECCON dataset JSON files into a flattened conversation list.

This module transforms the original nested JSON format into a list of conversation
objects with explicit conversation_id, target_turn, golden_turn, and enriched turns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

RAW_FILENAMES = [
    "dailydialog_train.json",
    "dailydialog_test.json",
    "dailydialog_valid.json",
]
FORMATTED_FILENAME = "reccon_formatted.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def validate_formatted_dataset(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Formatted dataset file does not exist: {path}")

    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Formatted dataset should be a list of conversations.")

    for conversation in data:
        if not isinstance(conversation, dict):
            raise ValueError("Each conversation must be a dictionary.")
        if not isinstance(conversation.get("conversation_id"), str):
            raise ValueError("Each conversation must include a string conversation_id.")
        if conversation.get("dataset_split") not in {"train", "valid", "test"}:
            raise ValueError(
                "Each conversation must include a dataset_split with value 'train', 'valid', or 'test'."
            )
        if not isinstance(conversation.get("target_turn"), int):
            raise ValueError("Each conversation must include an integer target_turn.")
        if not isinstance(conversation.get("golden_turn"), int):
            raise ValueError("Each conversation must include an integer golden_turn.")

        turns = conversation.get("turns")
        if not isinstance(turns, list):
            raise ValueError("Each conversation must include a turns list.")

        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError("Each turn must be a dictionary.")
            if "semantic_embedding" not in turn or not isinstance(turn["semantic_embedding"], list):
                raise ValueError("Each turn must include a semantic_embedding list.")
            if "vad_scores" not in turn or not isinstance(turn["vad_scores"], list):
                raise ValueError("Each turn must include a vad_scores list.")
            if "has_multiple_causes" not in turn or not isinstance(turn["has_multiple_causes"], bool):
                raise ValueError("Each turn must include a has_multiple_causes boolean.")


def normalize_conversation(
    conversation_id: str,
    conversation_list: list[list[dict[str, Any]]],
    dataset_split: str,
) -> dict[str, Any]:
    """Normalize a single RECCON conversation structure into a flattened dict."""
    if not conversation_list:
        return {
            "conversation_id": conversation_id,
            "dataset_split": dataset_split,
            "target_turn": -1,
            "golden_turn": -1,
            "turns": [],
        }

    # RECCON format wraps the conversation turns inside an outer list.
    turns = conversation_list[0]
    normalized_turns: list[dict[str, Any]] = []
    for turn in turns:
        normalized_turn = dict(turn)
        evidence = normalized_turn.get("expanded emotion cause evidence")
        normalized_turn["has_multiple_causes"] = (
            isinstance(evidence, list) and len(evidence) > 1
        )
        normalized_turn["semantic_embedding"] = []
        normalized_turn["vad_scores"] = []
        normalized_turns.append(normalized_turn)

    return {
        "conversation_id": conversation_id,
        "dataset_split": dataset_split,
        "target_turn": -1,
        "golden_turn": -1,
        "turns": normalized_turns,
    }


def format_dataset(raw_paths: list[Path]) -> list[dict[str, Any]]:
    """Load raw RECCON files and return a single flattened formatted dataset."""
    formatted_dataset: list[dict[str, Any]] = []
    for raw_path in raw_paths:
        raw_data = load_json(raw_path)
        split_name = "train" if raw_path.name.endswith("train.json") else "valid" if raw_path.name.endswith("valid.json") else "test"
        for conversation_id, conversation_list in raw_data.items():
            formatted_dataset.append(
                normalize_conversation(conversation_id, conversation_list, split_name)
            )
    return formatted_dataset


def write_formatted_dataset(output_path: Path | None = None) -> Path:
    """Write the formatted dataset to a JSON file in the data directory."""
    if output_path is None:
        output_path = DATA_DIR / FORMATTED_FILENAME

    input_paths = [DATA_DIR / filename for filename in RAW_FILENAMES]
    formatted_dataset = format_dataset(input_paths)
    write_json(output_path, formatted_dataset)
    validate_formatted_dataset(output_path)
    return output_path


def write_split_dataset_files(input_path: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    """Write train/valid/test split JSON files from a formatted or selected dataset."""
    if input_path is None:
        input_path = DATA_DIR / FORMATTED_FILENAME
    if output_dir is None:
        output_dir = DATA_DIR

    input_path = Path(str(input_path).strip())
    output_dir = Path(str(output_dir).strip())
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_json(input_path)
    if not isinstance(dataset, list):
        raise ValueError("Input dataset must be a list of conversations.")

    split_names = ["train", "valid", "test"]
    written_paths: list[Path] = []
    for split_name in split_names:
        split_dataset = [
            conversation
            for conversation in dataset
            if str(conversation.get("dataset_split") or conversation.get("split") or "").lower() == split_name
        ]
        output_path = output_dir / f"{input_path.stem}_{split_name}.json"
        write_json(output_path, split_dataset)
        written_paths.append(output_path)

    return written_paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Format RECCON data and optionally write split files.")
    parser.add_argument("--input", dest="input_path", help="Input dataset JSON file", default=None)
    parser.add_argument("--output", dest="output_path", help="Output formatted dataset JSON file", default=None)
    parser.add_argument(
        "--write-splits",
        dest="write_splits",
        action="store_true",
        help="Write train/valid/test split files from the input dataset.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Directory for split dataset files",
        default=None,
    )
    args = parser.parse_args()

    if args.write_splits:
        output_paths = write_split_dataset_files(
            input_path=Path(args.input_path) if args.input_path else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        print("Split dataset files written to:")
        for path in output_paths:
            print(f"  - {path}")
    else:
        output_path = write_formatted_dataset(Path(args.output_path) if args.output_path else None)
        print(f"Formatted dataset written to: {output_path}")


if __name__ == "__main__":
    main()
