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


def normalize_conversation(conversation_id: str, conversation_list: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Normalize a single RECCON conversation structure into a flattened dict."""
    if not conversation_list:
        return {
            "conversation_id": conversation_id,
            "target_turn": -1,
            "golden_turn": -1,
            "turns": [],
        }

    # RECCON format wraps the conversation turns inside an outer list.
    turns = conversation_list[0]
    normalized_turns: list[dict[str, Any]] = []
    for turn in turns:
        normalized_turn = dict(turn)
        normalized_turn["semantic_embedding"] = []
        normalized_turn["vad_scores"] = []
        normalized_turns.append(normalized_turn)

    return {
        "conversation_id": conversation_id,
        "target_turn": -1,
        "golden_turn": -1,
        "turns": normalized_turns,
    }


def format_dataset(raw_paths: list[Path]) -> list[dict[str, Any]]:
    """Load raw RECCON files and return a single flattened formatted dataset."""
    formatted_dataset: list[dict[str, Any]] = []
    for raw_path in raw_paths:
        raw_data = load_json(raw_path)
        for conversation_id, conversation_list in raw_data.items():
            formatted_dataset.append(normalize_conversation(conversation_id, conversation_list))
    return formatted_dataset


def write_formatted_dataset(output_path: Path | None = None) -> Path:
    """Write the formatted dataset to a JSON file in the data directory."""
    if output_path is None:
        output_path = DATA_DIR / FORMATTED_FILENAME

    input_paths = [DATA_DIR / filename for filename in RAW_FILENAMES]
    formatted_dataset = format_dataset(input_paths)
    write_json(output_path, formatted_dataset)
    return output_path


def main() -> None:
    output_path = write_formatted_dataset()
    print(f"Formatted dataset written to: {output_path}")


if __name__ == "__main__":
    main()
