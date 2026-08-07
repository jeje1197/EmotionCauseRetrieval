"""Select target and golden turns from formatted RECCON conversations.

This stage inspects each conversation and assigns target_turn and golden_turn fields
based on the presence of expanded emotion cause evidence in eligible turns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
FORMATTED_FILENAME = "reccon_formatted.json"
SELECTED_FILENAME = "reccon_selected.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def select_turns_for_conversation(conversation: dict[str, Any]) -> dict[str, Any]:
    """Compute target_turn and golden_turn fields for one conversation."""
    turns = conversation.get("turns", [])
    selected = dict(conversation)
    selected["target_turn"] = -1
    selected["golden_turn"] = -1

    for turn in turns:
        turn_index = turn.get("turn")
        if not isinstance(turn_index, int):
            continue
        if turn_index < 5:
            continue
        evidence = turn.get("expanded emotion cause evidence")
        if not evidence:
            continue

        # Select the first eligible target turn in the conversation.
        selected["target_turn"] = turn_index
        if isinstance(evidence, list) and evidence:
            selected["golden_turn"] = evidence[0]
        break

    return selected


def select_turns(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [select_turns_for_conversation(conversation) for conversation in dataset]


def write_selected_dataset(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    if input_path is None:
        input_path = DATA_DIR / FORMATTED_FILENAME
    if output_path is None:
        output_path = DATA_DIR / SELECTED_FILENAME

    dataset = load_json(input_path)
    selected_dataset = select_turns(dataset)
    write_json(output_path, selected_dataset)
    return output_path


def main() -> None:
    output_path = write_selected_dataset()
    print(f"Selected dataset written to: {output_path}")


if __name__ == "__main__":
    main()
