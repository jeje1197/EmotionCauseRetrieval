"""Select target and golden turns from formatted RECCON conversations.

This stage inspects each conversation and assigns target_turn and golden_turn fields
based on the presence of expanded emotion cause evidence in eligible turns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.scripts.format_dataset import write_split_dataset_files

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


def validate_selected_dataset(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Selected dataset file does not exist: {path}")

    dataset = load_json(path)
    if not isinstance(dataset, list):
        raise ValueError("Selected dataset should be a list of conversations.")

    selected_count = 0
    target_turn_distribution: dict[int, int] = {}

    for conversation in dataset:
        if not isinstance(conversation, dict):
            raise ValueError("Each conversation must be a dictionary.")
        if not isinstance(conversation.get("conversation_id"), str):
            raise ValueError("Each conversation must include a string conversation_id.")
        target_turn = conversation.get("target_turn")
        golden_turn = conversation.get("golden_turn")
        if not isinstance(target_turn, int) or not isinstance(golden_turn, int):
            raise ValueError("Each conversation must include integer target_turn and golden_turn values.")
        turns = conversation.get("turns")
        if not isinstance(turns, list):
            raise ValueError("Each conversation must include a turns list.")

        # Ensure the golden turn refers to a prior turn (golden < target)
        if target_turn >= 5 and golden_turn >= 0:
            if golden_turn >= target_turn:
                raise ValueError(
                    f"Conversation {conversation.get('conversation_id')} has golden_turn >= target_turn ({golden_turn} >= {target_turn})."
                )
            selected_count += 1
            target_turn_distribution[target_turn] = target_turn_distribution.get(target_turn, 0) + 1

    total_conversations = len(dataset)
    print(f"Selected conversations: {selected_count}/{total_conversations}")
    if target_turn_distribution:
        print("Target turn distribution:")
        for turn_value in sorted(target_turn_distribution):
            print(f"  turn {turn_value}: {target_turn_distribution[turn_value]}")
    else:
        print("No eligible target turns were selected.")
        raise ValueError("No valid selected turns found in the selected dataset.")


def _parse_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def select_turns_for_conversation(conversation: dict[str, Any]) -> dict[str, Any]:
    """Compute target_turn and golden_turn fields for one conversation."""
    turns = conversation.get("turns", [])
    selected = dict(conversation)
    selected["target_turn"] = -1
    selected["golden_turn"] = -1

    eligible_turns: list[dict[str, Any]] = []
    for turn in turns:
        turn_index = _parse_int(turn.get("turn"))
        if turn_index is None:
            continue
        if turn_index < 5:
            continue
        evidence = turn.get("expanded emotion cause evidence")
        if not evidence:
            continue
        eligible_turns.append(turn)

    single_cause_turns = [
        turn for turn in eligible_turns
        if isinstance(turn.get("expanded emotion cause evidence"), list)
        and len(turn.get("expanded emotion cause evidence", [])) == 1
    ]

    preferred_turns = single_cause_turns if single_cause_turns else eligible_turns
    if preferred_turns:
        chosen = preferred_turns[0]
        chosen_turn = _parse_int(chosen.get("turn"))
        chosen_evidence = chosen.get("expanded emotion cause evidence")
        selected["target_turn"] = chosen_turn if chosen_turn is not None else -1
        # Pick the first evidence index that is a valid int and strictly less than the target turn
        if isinstance(chosen_evidence, list) and chosen_evidence and chosen_turn is not None:
            for ev in chosen_evidence:
                golden_candidate = _parse_int(ev)
                if golden_candidate is not None and golden_candidate < chosen_turn:
                    selected["golden_turn"] = golden_candidate
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
    validate_selected_dataset(output_path)
    return output_path


def write_selected_split_dataset_files(
    selected_input_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    if selected_input_path is None:
        selected_input_path = DATA_DIR / SELECTED_FILENAME
    if output_dir is None:
        output_dir = DATA_DIR

    return write_split_dataset_files(input_path=selected_input_path, output_dir=output_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Select turns and optionally write split files.")
    parser.add_argument("--input", dest="input_path", help="Input formatted dataset JSON", default=None)
    parser.add_argument("--output", dest="output_path", help="Output selected dataset JSON", default=None)
    parser.add_argument(
        "--write-splits",
        dest="write_splits",
        action="store_true",
        help="Write train/valid/test split files from the selected dataset.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Directory for split dataset files",
        default=None,
    )
    args = parser.parse_args()

    output_path = write_selected_dataset(
        input_path=Path(args.input_path) if args.input_path else None,
        output_path=Path(args.output_path) if args.output_path else None,
    )
    print(f"Selected dataset written to: {output_path}")

    if args.write_splits:
        split_paths = write_selected_split_dataset_files(
            selected_input_path=Path(args.output_path) if args.output_path else output_path,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        print("Split dataset files written to:")
        for split_path in split_paths:
            print(f"  - {split_path}")


if __name__ == "__main__":
    main()
