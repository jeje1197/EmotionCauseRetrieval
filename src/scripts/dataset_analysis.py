"""Analyze message counts across the selected RECCON splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, multimode

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SPLITS = ("train", "valid", "test")
DEFAULT_OUTPUT = DATA_DIR / "output" / "conversation_length_histogram.png"
DEFAULT_EVIDENCE_OUTPUT = DATA_DIR / "output" / "evidence_message_histogram.png"
DEFAULT_CAUSE_OUTPUT = DATA_DIR / "output" / "evidence_cause_histogram.png"
EVIDENCE_FIELD = "expanded emotion cause evidence"


def load_conversation_counts(
    data_dir: Path,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """Count messages by evidence status for each conversation."""
    lengths: list[int] = []
    evidence_counts: list[int] = []
    nonempty_counts: list[int] = []
    single_cause_counts: list[int] = []
    multiple_cause_counts: list[int] = []
    for split in SPLITS:
        path = data_dir / f"reccon_selected_{split}.json"
        with path.open("r", encoding="utf-8") as handle:
            conversations = json.load(handle)
        if not isinstance(conversations, list):
            raise ValueError(f"{path}: expected a list of conversations")
        for index, conversation in enumerate(conversations):
            if not isinstance(conversation, dict):
                raise ValueError(f"{path}: conversation {index} must be an object")
            if conversation.get("dataset_split") != split:
                raise ValueError(f"{path}: conversation {index} must belong to split {split}")
            turns = conversation.get("turns")
            if not isinstance(turns, list):
                raise ValueError(f"{path}: conversation {index} must have a turns list")
            present = single = multiple = 0
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    raise ValueError(
                        f"{path}: conversation {index}, turn {turn_index} must be an object"
                    )
                if EVIDENCE_FIELD in turn:
                    present += 1
                    evidence = turn[EVIDENCE_FIELD]
                    if not isinstance(evidence, list):
                        raise ValueError(
                            f"{path}: conversation {index}, turn {turn_index}: "
                            f"{EVIDENCE_FIELD} must be a list"
                        )
                    if len(evidence) == 1:
                        single += 1
                    elif len(evidence) > 1:
                        multiple += 1
            lengths.append(len(turns))
            evidence_counts.append(present)
            nonempty_counts.append(single + multiple)
            single_cause_counts.append(single)
            multiple_cause_counts.append(multiple)
    if not lengths:
        raise ValueError("The selected split files contain no conversations")
    return lengths, evidence_counts, nonempty_counts, single_cause_counts, multiple_cause_counts


def save_histogram(
    lengths: list[int], output_path: Path, *, xlabel: str, title: str
) -> None:
    """Save one bar per observed message count."""
    counts = Counter(lengths)
    figure, axis = plt.subplots()
    axis.bar(sorted(counts), [counts[length] for length in sorted(counts)], width=0.9)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Number of conversations")
    axis.set_title(title)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path)
    finally:
        plt.close(figure)


def save_cause_histogram(
    nonempty_counts: list[int], single_counts: list[int],
    multiple_counts: list[int], output_path: Path,
) -> None:
    """Compare per-conversation counts of messages with annotated causes."""
    distributions = [
        ("Any annotated cause", Counter(nonempty_counts)),
        ("Exactly 1 annotated cause", Counter(single_counts)),
        ("2 or more annotated causes", Counter(multiple_counts)),
    ]
    max_count = max(max(counts) for _, counts in distributions)
    positions = range(max_count + 1)
    figure, axis = plt.subplots(figsize=(11, 6))
    for offset, (label, counts) in enumerate(distributions):
        axis.bar(
            [position + (offset - 1) * 0.25 for position in positions],
            [counts[position] for position in positions],
            width=0.25,
            label=label,
        )
    axis.set_xlabel("Number of messages in a conversation")
    axis.set_ylabel("Number of conversations with that many messages")
    axis.set_title("Messages with annotated emotion causes per conversation")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.legend(title="Message categories")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path)
    finally:
        plt.close(figure)


def print_count_metrics(label: str, counts: list[int]) -> None:
    print(f"Total {label}: {sum(counts)}")
    print(f"Mean {label} per conversation: {mean(counts):.2f}")
    print(f"Median {label} per conversation: {median(counts):g}")
    print(f"Mode {label} per conversation: {', '.join(map(str, sorted(multimode(counts))))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Directory containing reccon_selected_train/valid/test.json",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Path to the conversation-length histogram image",
    )
    parser.add_argument(
        "--evidence-output", type=Path, default=DEFAULT_EVIDENCE_OUTPUT,
        help="Path to the evidence-message histogram image",
    )
    parser.add_argument(
        "--cause-output", type=Path, default=DEFAULT_CAUSE_OUTPUT,
        help="Path to the evidence and cause-count histogram image",
    )
    args = parser.parse_args()

    outputs = [args.output, args.evidence_output, args.cause_output]
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error("--output, --evidence-output, and --cause-output must be different paths")

    lengths, evidence_counts, nonempty_counts, single_counts, multiple_counts = (
        load_conversation_counts(args.data_dir)
    )
    modes = sorted(multimode(lengths))
    print(f"Total conversations: {len(lengths)}")
    print(f"Mean messages: {mean(lengths):.2f}")
    print(f"Median messages: {median(lengths):g}")
    print(f"Mode messages: {', '.join(map(str, modes))}")
    print_count_metrics("messages with evidence field", evidence_counts)
    print_count_metrics("messages with non-empty evidence", nonempty_counts)
    print_count_metrics("messages with one cause", single_counts)
    print_count_metrics("messages with multiple causes", multiple_counts)
    save_histogram(
        lengths, args.output,
        xlabel="Number of messages in a conversation",
        title="RECCON conversation lengths",
    )
    print(f"Histogram saved to: {args.output}")
    save_histogram(
        evidence_counts, args.evidence_output,
        xlabel="Messages with an emotion-cause evidence field per conversation",
        title="Conversations by number of messages with an evidence field",
    )
    print(f"Evidence histogram saved to: {args.evidence_output}")
    save_cause_histogram(nonempty_counts, single_counts, multiple_counts, args.cause_output)
    print(f"Evidence and cause histogram saved to: {args.cause_output}")


if __name__ == "__main__":
    main()
