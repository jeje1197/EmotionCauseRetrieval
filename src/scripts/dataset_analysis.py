"""Analyze conversations and emotion-cause annotations across all RECCON splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, multimode

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.scripts.select_turns import select_turns_for_conversation

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SPLITS = ("train", "valid", "test")
DEFAULT_OUTPUT = DATA_DIR / "output" / "conversation_length_histogram.png"
DEFAULT_EVIDENCE_OUTPUT = DATA_DIR / "output" / "evidence_message_histogram.png"
DEFAULT_CAUSE_OUTPUT = DATA_DIR / "output" / "evidence_cause_histogram.png"
DEFAULT_ANALYSIS_DIR = DATA_DIR / "output" / "reccon_dataset_analysis_valid"
EVIDENCE_FIELD = "expanded emotion cause evidence"
EMOTION_ALIASES = {
    "angry": "anger",
    "happines": "happiness",
    "happy": "happiness",
    "sad": "sadness",
    "surprised": "surprise",
}


def load_conversations(data_dir: Path) -> list[dict]:
    conversations = []
    for split in SPLITS:
        path = data_dir / f"reccon_selected_{split}.json"
        with path.open("r", encoding="utf-8") as handle:
            split_conversations = json.load(handle)
        if not isinstance(split_conversations, list):
            raise ValueError(f"{path}: expected a list of conversations")
        for index, conversation in enumerate(split_conversations):
            if not isinstance(conversation, dict):
                raise ValueError(f"{path}: conversation {index} must be an object")
            if conversation.get("dataset_split") != split:
                raise ValueError(f"{path}: conversation {index} must belong to split {split}")
            if not isinstance(conversation.get("turns"), list):
                raise ValueError(f"{path}: conversation {index} must have a turns list")
            conversations.append(conversation)
    if not conversations:
        raise ValueError("The selected split files contain no conversations")
    return conversations


def distribution(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "mode": []}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "mode": sorted(multimode(values)),
    }


def analyze_conversations(conversations: list[dict]) -> tuple[dict, dict[str, list[int]]]:
    """Analyze complete conversations, excluding malformed evidence fields."""
    series: dict[str, list[int]] = {
        "messages_per_conversation": [],
        "causes_per_annotated_message": [],
        "annotated_messages_per_conversation": [],
        "valid_evidence_fields_per_conversation": [],
        "single_cause_messages_per_conversation": [],
        "multiple_cause_messages_per_conversation": [],
        "first_annotated_message_position": [],
        "first_cause_of_another_message_position": [],
        "shared_messages_per_conversation": [],
        "unique_messages_per_conversation": [],
    }
    eligible_series: dict[str, list[int]] = {
        "shared": [], "unique": [],
    }
    emotions: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    empty = missing = multiple = repeated_entries = invalid_fields = invalid_entries = 0
    invalid_fields_with_valid_entries = 0
    no_annotated = no_other_cause = 0
    for conversation in conversations:
        turns = conversation["turns"]
        context = f"conversation {conversation.get('conversation_id', '<unknown>')}"
        if any(not isinstance(turn, dict) for turn in turns):
            raise ValueError(f"{context}: each turn must be an object")
        ids = [turn.get("turn") for turn in turns]
        if any(type(number) is not int for number in ids) or sorted(ids) != list(range(1, len(turns) + 1)):
            raise ValueError(f"{context}: turn numbers must be unique and consecutive starting at 1")
        annotated = []
        cause_positions = set()
        present = singles = multiples = 0
        for turn in turns:
            number = turn["turn"]
            if EVIDENCE_FIELD not in turn:
                missing += 1
                empty += 1
                continue
            evidence = turn[EVIDENCE_FIELD]
            if not isinstance(evidence, list):
                invalid_fields += 1
                invalid_reasons["field_not_list"] += 1
                continue
            bad = [cause for cause in evidence if type(cause) is not int or cause not in ids]
            if bad:
                invalid_fields += 1
                invalid_entries += len(bad)
                invalid_fields_with_valid_entries += len(bad) < len(evidence)
                for cause in bad:
                    reason = "non_integer_entry" if type(cause) is not int else "unknown_turn_number"
                    invalid_reasons[reason] += 1
                continue
            present += 1
            if not evidence:
                empty += 1
                continue
            emotion = turn.get("emotion")
            if not isinstance(emotion, str) or not emotion.strip():
                raise ValueError(f"{context}, turn {number}: annotated message must have an emotion")
            emotion = emotion.strip().lower()
            emotion = EMOTION_ALIASES.get(emotion, emotion)
            numbered = set(evidence)
            repeated_entries += len(evidence) - len(set(evidence))
            cause_positions.update(cause for cause in numbered if cause != number)
            series["causes_per_annotated_message"].append(len(evidence))
            multiple += len(evidence) > 1
            singles += len(evidence) == 1
            multiples += len(evidence) > 1
            emotions[emotion] += 1
            annotated.append((number, numbered))

        frequencies = Counter(cause for _, causes in annotated for cause in causes)
        shared = unique = 0
        for _, causes in annotated:
            if any(frequencies[cause] > 1 for cause in causes):
                shared += 1
            else:
                unique += 1
        series["messages_per_conversation"].append(len(turns))
        series["annotated_messages_per_conversation"].append(len(annotated))
        series["valid_evidence_fields_per_conversation"].append(present)
        series["single_cause_messages_per_conversation"].append(singles)
        series["multiple_cause_messages_per_conversation"].append(multiples)
        series["shared_messages_per_conversation"].append(shared)
        series["unique_messages_per_conversation"].append(unique)
        if len(annotated) >= 2:
            eligible_series["shared"].append(shared)
            eligible_series["unique"].append(unique)
        if annotated:
            series["first_annotated_message_position"].append(min(item[0] for item in annotated))
        else:
            no_annotated += 1
        if cause_positions:
            series["first_cause_of_another_message_position"].append(min(cause_positions))
        else:
            no_other_cause += 1

    report = {
        "definitions": {
            "scope": "All turns in the selected train, valid, and test conversations.",
            "cause_entries": "Entries in valid evidence lists, including repeated turn references.",
            "invalid_fields": "Entire evidence fields are excluded if not lists or if any entry is not an integer turn number in the conversation. Messages remain in conversation and message totals.",
            "emotion": "Labels are trimmed, lowercased, and normalized using emotion_aliases; excited remains separate.",
            "positions": "One-based turn numbers; first-cause positions exclude self-references.",
            "sharing": "A numbered cause is shared if it occurs in another message in the same conversation.",
            "unique": "A message with no shared numbered cause.",
        },
        "emotion_aliases": EMOTION_ALIASES,
        "total_conversations": len(conversations),
        "total_messages": sum(series["messages_per_conversation"]),
        "messages_with_nonempty_evidence": len(series["causes_per_annotated_message"]),
        "messages_with_empty_or_missing_evidence": empty,
        "messages_with_missing_evidence_field": missing,
        "messages_with_multiple_cause_entries": multiple,
        "excluded_invalid_evidence_fields": invalid_fields,
        "excluded_invalid_list_entries": invalid_entries,
        "excluded_invalid_fields_with_valid_entries": invalid_fields_with_valid_entries,
        "excluded_invalid_reasons": dict(sorted(invalid_reasons.items())),
        "repeated_cause_entries_within_messages": repeated_entries,
        "conversations_without_annotated_messages": no_annotated,
        "conversations_without_a_numbered_cause_of_another_message": no_other_cause,
        "distributions": {key: distribution(values) for key, values in series.items()},
        "shared_unique_in_conversations_with_multiple_annotated_messages": {
            key: distribution(values) for key, values in eligible_series.items()
        },
        "annotated_messages_by_emotion": dict(sorted(emotions.items())),
    }
    for key, values in eligible_series.items():
        series[f"{key}_messages_in_multiple_annotated_conversations"] = values
    return report, series


def analyze_eligibility(conversations: list[dict]) -> dict:
    """Compare proposed pre-embedding eligibility with the current selector."""
    keys = (
        "conversations", "valid_annotation_after_turn_4", "single_cause_after_turn_4",
        "single_previous_cause_after_turn_4", "nonempty_cause_utterance",
        "qualifying_messages", "selected_target_and_golden", "selected_single_cause",
    )
    by_split = {split: dict.fromkeys(keys, 0) for split in SPLITS}
    first_positions: list[int] = []
    for conversation in conversations:
        split = conversation.get("dataset_split")
        if split not in by_split:
            raise ValueError(f"Invalid dataset split: {split!r}")
        counts = by_split[split]
        counts["conversations"] += 1
        turns = conversation["turns"]
        by_turn = {turn["turn"]: turn for turn in turns}
        annotated = []
        for turn in turns:
            evidence = turn.get(EVIDENCE_FIELD, [])
            if (
                turn["turn"] >= 5 and isinstance(evidence, list) and evidence
                and all(type(cause) is int and cause in by_turn for cause in evidence)
            ):
                annotated.append(turn)
        if annotated:
            counts["valid_annotation_after_turn_4"] += 1
            first_positions.append(min(turn["turn"] for turn in annotated))
        single = [turn for turn in annotated if len(turn[EVIDENCE_FIELD]) == 1]
        counts["single_cause_after_turn_4"] += bool(single)
        previous = [
            turn for turn in single if turn[EVIDENCE_FIELD][0] < turn["turn"]
        ]
        counts["single_previous_cause_after_turn_4"] += bool(previous)
        qualifying = [
            turn for turn in previous
            if isinstance(by_turn[turn[EVIDENCE_FIELD][0]].get("utterance"), str)
            and by_turn[turn[EVIDENCE_FIELD][0]]["utterance"].strip()
        ]
        counts["nonempty_cause_utterance"] += bool(qualifying)
        counts["qualifying_messages"] += len(qualifying)

        selected = select_turns_for_conversation(conversation)
        target = selected["target_turn"]
        if target >= 5 and selected["golden_turn"] >= 0:
            counts["selected_target_and_golden"] += 1
            evidence = by_turn[target].get(EVIDENCE_FIELD)
            counts["selected_single_cause"] += isinstance(evidence, list) and len(evidence) == 1
    return {
        "by_split": by_split,
        "totals": {key: sum(counts[key] for counts in by_split.values()) for key in keys},
        "first_valid_annotation_after_turn_4": distribution(first_positions),
        "conversations_without_valid_annotation_after_turn_4": len(conversations) - len(first_positions),
    }


def load_saved_experiment(results_dir: Path) -> dict:
    """Read counts from explicitly supplied, existing per-split experiment results."""
    by_split = {}
    for split in SPLITS:
        path = results_dir / f"results_{split}_100_0.json"
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        metrics = result["metrics"]
        count = metrics["evaluated_conversation_count"]
        if result["split"] != split or type(count) is not int or count < 0:
            raise ValueError(f"{path}: invalid split or evaluated count")
        if len(result["conversation_results"]) != count:
            raise ValueError(f"{path}: evaluated count does not match conversation results")
        by_split[split] = {"evaluated_conversations": count, "source": str(path)}
    return {
        "by_split": by_split,
        "evaluated_conversations": sum(item["evaluated_conversations"] for item in by_split.values()),
    }


def render_markdown(report: dict) -> str:
    eligibility = report["eligibility"]
    totals = eligibility["totals"]
    lines = [
        "# RECCON dataset analysis",
        "",
        "Generated from the selected train, valid, and test files. All turns are",
        "included in dataset statistics; eligibility is evaluated before embedding.",
        "",
        "## Dataset overview",
        "",
        f"- Conversations: {report['total_conversations']:,}",
        f"- Messages: {report['total_messages']:,}",
        f"- Messages with valid non-empty evidence: {report['messages_with_nonempty_evidence']:,}",
        f"- Messages with empty or missing evidence: {report['messages_with_empty_or_missing_evidence']:,}",
        f"- Invalid evidence fields excluded: {report['excluded_invalid_evidence_fields']:,}"
        f" ({report['excluded_invalid_list_entries']:,} invalid list entries;"
        f" {report['excluded_invalid_fields_with_valid_entries']:,} fields also contain valid entries).",
        f"- Valid annotated messages with multiple cause entries: {report['messages_with_multiple_cause_entries']:,}",
        "",
        "Invalid fields, including mixed valid/invalid lists, are excluded in full",
        "from cause analyses. Messages remain in conversation-length totals. Cause",
        "entry counts include duplicate references. Emotion labels are lowercased",
        "and normalized using the aliases in the JSON report; excited stays separate.",
        "",
        "## Distributions across all conversations",
        "",
        "| Distribution | Count | Min | Max | Mean | Median | Mode |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    distributions = [
        ("Messages per conversation", report["distributions"]["messages_per_conversation"]),
        ("Cause entries per annotated message", report["distributions"]["causes_per_annotated_message"]),
        ("Annotated messages per conversation", report["distributions"]["annotated_messages_per_conversation"]),
        ("First annotated message position", report["distributions"]["first_annotated_message_position"]),
        ("First message causing another turn", report["distributions"]["first_cause_of_another_message_position"]),
        ("Shared-cause messages per eligible conversation", report["shared_unique_in_conversations_with_multiple_annotated_messages"]["shared"]),
        ("Unique-cause messages per eligible conversation", report["shared_unique_in_conversations_with_multiple_annotated_messages"]["unique"]),
        ("First valid annotated message after turn 4", eligibility["first_valid_annotation_after_turn_4"]),
    ]
    for label, stats in distributions:
        def display(value: object) -> str:
            if value is None:
                return "N/A"
            if isinstance(value, list):
                return ", ".join(map(str, value)) or "N/A"
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)
        lines.append(
            f"| {label} | " + " | ".join(
                display(stats[key]) for key in ("count", "min", "max", "mean", "median", "mode")
            ) + " |"
        )
    lines.extend([
        "",
        f"First-annotation positions exclude {report['conversations_without_annotated_messages']} conversations without valid annotations.",
        f"First-cause positions exclude {report['conversations_without_a_numbered_cause_of_another_message']} conversations"
        " without a cause referencing another turn; self-references are excluded.",
        f"First annotations after turn 4 exclude {eligibility['conversations_without_valid_annotation_after_turn_4']} conversations.",
        "Shared/unique distributions include only conversations with at least two valid",
        "annotated messages. Messages share a cause if they reference the same numbered turn.",
        "",
        "### Normalized emotions in valid annotated messages",
        "",
        "| Emotion | Messages |",
        "|---|---:|",
    ])
    for emotion, count in sorted(report["annotated_messages_by_emotion"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {emotion} | {count:,} |")
    lines.extend([
        "",
        "## Proposed pre-embedding eligibility criteria",
        "",
        "A conversation qualifies if any message meets all of these rules:",
        "",
        "1. The message is turn 5 or later.",
        "2. Its evidence field is valid and contains exactly one cause entry.",
        "3. That cause references a strictly earlier turn in the same conversation.",
        "4. The cause turn has a non-empty, non-whitespace utterance.",
        "",
        "| Cumulative criterion | Conversations |",
        "|---|---:|",
        f"| All conversations | {totals['conversations']:,} |",
        f"| Valid, non-empty evidence after turn 4 | {totals['valid_annotation_after_turn_4']:,} |",
        f"| Exactly one valid cause entry | {totals['single_cause_after_turn_4']:,} |",
        f"| Cause is a strictly earlier turn | {totals['single_previous_cause_after_turn_4']:,} |",
        f"| Cause utterance is non-empty | {totals['nonempty_cause_utterance']:,} |",
        "",
        "| Split | Eligible conversations | Qualifying messages |",
        "|---|---:|---:|",
    ])
    for split, counts in eligibility["by_split"].items():
        lines.append(f"| {split} | {counts['nonempty_cause_utterance']:,} | {counts['qualifying_messages']:,} |")
    lines.extend([
        f"| **Total** | **{totals['nonempty_cause_utterance']:,}** | **{totals['qualifying_messages']:,}** |",
        "",
        "## Comparison with the current selector",
        "",
        f"- Current selection yields {totals['selected_target_and_golden']:,} target/golden pairs",
        f"  and {totals['selected_single_cause']:,} with single-entry target evidence.",
        "- The selector prefers the first single-cause target before checking whether",
        "  it has a prior cause; it can miss a later qualifying message. If there are",
        "  no single-cause candidates it falls back to multiple-cause targets.",
        "- The proposed rules are an analysis, not a change to the pipeline selector.",
    ])
    saved = report.get("saved_experiment")
    if saved is not None:
        lines.extend([
            "",
            "## Supplied saved experiment",
            "",
            f"The supplied results report {saved['evaluated_conversations']:,} evaluated conversations.",
            "These are post-embedding results, not the pre-embedding eligibility count.",
            "The historical experiment discussed in the project investigation used",
            "an evaluator that required features for every turn even though embedding",
            "stopped at the target. The current evaluator instead checks turns through",
            "the target; saved counts alone do not establish which version was used.",
            "",
            "| Split | Evaluated | Result file |",
            "|---|---:|---|",
        ])
        for split, entry in saved["by_split"].items():
            lines.append(f"| {split} | {entry['evaluated_conversations']:,} | `{entry['source']}` |")
    else:
        lines.extend([
            "",
            "No saved experiment results were supplied; rerun with",
            "`--saved-results-dir` to include their recorded evaluation counts.",
        ])
    return "\n".join(lines) + "\n"


def save_histogram(
    lengths: list[int], output_path: Path, *, xlabel: str, title: str,
    ylabel: str = "Number of conversations",
) -> None:
    """Save one bar per observed message count."""
    counts = Counter(lengths)
    figure, axis = plt.subplots()
    axis.bar(sorted(counts), [counts[length] for length in sorted(counts)], width=0.9)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path)
    finally:
        plt.close(figure)


def save_category_chart(counts: dict[str, int], output_path: Path, *, title: str, xlabel: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(list(counts), list(counts.values()))
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Number of messages")
    axis.set_title(title)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.tick_params(axis="x", labelrotation=25)
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
    parser.add_argument(
        "--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR,
        help="Directory for the detailed JSON report and additional histograms",
    )
    parser.add_argument(
        "--markdown-report", type=Path, default=None,
        help="Path for the written report (default: report.md in --analysis-dir)",
    )
    parser.add_argument(
        "--saved-results-dir", type=Path, default=None,
        help="Directory with existing results_<split>_100_0.json files to include for comparison",
    )
    args = parser.parse_args()
    markdown_path = args.markdown_report or args.analysis_dir / "report.md"

    histogram_specs = {
        "causes_per_annotated_message": ("Cause entries per annotated message", "Number of messages"),
        "annotated_messages_per_conversation": ("Annotated messages per conversation", "Number of conversations"),
        "first_annotated_message_position": ("First annotated message position", "Number of conversations"),
        "first_cause_of_another_message_position": ("First cause of another message position", "Number of conversations"),
        "shared_messages_per_conversation": ("Messages with a shared numbered cause per conversation", "Number of conversations"),
        "unique_messages_per_conversation": ("Messages with unique numbered causes per conversation", "Number of conversations"),
        "shared_messages_in_multiple_annotated_conversations": ("Shared-cause messages, conversations with multiple annotations", "Number of conversations"),
        "unique_messages_in_multiple_annotated_conversations": ("Unique-cause messages, conversations with multiple annotations", "Number of conversations"),
    }
    outputs = [
        args.output, args.evidence_output, args.cause_output,
        args.analysis_dir / "report.json", markdown_path,
        args.analysis_dir / "evidence_status.png",
        args.analysis_dir / "annotated_messages_by_emotion.png",
        *(args.analysis_dir / f"{key}.png" for key in histogram_specs),
    ]
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error("All report and histogram output paths must be different")

    inputs = [args.data_dir / f"reccon_selected_{split}.json" for split in SPLITS]
    if args.saved_results_dir is not None:
        inputs.extend(args.saved_results_dir / f"results_{split}_100_0.json" for split in SPLITS)
    if {path.resolve() for path in outputs} & {path.resolve() for path in inputs}:
        parser.error("Report and histogram paths must not overwrite input files")

    conversations = load_conversations(args.data_dir)
    report, series = analyze_conversations(conversations)
    report["eligibility"] = analyze_eligibility(conversations)
    if args.saved_results_dir is not None:
        report["saved_experiment"] = load_saved_experiment(args.saved_results_dir)
    lengths = series["messages_per_conversation"]
    evidence_counts = series["valid_evidence_fields_per_conversation"]
    nonempty_counts = series["annotated_messages_per_conversation"]
    single_counts = series["single_cause_messages_per_conversation"]
    multiple_counts = series["multiple_cause_messages_per_conversation"]
    modes = sorted(multimode(lengths))
    print(f"Total conversations: {len(lengths)}")
    print(f"Mean messages: {mean(lengths):.2f}")
    print(f"Median messages: {median(lengths):g}")
    print(f"Mode messages: {', '.join(map(str, modes))}")
    print_count_metrics("messages with valid evidence field", evidence_counts)
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
        xlabel="Messages with a valid evidence field per conversation",
        title="Conversations by number of messages with a valid evidence field",
    )
    print(f"Evidence histogram saved to: {args.evidence_output}")
    save_cause_histogram(nonempty_counts, single_counts, multiple_counts, args.cause_output)
    print(f"Evidence and cause histogram saved to: {args.cause_output}")
    for key, (title, ylabel) in histogram_specs.items():
        save_histogram(
            series[key], args.analysis_dir / f"{key}.png",
            xlabel=title, title=title, ylabel=ylabel,
        )
    save_category_chart(
        {
            "Non-empty": report["messages_with_nonempty_evidence"],
            "Empty or missing": report["messages_with_empty_or_missing_evidence"],
            "Invalid / excluded": report["excluded_invalid_evidence_fields"],
        },
        args.analysis_dir / "evidence_status.png",
        title="Messages by emotion-cause evidence status", xlabel="Evidence status",
    )
    save_category_chart(
        report["annotated_messages_by_emotion"],
        args.analysis_dir / "annotated_messages_by_emotion.png",
        title="Messages with emotion causes by emotion", xlabel="Emotion",
    )
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.analysis_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    print("\nDetailed analysis:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Detailed report and additional histograms saved to: {args.analysis_dir}")
    print(f"Written report saved to: {markdown_path}")


if __name__ == "__main__":
    main()
