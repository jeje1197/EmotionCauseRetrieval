"""Sample eligible conversations and measure VAD stability in three stages."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Any

from src.clients.gemini_client import GeminiClient
from src.scripts.dataset_analysis import (
    DATA_DIR,
    EMOTION_ALIASES,
    EVIDENCE_FIELD,
    load_conversations,
    qualifying_targets,
)
from src.scripts.embed_dataset_helpers import load_json, retry_call, score_vad, validate_vad_scores
from src.scripts.vad_stability import FIELDS, atomic_write_json, is_blank, stability_metrics

DEFAULT_DIR = DATA_DIR / "vad_stability"
DEFAULT_INPUT = DEFAULT_DIR / "stability_dataset.json"
DEFAULT_EMBEDDED = DEFAULT_DIR / "stability_dataset_embedded.json"
DEFAULT_REPORT = DEFAULT_DIR / "stability_sample_report.json"
SAMPLE_SEED = 42


def sample_conversations(conversations: list[dict[str, Any]], seed: int = SAMPLE_SEED) -> list[dict[str, Any]]:
    """Draw 10%, stratified by split and first qualifying target emotion."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str]] = set()
    for conversation in conversations:
        targets = qualifying_targets(conversation)
        if not targets:
            continue
        target = min(targets, key=lambda turn: turn["turn"])
        split = conversation.get("dataset_split")
        identifier = conversation.get("conversation_id")
        emotion = target.get("emotion")
        if not isinstance(split, str) or not isinstance(identifier, str) or not identifier:
            raise ValueError("Eligible conversations must have a split and conversation_id")
        if (split, identifier) in identities:
            raise ValueError(f"Duplicate conversation identity: {split}/{identifier}")
        identities.add((split, identifier))
        if not isinstance(emotion, str) or not emotion.strip():
            raise ValueError(f"{split}/{identifier}: target emotion must be a non-empty string")
        label = emotion.strip().lower()
        groups[(split, EMOTION_ALIASES.get(label, label))].append(conversation)
    total = sum(len(group) for group in groups.values())
    if not total:
        raise ValueError("No conversations satisfy the eligibility criteria")

    sample_size = (total + 5) // 10
    if sample_size == 0:
        raise ValueError(f"Eligible population of {total} is too small for a 10% sample")
    quotas = {key: len(group) // 10 for key, group in groups.items()}
    remainder = sample_size - sum(quotas.values())
    ranked = sorted(groups, key=lambda key: (-(len(groups[key]) % 10), key))
    for key in ranked[:remainder]:
        quotas[key] += 1

    rng = random.Random(seed)
    sample = []
    for key in sorted(groups):
        population = sorted(groups[key], key=lambda item: item["conversation_id"])
        for original in rng.sample(population, quotas[key]):
            conversation = copy.deepcopy(original)
            target = min(qualifying_targets(conversation), key=lambda turn: turn["turn"])
            conversation["target_turn"] = target["turn"]
            conversation["golden_turn"] = target[EVIDENCE_FIELD][0]
            for turn in conversation["turns"]:
                turn.pop("vad_scores", None)
                for field in FIELDS:
                    turn.pop(field, None)
            sample.append(conversation)
    return sample


def validate_sample(dataset: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"{path}: expected a non-empty list of conversations")
    seen = set()
    for conversation in dataset:
        if not isinstance(conversation, dict) or not isinstance(conversation.get("turns"), list):
            raise ValueError(f"{path}: each conversation must contain a turns list")
        split, identifier = conversation.get("dataset_split"), conversation.get("conversation_id")
        if not isinstance(split, str) or split not in {"train", "valid", "test"}:
            raise ValueError(f"{path}: invalid dataset split {split!r}")
        if not isinstance(identifier, str) or not identifier or (split, identifier) in seen:
            raise ValueError(f"{path}: missing or duplicate conversation ID {identifier!r}")
        seen.add((split, identifier))
        turns = conversation["turns"]
        if any(not isinstance(turn, dict) for turn in turns):
            raise ValueError(f"{path}: {identifier}: every turn must be an object")
        ids = [turn.get("turn") for turn in turns]
        if any(type(n) is not int for n in ids) or sorted(ids) != list(range(1, len(ids) + 1)):
            raise ValueError(f"{path}: {identifier}: invalid turn numbering")
        targets = qualifying_targets(conversation)
        if not targets:
            raise ValueError(f"{path}: {identifier}: no qualifying target")
        first = min(targets, key=lambda turn: turn["turn"])
        if conversation.get("target_turn") != first["turn"] or conversation.get("golden_turn") != first[EVIDENCE_FIELD][0]:
            raise ValueError(f"{path}: {identifier}: target/golden turn does not match the first qualifying target")
        for turn in turns:
            if turn["turn"] <= first["turn"]:
                utterance = turn.get("utterance")
                if not isinstance(utterance, str) or not utterance.strip():
                    raise ValueError(f"{path}: {identifier}: turn {turn['turn']} has an empty utterance")
                for field in FIELDS:
                    if not is_blank(turn.get(field)):
                        try:
                            validate_vad_scores(turn[field])
                        except ValueError as exc:
                            raise ValueError(f"{path}: {identifier}: turn {turn['turn']}, {field}: {exc}") from exc
    return dataset


def without_scores(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = copy.deepcopy(dataset)
    for conversation in result:
        for turn in conversation["turns"]:
            for field in FIELDS:
                turn.pop(field, None)
    return result


def generate_scores(
    input_path: Path, output_path: Path, *, overwrite: bool = False,
    checkpoint_interval: int = 50, max_retries: int = 5, concurrent_requests: int = 10,
) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and embedded output paths must be different")
    if checkpoint_interval < 1 or max_retries < 0 or concurrent_requests < 1:
        raise ValueError("Checkpoint interval and concurrency must be positive and retries non-negative")
    source = validate_sample(load_json(input_path), input_path)
    if output_path.exists():
        dataset = validate_sample(load_json(output_path), output_path)
        if without_scores(dataset) != without_scores(source):
            raise ValueError("Existing embedded dataset does not match the input sample; choose a different output path")
    else:
        dataset = without_scores(source)
        atomic_write_json(output_path, dataset)

    if overwrite:
        dataset = without_scores(dataset)
        atomic_write_json(output_path, dataset)

    tasks = (
        (conversation, turn, field)
        for conversation in dataset
        for turn in conversation["turns"]
        if turn["turn"] <= conversation["target_turn"]
        for field in FIELDS
        if is_blank(turn.get(field))
    )
    first_batch = list(islice(tasks, concurrent_requests))
    if not first_batch:
        print(f"Generated 0 VAD scores; saved {output_path}")
        return
    client = GeminiClient()
    pending = generated = 0

    def request(text: str) -> dict[str, float]:
        return validate_vad_scores(retry_call(
            score_vad, max_retries=max_retries, client=client, text=text,
        ))

    try:
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            batch = first_batch
            while batch:
                futures = {
                    executor.submit(request, turn["utterance"]): (conversation, turn, field)
                    for conversation, turn, field in batch
                }
                failure: Exception | None = None
                for future in as_completed(futures):
                    conversation, turn, field = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        if failure is None:
                            failure = RuntimeError(
                                f"VAD request failed for {conversation['conversation_id']}, "
                                f"turn {turn['turn']}, {field}: {exc}"
                            )
                            failure.__cause__ = exc
                        continue
                    turn[field] = result
                    generated += 1
                    pending += 1
                    if pending >= checkpoint_interval:
                        atomic_write_json(output_path, dataset)
                        pending = 0
                        print(f"Checkpointed {generated} generated scores")
                if failure is not None:
                    raise failure
                batch = list(islice(tasks, concurrent_requests))
    finally:
        if pending:
            atomic_write_json(output_path, dataset)
    print(f"Generated {generated} VAD scores; saved {output_path}")


def write_report(input_path: Path, report_path: Path) -> dict[str, Any]:
    if input_path.resolve() == report_path.resolve():
        raise ValueError("Report path must be different from the dataset path")
    dataset = validate_sample(load_json(input_path), input_path)
    prefixes = [
        {**conversation, "turns": [
            turn for turn in conversation["turns"] if turn["turn"] <= conversation["target_turn"]
        ]}
        for conversation in dataset
    ]
    report = stability_metrics([prefixes])
    if report["incomplete_turns"]:
        raise ValueError(f"{report['incomplete_turns']} messages have incomplete VAD runs; resume scoring first")
    report["conversations"] = len(dataset)
    report["scope"] = "Messages through the first qualifying target in each sampled conversation"
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2))
    print(f"Stability report saved to: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    sample = subparsers.add_parser("sample", help="Create a reproducible 10% stratified sample")
    sample.add_argument("--data-dir", type=Path, default=DATA_DIR)
    sample.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    sample.add_argument("--seed", type=int, default=SAMPLE_SEED)
    score = subparsers.add_parser("score", help="Generate or resume three VAD scores per message")
    score.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    score.add_argument("--output", type=Path, default=DEFAULT_EMBEDDED)
    score.add_argument("--overwrite", action="store_true", help="Clear and regenerate all three VAD scores")
    score.add_argument("--checkpoint-interval", type=int, default=50)
    score.add_argument("--max-retries", type=int, default=5)
    score.add_argument("--concurrent-requests", type=int, default=10, help="Maximum concurrent VAD requests (default: 10)")
    report = subparsers.add_parser("report", help="Compute stability metrics without API calls")
    report.add_argument("--input", type=Path, default=DEFAULT_EMBEDDED)
    report.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.stage == "sample":
        if args.output.exists():
            raise FileExistsError(f"Sample output already exists: {args.output}; choose a new output path")
        dataset = sample_conversations(load_conversations(args.data_dir), args.seed)
        validate_sample(dataset, args.output)
        atomic_write_json(args.output, dataset)
        turns = sum(sum(t["turn"] <= c["target_turn"] for t in c["turns"]) for c in dataset)
        print(f"Saved {len(dataset)} conversations, {turns} prefix messages, and {turns * 3} VAD requests to {args.output}")
    elif args.stage == "score":
        generate_scores(
            args.input, args.output, overwrite=args.overwrite,
            checkpoint_interval=args.checkpoint_interval, max_retries=args.max_retries,
            concurrent_requests=args.concurrent_requests,
        )
    else:
        write_report(args.input, args.output)


if __name__ == "__main__":
    main()
