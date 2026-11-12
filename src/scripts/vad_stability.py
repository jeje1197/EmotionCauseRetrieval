"""Generate three VAD runs and report stability across all selected splits."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from src.clients.gemini_client import GeminiClient
from src.scripts.embed_dataset_helpers import (
    DATA_DIR,
    load_json,
    retry_call,
    score_vad,
    validate_vad_scores,
)

SPLITS = ("train", "valid", "test")
FIELDS = ("vad_scores_1", "vad_scores_2", "vad_scores_3")
DIMENSIONS = ("valence", "arousal", "dominance")
DEFAULT_OUTPUT_DIR = DATA_DIR / "vad_stability"


def atomic_write_json(path: Path, data: Any) -> None:
    """Replace a checkpoint only after its complete JSON has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
            prefix=f".{path.name}.", suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_split(path: Path, split: str) -> list[dict[str, Any]]:
    dataset = load_json(path)
    if not isinstance(dataset, list):
        raise ValueError(f"{path}: expected a list of conversations")
    for index, conversation in enumerate(dataset):
        if not isinstance(conversation, dict) or conversation.get("dataset_split") != split:
            raise ValueError(f"{path}: conversation {index} must belong to split {split}")
        turns = conversation.get("turns")
        if not isinstance(turns, list):
            raise ValueError(f"{path}: conversation {index} must have a turns list")
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict) or not isinstance(turn.get("utterance"), str):
                raise ValueError(f"{path}: conversation {index}, turn {turn_index} must have a string utterance")
            for field in FIELDS:
                value = turn.get(field)
                if value is not None and value != [] and value != {}:
                    try:
                        validate_vad_scores(value)
                    except ValueError as exc:
                        raise ValueError(f"{path}: conversation {index}, turn {turn_index}, {field}: {exc}") from exc
    return dataset


def is_blank(value: Any) -> bool:
    return value is None or value == [] or value == {}


def stability_metrics(datasets: list[list[dict[str, Any]]]) -> dict[str, Any]:
    runs: dict[str, dict[str, list[float]]] = {
        field: {dimension: [] for dimension in DIMENSIONS} for field in FIELDS
    }
    complete: list[list[dict[str, float]]] = []
    total_turns = 0
    for dataset in datasets:
        for conversation in dataset:
            for turn in conversation["turns"]:
                total_turns += 1
                scores: list[dict[str, float]] = []
                for field in FIELDS:
                    value = turn.get(field)
                    if not is_blank(value):
                        validated = validate_vad_scores(value)
                        scores.append(validated)
                        for dimension in DIMENSIONS:
                            runs[field][dimension].append(validated[dimension])
                if len(scores) == len(FIELDS):
                    complete.append(scores)

    run_summary: dict[str, Any] = {}
    for field in FIELDS:
        count = len(runs[field][DIMENSIONS[0]])
        run_summary[field] = {
            "count": count,
            "dimensions": {
                dimension: {
                    "mean": mean(values) if values else None,
                    "population_stddev": pstdev(values) if values else None,
                }
                for dimension, values in runs[field].items()
            },
        }

    comparisons: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        per_turn_std = [pstdev([score[dimension] for score in scores]) for scores in complete]
        per_turn_range = [
            max(score[dimension] for score in scores) - min(score[dimension] for score in scores)
            for scores in complete
        ]
        pairs = {}
        for left, right in combinations(range(len(FIELDS)), 2):
            differences = [abs(scores[left][dimension] - scores[right][dimension]) for scores in complete]
            pairs[f"{FIELDS[left]}_vs_{FIELDS[right]}"] = (
                mean(differences) if differences else None
            )
        comparisons[dimension] = {
            "mean_within_turn_population_stddev": mean(per_turn_std) if per_turn_std else None,
            "mean_within_turn_range": mean(per_turn_range) if per_turn_range else None,
            "max_within_turn_range": max(per_turn_range) if per_turn_range else None,
            "pairwise_mean_absolute_difference": pairs,
        }
    return {
        "total_turns": total_turns,
        "complete_turns": len(complete),
        "incomplete_turns": total_turns - len(complete),
        "runs": run_summary,
        "within_turn_comparisons": comparisons,
    }


def run_stability(
    input_dir: Path, output_dir: Path, report_path: Path, *,
    overwrite: bool = False, report_only: bool = False,
    checkpoint_interval: int = 50, max_retries: int = 5,
) -> dict[str, Any]:
    if checkpoint_interval < 1:
        raise ValueError("checkpoint interval must be at least 1")
    if max_retries < 0:
        raise ValueError("max retries must be non-negative")
    input_paths = [input_dir / f"reccon_selected_{split}.json" for split in SPLITS]
    output_paths = [output_dir / f"reccon_{split}_vad_stability.json" for split in SPLITS]
    if set(path.resolve() for path in input_paths) & set(path.resolve() for path in output_paths):
        raise ValueError("Stability output paths must not overwrite input files")
    if report_path.resolve() in {path.resolve() for path in input_paths + output_paths}:
        raise ValueError("Report path must be different from every dataset path")

    datasets = []
    for split, source, output in zip(SPLITS, input_paths, output_paths):
        if output.exists():
            dataset = load_split(output, split)
        elif report_only:
            raise FileNotFoundError(f"Stability dataset not found: {output}")
        else:
            dataset = load_split(source, split)
            for conversation in dataset:
                for turn in conversation["turns"]:
                    if "vad_scores" in turn and is_blank(turn["vad_scores"]):
                        del turn["vad_scores"]
            atomic_write_json(output, dataset)
        datasets.append(dataset)

    client: GeminiClient | None = None
    if not report_only:
        for split, dataset, output in zip(SPLITS, datasets, output_paths):
            pending = 0
            generated = 0
            try:
                for conversation in dataset:
                    for turn in conversation["turns"]:
                        for field in FIELDS:
                            if not overwrite and not is_blank(turn.get(field)):
                                continue
                            if client is None:
                                client = GeminiClient()
                            turn[field] = validate_vad_scores(retry_call(
                                score_vad, max_retries=max_retries,
                                client=client, text=turn["utterance"],
                            ))
                            pending += 1
                            generated += 1
                            if pending >= checkpoint_interval:
                                atomic_write_json(output, dataset)
                                pending = 0
                                print(f"{split}: checkpointed {generated} generated scores")
            finally:
                if pending:
                    atomic_write_json(output, dataset)
            print(f"{split}: generated {generated} scores; saved {output}")

    metrics = stability_metrics(datasets)
    if metrics["total_turns"] == 0:
        raise ValueError("The selected split files contain no turns")
    atomic_write_json(report_path, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Stability report saved to: {report_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate all three runs.")
    parser.add_argument("--report-only", action="store_true", help="Report existing scores without API calls.")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Save after this many scores.")
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    if args.overwrite and args.report_only:
        parser.error("--overwrite and --report-only cannot be combined")
    report_path = args.report or args.output_dir / "vad_stability_report.json"
    run_stability(
        args.input_dir, args.output_dir, report_path, overwrite=args.overwrite,
        report_only=args.report_only, checkpoint_interval=args.checkpoint_interval,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
