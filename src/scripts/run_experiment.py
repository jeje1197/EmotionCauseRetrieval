"""Run retrieval experiments on embedded RECCON split datasets.

This module ranks prior turns using a weighted blend of semantic and VAD
similarity, computes retrieval metrics, and writes per-split JSON results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.models import Conversation, ConversationTurn, VadScores

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output"
SPLITS = ("train", "valid", "test")


def resolve_split_input_paths(input_path: Path | None = None, splits: list[str] | None = None) -> list[tuple[str, Path]]:
    if input_path is not None:
        split = infer_split_from_path(input_path)
        if split is None:
            raise ValueError(
                f"Could not infer split from input path: {input_path}. "
                "Expected a filename containing train, valid, or test."
            )
        if not input_path.exists():
            raise FileNotFoundError(f"Input dataset file does not exist: {input_path}")
        return [(split, input_path)]

    if not splits:
        raise ValueError("No splits were provided. Use --splits train|valid|test|all or pass --input.")

    requested_splits = list(SPLITS) if "all" in splits else splits

    split_inputs: list[tuple[str, Path]] = []
    for split in requested_splits:
        semantic_first = DATA_DIR / f"reccon_{split}_embedded_semantic.json"
        fallback = DATA_DIR / f"reccon_{split}_embedded.json"
        if semantic_first.exists():
            split_inputs.append((split, semantic_first))
        elif fallback.exists():
            split_inputs.append((split, fallback))
        else:
            raise FileNotFoundError(
                f"No embedded dataset file found for split '{split}' in {DATA_DIR}. "
                f"Expected {semantic_first.name} or {fallback.name}."
            )

    return split_inputs


def infer_split_from_path(path: Path) -> str | None:
    stem = path.stem.lower()
    for split in SPLITS:
        if f"_{split}_" in stem or stem.endswith(f"_{split}"):
            return split
    return None


def load_conversations(path: Path) -> list[Conversation]:
    with path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    def _safe_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    conversations: list[Conversation] = []
    for item in raw_data:
        dataset_split = str(item.get("dataset_split") or item.get("split") or "").lower()
        if dataset_split not in {"train", "valid", "test"}:
            # Keep model behavior strict for split labels.
            continue

        turns: list[ConversationTurn] = []
        for turn in item.get("turns", []):
            semantic_embedding: list[float] = []
            raw_embedding = turn.get("semantic_embedding")
            if isinstance(raw_embedding, list):
                for value in raw_embedding:
                    if isinstance(value, (int, float)):
                        semantic_embedding.append(float(value))

            vad_scores = None
            raw_vad = turn.get("vad_scores")
            if isinstance(raw_vad, dict):
                try:
                    vad_scores = VadScores(
                        valence=float(raw_vad["valence"]),
                        arousal=float(raw_vad["arousal"]),
                        dominance=float(raw_vad["dominance"]),
                    )
                except (KeyError, TypeError, ValueError):
                    vad_scores = None

            turns.append(
                ConversationTurn(
                    turn=_safe_int(turn.get("turn"), default=-1),
                    speaker=str(turn.get("speaker", "")),
                    utterance=str(turn.get("utterance", "")),
                    emotion=str(turn.get("emotion", "")),
                    semantic_embedding=semantic_embedding,
                    vad_scores=vad_scores,
                    has_multiple_causes=bool(turn.get("has_multiple_causes", False)),
                )
            )

        conversations.append(
            Conversation(
                conversation_id=str(item.get("conversation_id", "")),
                dataset_split=dataset_split,
                target_turn=_safe_int(item.get("target_turn"), default=-1),
                golden_turn=_safe_int(item.get("golden_turn"), default=-1),
                turns=turns,
            )
        )

    return conversations


def vectorize_vad(vad_scores: dict[str, Any]) -> list[float]:
    return [
        float(vad_scores["valence"]),
        float(vad_scores["arousal"]),
        float(vad_scores["dominance"]),
    ]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_semantic_similarity(target_turn: ConversationTurn, candidate_turn: ConversationTurn) -> float:
    return cosine_similarity(
        target_turn.semantic_embedding,
        candidate_turn.semantic_embedding,
    )


def compute_emotion_similarity(target_turn: ConversationTurn, candidate_turn: ConversationTurn) -> float:
    return cosine_similarity(
        vectorize_vad(target_turn.vad_scores.__dict__),
        vectorize_vad(candidate_turn.vad_scores.__dict__),
    )


def normalize_weights(weight_a: float, weight_b: float) -> tuple[float, float]:
    total = weight_a + weight_b
    if total == 0:
        raise ValueError("At least one weight must be positive.")
    if math.isclose(total, 1.0, rel_tol=1e-9):
        return weight_a, weight_b
    if math.isclose(total, 100.0, rel_tol=1e-9):
        return weight_a / 100.0, weight_b / 100.0
    return weight_a / total, weight_b / total


def compute_reranker_score(
    semantic_similarity: float,
    emotion_similarity: float,
    weight_a: float,
    weight_b: float,
) -> float:
    a, b = normalize_weights(weight_a, weight_b)
    return a * semantic_similarity + b * emotion_similarity


def get_valid_conversations(dataset: list[Conversation]) -> list[Conversation]:
    valid_conversations: list[Conversation] = []
    for conversation in dataset:
        if conversation.target_turn < 0 or conversation.golden_turn < 0:
            continue
        if not conversation.turns:
            continue
        if any(
            not turn.semantic_embedding or turn.vad_scores is None
            for turn in conversation.turns
        ):
            continue
        valid_conversations.append(conversation)
    return valid_conversations


def rank_previous_turns(
    conversation: Conversation,
    weight_a: float,
    weight_b: float,
) -> list[dict[str, Any]]:
    target_turn_index = conversation.target_turn
    target = next((turn for turn in conversation.turns if turn.turn == target_turn_index), None)
    if target is None:
        return []

    prior_turns = [
        turn for turn in conversation.turns
        if isinstance(turn.turn, int) and turn.turn < target_turn_index
    ]
    ranked: list[dict[str, Any]] = []
    for candidate in prior_turns:
        semantic_similarity = compute_semantic_similarity(target, candidate)
        emotion_similarity = compute_emotion_similarity(target, candidate)
        reranker_score = compute_reranker_score(
            semantic_similarity,
            emotion_similarity,
            weight_a,
            weight_b,
        )
        ranked.append(
            {
                "candidate_turn": candidate.turn,
                "speaker": candidate.speaker,
                "utterance": candidate.utterance,
                "semantic_similarity": semantic_similarity,
                "emotion_similarity": emotion_similarity,
                "reranker_score": reranker_score,
            }
        )
    return sorted(ranked, key=lambda item: item["reranker_score"], reverse=True)


def compute_conversation_metrics(
    ranked_candidates: list[dict[str, Any]],
    golden_turn: int,
) -> dict[str, float]:
    ranks = [index + 1 for index, candidate in enumerate(ranked_candidates) if candidate["candidate_turn"] == golden_turn]
    if not ranks:
        return {
            "hits_at_1": 0.0,
            "hits_at_3": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "reciprocal_rank": 0.0,
        }

    rank = ranks[0]
    return {
        "hits_at_1": 1.0 if rank <= 1 else 0.0,
        "hits_at_3": 1.0 if rank <= 3 else 0.0,
        "recall_at_1": 1.0 if rank <= 1 else 0.0,
        "recall_at_3": 1.0 if rank <= 3 else 0.0,
        "reciprocal_rank": 1.0 / rank,
    }


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    total = len(metrics_list)
    if total == 0:
        return {
            "hits_at_1": 0.0,
            "hits_at_3": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "mrr": 0.0,
        }

    aggregated = {
        "hits_at_1": sum(metrics["hits_at_1"] for metrics in metrics_list) / total,
        "hits_at_3": sum(metrics["hits_at_3"] for metrics in metrics_list) / total,
        "recall_at_1": sum(metrics["recall_at_1"] for metrics in metrics_list) / total,
        "recall_at_3": sum(metrics["recall_at_3"] for metrics in metrics_list) / total,
        "mrr": sum(metrics["reciprocal_rank"] for metrics in metrics_list) / total,
    }
    return aggregated


def build_conversation_result(
    conversation: Conversation,
    ranked_candidates: list[dict[str, Any]],
    weight_a: float,
    weight_b: float,
) -> dict[str, Any]:
    golden_rank = None
    ranked_messages: list[dict[str, Any]] = []
    for index, candidate in enumerate(ranked_candidates):
        rank = index + 1
        is_golden = candidate["candidate_turn"] == conversation.golden_turn
        if is_golden and golden_rank is None:
            golden_rank = rank
        ranked_messages.append(
            {
                "rank": rank,
                "turn": candidate["candidate_turn"],
                "speaker": candidate["speaker"],
                "utterance": candidate["utterance"],
                "semantic_similarity": candidate["semantic_similarity"],
                "emotion_similarity": candidate["emotion_similarity"],
                "reranker_score": candidate["reranker_score"],
                "is_golden": is_golden,
                "weight_distribution": {
                    "semantic": weight_a,
                    "vad": weight_b,
                },
            }
        )

    return {
        "conversation_id": conversation.conversation_id,
        "target_turn": conversation.target_turn,
        "golden_turn": conversation.golden_turn,
        "golden_rank": golden_rank,
        "ranked_messages": ranked_messages,
    }


def weight_tokens(weight_a: float, weight_b: float) -> tuple[int, int]:
    norm_a, norm_b = normalize_weights(weight_a, weight_b)
    return int(round(norm_a * 100)), int(round(norm_b * 100))


def output_filename(split: str, weight_a: float, weight_b: float) -> str:
    pct_a, pct_b = weight_tokens(weight_a, weight_b)
    return f"results_{split}_{pct_a}_{pct_b}.json"


def run_experiment(
    split: str,
    input_path: Path,
    output_dir: Path,
    weight_pairs: list[tuple[float, float]] | None = None,
) -> list[Path]:
    if weight_pairs is None:
        weight_pairs = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]

    raw_dataset = load_conversations(input_path)
    dataset = get_valid_conversations(raw_dataset)
    written_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for weight_a, weight_b in weight_pairs:
        metrics_list: list[dict[str, float]] = []
        conversation_results: list[dict[str, Any]] = []
        for conversation in dataset:
            ranked_candidates = rank_previous_turns(conversation, weight_a, weight_b)
            if not ranked_candidates:
                continue
            metrics = compute_conversation_metrics(ranked_candidates, conversation.golden_turn)
            metrics_list.append(metrics)
            conversation_results.append(
                build_conversation_result(
                    conversation=conversation,
                    ranked_candidates=ranked_candidates,
                    weight_a=weight_a,
                    weight_b=weight_b,
                )
            )

        pct_a, pct_b = weight_tokens(weight_a, weight_b)
        result_payload = {
            "split": split,
            "input_path": str(input_path),
            "weight_distribution": {
                "semantic": weight_a,
                "vad": weight_b,
                "semantic_percent": pct_a,
                "vad_percent": pct_b,
            },
            "metrics": {
                **aggregate_metrics(metrics_list),
                "raw_conversation_count": len(raw_dataset),
                "valid_conversation_count": len(dataset),
                "evaluated_conversation_count": len(metrics_list),
            },
            "conversation_results": conversation_results,
        }

        out_path = output_dir / output_filename(split, weight_a, weight_b)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(result_payload, handle, indent=2)
        written_paths.append(out_path)

    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG experiment on the embedded RECCON dataset.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to one split embedded dataset JSON file.",
    )
    source_group.add_argument(
        "--splits",
        nargs="+",
        choices=[*SPLITS, "all"],
        default=None,
        help="One or more splits to run (train valid test), or 'all'.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
        help="Weight values for the semantic similarity component. The emotion weight is computed as 1 - a.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where split result files are written.",
    )
    args = parser.parse_args()

    weight_pairs = [(a, 1.0 - a) for a in args.weights]
    split_inputs = resolve_split_input_paths(args.input, args.splits)

    all_written: list[Path] = []
    for split, input_path in split_inputs:
        written = run_experiment(
            split=split,
            input_path=input_path,
            output_dir=args.output_dir,
            weight_pairs=weight_pairs,
        )
        all_written.extend(written)

    print("Experiment results written to:")
    for path in all_written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
