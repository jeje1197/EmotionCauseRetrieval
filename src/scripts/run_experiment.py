"""Run RAG experiment metrics on the embedded RECCON dataset.

This module computes ranking scores using both semantic and emotion similarity
features, then evaluates Hits@1, Hits@3, Recall@1, Recall@3, and MRR.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.models import Conversation, ConversationTurn

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
EMBEDDED_FILENAME = "reccon_embedded.json"


def load_conversations(path: Path) -> list[Conversation]:
    return Conversation.load_list(path)


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


def run_experiment(
    input_path: Path | None = None,
    weight_pairs: list[tuple[float, float]] | None = None,
) -> dict[str, dict[str, float]]:
    if input_path is None:
        input_path = DATA_DIR / EMBEDDED_FILENAME
    if weight_pairs is None:
        weight_pairs = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]

    raw_dataset = load_conversations(input_path)
    dataset = get_valid_conversations(raw_dataset)
    results: dict[str, dict[str, float]] = {}

    for weight_a, weight_b in weight_pairs:
        metrics_list: list[dict[str, float]] = []
        for conversation in dataset:
            ranked_candidates = rank_previous_turns(conversation, weight_a, weight_b)
            if not ranked_candidates:
                continue
            metrics = compute_conversation_metrics(ranked_candidates, conversation["golden_turn"])
            metrics_list.append(metrics)

        results[f"a={weight_a},b={weight_b}"] = aggregate_metrics(metrics_list)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG experiment on the embedded RECCON dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / EMBEDDED_FILENAME,
        help="Path to the embedded dataset JSON file.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
        help="Weight values for the semantic similarity component. The emotion weight is computed as 1 - a.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path for experiment metrics.",
    )
    args = parser.parse_args()

    weight_pairs = [(a, 1.0 - a) for a in args.weights]
    experiment_results = run_experiment(input_path=args.input, weight_pairs=weight_pairs)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(experiment_results, handle, indent=2)
        print(f"Experiment results written to: {args.output}")
    else:
        print(json.dumps(experiment_results, indent=2))


if __name__ == "__main__":
    main()
