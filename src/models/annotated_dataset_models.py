"""Typed models for loaded RECCON dataset artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VadScores:
    valence: float
    arousal: float
    dominance: float

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "VadScores":
        return VadScores(
            valence=float(data["valence"]),
            arousal=float(data["arousal"]),
            dominance=float(data["dominance"]),
        )


@dataclass
class ConversationTurn:
    turn: int
    speaker: str
    utterance: str
    emotion: str
    semantic_embedding: list[float] = field(default_factory=list)
    vad_scores: VadScores | None = None
    expanded_emotion_cause_evidence: list[int] | None = None
    expanded_emotion_cause_span: list[str] | None = None
    explanation: str | None = None
    type: list[str] | None = None
    has_multiple_causes: bool = False

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ConversationTurn":
        return ConversationTurn(
            turn=int(data.get("turn", -1)),
            speaker=str(data.get("speaker", "")),
            utterance=str(data.get("utterance", "")),
            emotion=str(data.get("emotion", "")),
            semantic_embedding=[float(value) for value in data.get("semantic_embedding", [])],
            vad_scores=VadScores.from_dict(data["vad_scores_1"]) if data.get("vad_scores_1") else None,
            expanded_emotion_cause_evidence=[int(value) for value in data.get("expanded emotion cause evidence", [])] if data.get("expanded emotion cause evidence") else None,
            expanded_emotion_cause_span=[str(value) for value in data.get("expanded emotion cause span", [])] if data.get("expanded emotion cause span") else None,
            explanation=str(data.get("explanation")) if data.get("explanation") is not None else None,
            type=[str(value) for value in data.get("type", [])] if data.get("type") else None,
            has_multiple_causes=bool(data.get("has_multiple_causes", False)),
        )


@dataclass
class Conversation:
    conversation_id: str
    dataset_split: str
    target_turn: int
    golden_turn: int
    turns: list[ConversationTurn]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Conversation":
        dataset_split = str(data.get("dataset_split", "")).lower()
        if dataset_split not in {"train", "valid", "test"}:
            raise ValueError(
                "Conversation data must include dataset_split with value 'train', 'valid', or 'test'."
            )
        return Conversation(
            conversation_id=str(data["conversation_id"]),
            dataset_split=dataset_split,
            target_turn=int(data["target_turn"]),
            golden_turn=int(data["golden_turn"]),
            turns=[ConversationTurn.from_dict(turn) for turn in data.get("turns", [])],
        )

    @staticmethod
    def load_list(path: Path) -> list["Conversation"]:
        import json

        with path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
        return [Conversation.from_dict(item) for item in raw_data]
