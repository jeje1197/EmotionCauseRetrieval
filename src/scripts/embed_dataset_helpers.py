from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from src.clients.gemini_client import GeminiClient
from src.prompts.vad_prompt import VAD_PROMPT

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SELECTED_FILENAME = "reccon_selected.json"
EMBEDDED_FILENAME = "reccon_embedded.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def retry_call(
    func: Callable, max_retries: int = 5, initial_delay: float = 1.0, backoff_factor: float = 2.0, *args, **kwargs
):
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(initial_delay * (backoff_factor ** (attempt - 1)))


def validate_embedding(embedding: Any) -> list[float]:
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("Embedding response is invalid or empty.")
    validated_embedding: list[float] = []
    for value in embedding:
        if not isinstance(value, (int, float)):
            raise ValueError("Embedding values must be numeric.")
        validated_embedding.append(float(value))
    return validated_embedding


def validate_vad_scores(vad_scores: Any) -> dict[str, float]:
    if not isinstance(vad_scores, dict):
        raise ValueError("VAD response must be a dictionary.")
    for key in ("valence", "arousal", "dominance"):
        if key not in vad_scores:
            raise ValueError(f"VAD response missing key: {key}")
        value = vad_scores[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"VAD value for {key} must be numeric.")
        if value < -1.0 or value > 1.0:
            raise ValueError(f"VAD value for {key} must be between -1 and 1.")
    return {key: float(vad_scores[key]) for key in ("valence", "arousal", "dominance")}


def create_embedding(client: GeminiClient, text: str) -> list[float]:
    embeddings = client.embed_texts([text])
    if not embeddings:
        raise RuntimeError("Embedding request did not return a vector.")
    return validate_embedding(embeddings[0])


def _parse_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def score_vad(client: GeminiClient, text: str) -> dict[str, float]:
    schema = {
        "type": "object",
        "properties": {
            "valence": {"type": "number"},
            "arousal": {"type": "number"},
            "dominance": {"type": "number"},
        },
        "required": ["valence", "arousal", "dominance"],
    }
    # Use Gemini 3 Flash Preview for VAD scoring
    vad_model = "gemini-3-flash-preview"
    result = client.generate_structured_json(
        VAD_PROMPT.format(user_query=text),
        schema,
        model=vad_model,
        temperature=0.0,
        seed=0,
        top_p=0.95,
        thinking_level="minimal",
        max_output_tokens=1024,
    )
    return validate_vad_scores(result)


def count_embeddings(dataset: list[dict]) -> tuple[int, int, int]:
    total_turns = 0
    semantic_success = 0
    vad_success = 0
    for conversation in dataset:
        for turn in conversation.get("turns", []):
            total_turns += 1
            if turn.get("semantic_embedding"):
                semantic_success += 1
            if turn.get("vad_scores_1"):
                vad_success += 1
    return total_turns, semantic_success, vad_success


def validate_embedded_dataset(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Embedded dataset file does not exist: {path}")
    dataset = load_json(path)
    if not isinstance(dataset, list):
        raise ValueError("Embedded dataset should be a list of conversations.")

    for conversation in dataset:
        if not isinstance(conversation, dict):
            raise ValueError("Each conversation must be a dictionary.")
        turns = conversation.get("turns")
        if not isinstance(turns, list):
            raise ValueError("Each conversation must include a turns list.")
        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError("Each turn must be a dictionary.")
            # Only require embeddings/VAD for turns up to the conversation's target_turn
            target_turn = _parse_int(conversation.get("target_turn"))
            turn_num = _parse_int(turn.get("turn"))
            # If target_turn is not set, require all turns to be embedded/scored
            if target_turn is None:
                require_check = True
            else:
                require_check = (turn_num is not None and turn_num <= target_turn)

            if require_check:
                if not turn.get("semantic_embedding"):
                    raise ValueError("Each required turn must include a non-empty semantic_embedding.")
                if not turn.get("vad_scores_1"):
                    raise ValueError("Each required turn must include non-empty vad_scores_1.")
