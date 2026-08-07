"""Embed RECCON dataset utterances and score them with VAD metrics.

This module uses the Gemini API client to create semantic embeddings and to
compute VAD scores for each utterance. The embedding and scoring process is
configurable with retry and exponential backoff behavior.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import time
from pathlib import Path
from typing import Any

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
    func,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    *args,
    **kwargs,
) -> Any:
    """Call a function with retry and exponential backoff on exceptions."""
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_time = initial_delay * (backoff_factor**(attempt - 1))
            time.sleep(sleep_time)


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
        if not isinstance(value, (int, float)):
            raise ValueError(f"VAD value for {key} must be numeric.")
        if value < -1.0 or value > 1.0:
            raise ValueError(f"VAD value for {key} must be between -1 and 1.")
    return {key: float(vad_scores[key]) for key in ("valence", "arousal", "dominance")}


def create_embedding(client: GeminiClient, text: str) -> list[float]:
    """Create a semantic embedding for text using Gemini embeddings."""
    embeddings = client.embed_texts([text])
    if not embeddings:
        raise RuntimeError("Embedding request did not return a vector.")
    return validate_embedding(embeddings[0])


def score_vad(client: GeminiClient, text: str) -> dict[str, float]:
    """Generate VAD scores for an utterance using Gemini structured JSON completion."""
    schema = {
        "type": "object",
        "properties": {
            "valence": {"type": "number"},
            "arousal": {"type": "number"},
            "dominance": {"type": "number"},
        },
        "required": ["valence", "arousal", "dominance"],
    }
    result = client.generate_structured_json(VAD_PROMPT.format(user_query=text), schema)
    return validate_vad_scores({key: float(value) for key, value in result.items()})


def count_embeddings(dataset: list[dict[str, Any]]) -> tuple[int, int, int]:
    total_turns = 0
    semantic_success = 0
    vad_success = 0
    for conversation in dataset:
        for turn in conversation.get("turns", []):
            total_turns += 1
            if turn.get("semantic_embedding"):
                semantic_success += 1
            if turn.get("vad_scores"):
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
            if not turn.get("semantic_embedding"):
                raise ValueError("Each turn must include a non-empty semantic_embedding.")
            if not turn.get("vad_scores"):
                raise ValueError("Each turn must include non-empty vad_scores.")


def _embed_turn(
    conversation_index: int,
    turn_index: int,
    turn_data: dict[str, Any],
    max_retries: int,
    initial_delay: float,
    backoff_factor: float,
) -> tuple[int, int, dict[str, Any], float, float]:
    utterance = turn_data.get("utterance", "")
    if not isinstance(utterance, str):
        utterance = str(utterance)

    client = GeminiClient()
    updated_turn = dict(turn_data)
    semantic_duration = 0.0
    vad_duration = 0.0

    if not updated_turn.get("semantic_embedding"):
        start_time = time.perf_counter()
        updated_turn["semantic_embedding"] = retry_call(
            create_embedding,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            client=client,
            text=utterance,
        )
        semantic_duration = time.perf_counter() - start_time

    if not updated_turn.get("vad_scores"):
        start_time = time.perf_counter()
        updated_turn["vad_scores"] = retry_call(
            score_vad,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            client=client,
            text=utterance,
        )
        vad_duration = time.perf_counter() - start_time

    return conversation_index, turn_index, updated_turn, semantic_duration, vad_duration


def embed_dataset(
    input_path: Path | None = None,
    output_path: Path | None = None,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    concurrent_requests: int = 5,
) -> Path:
    """Embed selected dataset turns and write the result to a JSON file."""
    if input_path is None:
        input_path = DATA_DIR / SELECTED_FILENAME
    if output_path is None:
        output_path = DATA_DIR / EMBEDDED_FILENAME

    dataset = load_json(input_path)
    tasks: list[tuple[int, int, dict[str, Any]]] = []
    for conversation_index, conversation in enumerate(dataset):
        for turn_index, turn in enumerate(conversation.get("turns", [])):
            if not turn.get("semantic_embedding") or not turn.get("vad_scores"):
                tasks.append((conversation_index, turn_index, turn))

    stage_start = time.perf_counter()
    semantic_duration_total = 0.0
    vad_duration_total = 0.0
    semantic_calls = 0
    vad_calls = 0

    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            future_to_task = {
                executor.submit(
                    _embed_turn,
                    conversation_index,
                    turn_index,
                    turn,
                    max_retries,
                    initial_delay,
                    backoff_factor,
                ): (conversation_index, turn_index)
                for conversation_index, turn_index, turn in tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                (
                    updated_conversation_index,
                    updated_turn_index,
                    updated_turn,
                    semantic_duration,
                    vad_duration,
                ) = future.result()
                dataset[updated_conversation_index]["turns"][updated_turn_index] = updated_turn
                semantic_duration_total += semantic_duration
                vad_duration_total += vad_duration
                if semantic_duration > 0:
                    semantic_calls += 1
                if vad_duration > 0:
                    vad_calls += 1

    stage_duration = time.perf_counter() - stage_start
    total_turns, semantic_success, vad_success = count_embeddings(dataset)
    write_json(output_path, dataset)
    validate_embedded_dataset(output_path)

    print(
        f"Embedding summary: {semantic_success}/{total_turns} semantic embeddings, "
        f"{vad_success}/{total_turns} VAD scores."
    )
    print(
        f"Embedding stage duration: {stage_duration:.2f}s | "
        f"semantic calls: {semantic_calls}, avg {semantic_duration_total / semantic_calls:.2f}s if >0 else 0, "
        f"vad calls: {vad_calls}, avg {vad_duration_total / vad_calls:.2f}s if >0 else 0"
    )

    return output_path


def main() -> None:
    output_path = embed_dataset()
    print(f"Embedded dataset written to: {output_path}")


if __name__ == "__main__":
    main()
