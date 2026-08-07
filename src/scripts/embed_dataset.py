"""Embed RECCON dataset utterances and score them with VAD metrics.

This module uses the Gemini API client to create semantic embeddings and to
compute VAD scores for each utterance. The embedding and scoring process is
configurable with retry and exponential backoff behavior.
"""

from __future__ import annotations

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


def create_embedding(client: GeminiClient, text: str) -> list[float]:
    """Create a semantic embedding for text using Gemini embeddings."""
    # The exact API call may differ depending on installed SDK version.
    response = client.client.embeddings.create(model="gemini-embedding-001", input=text)
    embedding = getattr(response, "embedding", None)
    if embedding is None and hasattr(response, "data"):
        embedding = response.data[0].embedding
    if embedding is None:
        raise RuntimeError("Embedding request did not return a vector.")
    return list(embedding)


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
    return {key: float(value) for key, value in result.items()}


def embed_dataset(
    input_path: Path | None = None,
    output_path: Path | None = None,
    client: GeminiClient | None = None,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Path:
    """Embed selected dataset turns and write the result to a JSON file."""
    if input_path is None:
        input_path = DATA_DIR / SELECTED_FILENAME
    if output_path is None:
        output_path = DATA_DIR / EMBEDDED_FILENAME
    if client is None:
        client = GeminiClient(model="gemini-embedding-001")

    dataset = load_json(input_path)
    for conversation in dataset:
        for turn in conversation.get("turns", []):
            utterance = turn.get("utterance", "")
            if not isinstance(utterance, str):
                utterance = str(utterance)

            if not turn.get("semantic_embedding"):
                embedding = retry_call(
                    create_embedding,
                    max_retries=max_retries,
                    initial_delay=initial_delay,
                    backoff_factor=backoff_factor,
                    client=client,
                    text=utterance,
                )
                turn["semantic_embedding"] = embedding

            if not turn.get("vad_scores"):
                vad_scores = retry_call(
                    score_vad,
                    max_retries=max_retries,
                    initial_delay=initial_delay,
                    backoff_factor=backoff_factor,
                    client=client,
                    text=utterance,
                )
                turn["vad_scores"] = vad_scores

    write_json(output_path, dataset)
    return output_path


def main() -> None:
    output_path = embed_dataset()
    print(f"Embedded dataset written to: {output_path}")


if __name__ == "__main__":
    main()
