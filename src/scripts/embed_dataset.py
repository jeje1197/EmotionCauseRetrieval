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


def _embed_chunk(
    chunk: list[tuple[int, int, dict[str, Any]]],
    max_retries: int,
    initial_delay: float,
    backoff_factor: float,
) -> list[dict[str, Any]]:
    """Embed a chunk of turns using a single Gemini client instance.

    Returns a list of per-turn results: (conversation_index, turn_index, updated_turn, semantic_duration, vad_duration)
    """
    client = GeminiClient()
    results: list[dict[str, Any]] = []

    # Prepare texts for embeddings where needed
    texts_to_embed: list[str] = []
    embed_positions: list[int] = []
    for i, (_conv_idx, _turn_idx, turn) in enumerate(chunk):
        utterance = turn.get("utterance", "")
        if not isinstance(utterance, str):
            utterance = str(utterance)
        if not turn.get("semantic_embedding"):
            texts_to_embed.append(utterance)
            embed_positions.append(i)

    embedding_results: list[list[float]] | None = None
    if texts_to_embed:
        start_time = time.perf_counter()
        embedding_results = retry_call(
            client.embed_texts,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            texts=list(texts_to_embed),
        )
        embedding_time = time.perf_counter() - start_time
        print(
            f"Embedding batch finished for chunk: requested={len(texts_to_embed)} vectors, "
            f"took={embedding_time:.2f}s"
        )
    else:
        embedding_time = 0.0

    # Assign embedding results back to chunk entries
    if embedding_results:
        for pos_index, emb in enumerate(embedding_results):
            chunk_index = embed_positions[pos_index]
            chunk[chunk_index][2]["semantic_embedding"] = emb

    # For VAD scores, call score_vad per utterance but reuse the same client
    for conv_idx, turn_idx, turn in chunk:
        semantic_duration = 0.0
        vad_duration = 0.0
        error: str | None = None
        try:
            # semantic_embedding already set above if needed
            if not turn.get("semantic_embedding"):
                # fallback: create single embedding
                start_time = time.perf_counter()
                turn["semantic_embedding"] = retry_call(
                    create_embedding,
                    max_retries=max_retries,
                    initial_delay=initial_delay,
                    backoff_factor=backoff_factor,
                    client=client,
                    text=str(turn.get("utterance", "")),
                )
                semantic_duration = time.perf_counter() - start_time
                print(f"Semantic embedding computed for conv={conv_idx} turn={turn_idx} duration={semantic_duration:.2f}s")

            if not turn.get("vad_scores"):
                start_time = time.perf_counter()
                turn["vad_scores"] = retry_call(
                    score_vad,
                    max_retries=max_retries,
                    initial_delay=initial_delay,
                    backoff_factor=backoff_factor,
                    client=client,
                    text=str(turn.get("utterance", "")),
                )
                vad_duration = time.perf_counter() - start_time
                print(f"VAD computed for conv={conv_idx} turn={turn_idx} duration={vad_duration:.2f}s")

        except Exception as exc:
            error = str(exc)

        results.append(
            {
                "conversation_index": conv_idx,
                "turn_index": turn_idx,
                "updated_turn": dict(turn),
                "semantic_duration": semantic_duration,
                "vad_duration": vad_duration,
                "error": error,
            }
        )
        if error:
            print(f"Error during chunk processing conv={conv_idx} turn={turn_idx}: {error}")

    return results


def embed_dataset(
    input_path: Path | None = None,
    output_path: Path | None = None,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    concurrent_requests: int = 10,
) -> Path:
    """Embed selected dataset turns and write the result to a JSON file."""
    if input_path is None:
        input_path = DATA_DIR / SELECTED_FILENAME
    if output_path is None:
        output_path = DATA_DIR / EMBEDDED_FILENAME

    dataset = load_json(input_path)
    tasks: list[tuple[int, int, dict[str, Any]]] = []
    for conversation_index, conversation in enumerate(dataset):
        target_turn = _parse_int(conversation.get("target_turn"))
        if target_turn is None or target_turn < 0:
            # Skip conversations without a valid target_turn
            continue
        for turn_index, turn in enumerate(conversation.get("turns", [])):
            turn_num = _parse_int(turn.get("turn"))
            if turn_num is None:
                continue
            # Do not embed turns after the conversation's target turn
            if turn_num > target_turn:
                continue
            if not turn.get("semantic_embedding") or not turn.get("vad_scores"):
                tasks.append((conversation_index, turn_index, turn))

    stage_start = time.perf_counter()
    semantic_duration_total = 0.0
    vad_duration_total = 0.0
    semantic_calls = 0
    vad_calls = 0

    failed_tasks: list[dict[str, Any]] = []
    if tasks:
        total_tasks = len(tasks)
        print(f"Submitting {total_tasks} embedding tasks (max_workers={concurrent_requests})")

        # Chunk tasks into batches for lightweight batching of embeddings
        batch_size = 16
        chunks: list[list[tuple[int, int, dict[str, Any]]]] = [
            tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            future_to_chunk = {
                executor.submit(
                    _embed_chunk,
                    chunk,
                    max_retries,
                    initial_delay,
                    backoff_factor,
                ): chunk
                for chunk in chunks
            }

            completed = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    results = future.result()
                except Exception as exc:
                    # Entire chunk failed unexpectedly; mark all turns as failed
                    err_msg = str(exc)
                    for conv_idx, turn_idx, _ in chunk:
                        turn = dataset[conv_idx]["turns"][turn_idx]
                        turn["_embedding_error"] = err_msg
                        failed_tasks.append({"conversation_index": conv_idx, "turn_index": turn_idx, "error": err_msg})
                        completed += 1
                        print(f"Chunk failed: conv={conv_idx} turn={turn_idx} error={err_msg}")
                    continue

                # Process per-turn results from chunk
                for item in results:
                    conv_idx = item["conversation_index"]
                    turn_idx = item["turn_index"]
                    updated_turn = item["updated_turn"]
                    semantic_duration = item["semantic_duration"]
                    vad_duration = item["vad_duration"]
                    error = item.get("error")

                    if error:
                        turn = dataset[conv_idx]["turns"][turn_idx]
                        turn["_embedding_error"] = error
                        failed_tasks.append({"conversation_index": conv_idx, "turn_index": turn_idx, "error": error})
                        completed += 1
                        print(f"Failed task {completed}/{total_tasks}: conv={conv_idx} turn={turn_idx} error={error}")
                        continue

                    # Success
                    dataset[conv_idx]["turns"][turn_idx] = updated_turn
                    dataset[conv_idx]["turns"][turn_idx].pop("_embedding_error", None)
                    semantic_duration_total += semantic_duration
                    vad_duration_total += vad_duration
                    if semantic_duration > 0:
                        semantic_calls += 1
                    if vad_duration > 0:
                        vad_calls += 1
                    completed += 1
                    semantic_flag = semantic_duration > 0
                    vad_flag = vad_duration > 0
                    print(
                        f"Completed task {completed}/{total_tasks}: conv={conv_idx} turn={turn_idx} "
                        f"(semantic_call={semantic_flag}, vad_call={vad_flag}) "
                        f"durations: semantic={semantic_duration:.2f}s vad={vad_duration:.2f}s"
                    )

                # Periodic progress update
                if completed % 50 == 0 or completed == total_tasks:
                    print(f"Embedding progress: {completed}/{total_tasks} tasks finished")

            print("All embedding tasks finished (some may have failed).")

    # Summary of failures
    if failed_tasks:
        print(f"Embedding completed with {len(failed_tasks)} failed tasks. \n" +
              "Failed turns are annotated with the '_embedding_error' field in the output JSON.")

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


def repair_embeddings(
    input_path: Path | None = None,
    output_path: Path | None = None,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    concurrent_requests: int = 5,
):
    """Retry embedding for turns annotated with '_embedding_error' or missing embeddings.

    The function updates the dataset in-place and writes the fixed dataset to
    the provided output path (or overwrites the input path if output_path is None).
    """
    if input_path is None:
        input_path = DATA_DIR / SELECTED_FILENAME
    if output_path is None:
        output_path = DATA_DIR / EMBEDDED_FILENAME

    dataset = load_json(input_path)
    tasks: list[tuple[int, int, dict[str, Any]]] = []
    for conversation_index, conversation in enumerate(dataset):
        target_turn = _parse_int(conversation.get("target_turn"))
        if target_turn is None or target_turn < 0:
            continue
        for turn_index, turn in enumerate(conversation.get("turns", [])):
            turn_num = _parse_int(turn.get("turn"))
            if turn_num is None or turn_num > target_turn:
                continue
            if turn.get("_embedding_error") or not turn.get("semantic_embedding") or not turn.get("vad_scores"):
                tasks.append((conversation_index, turn_index, turn))

    if not tasks:
        print("No failed embeddings found to repair.")
        return output_path

    print(f"Repairing {len(tasks)} failed embedding tasks (max_workers={concurrent_requests})")

    # Chunk tasks for lightweight batching
    batch_size = 16
    chunks: list[list[tuple[int, int, dict[str, Any]]]] = [
        tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        future_to_chunk = {
            executor.submit(
                _embed_chunk,
                chunk,
                max_retries,
                initial_delay,
                backoff_factor,
            ): chunk
            for chunk in chunks
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                results = future.result()
            except Exception as exc:
                err_msg = str(exc)
                for conv_idx, turn_idx, _ in chunk:
                    turn = dataset[conv_idx]["turns"][turn_idx]
                    turn["_embedding_error"] = err_msg
                    completed += 1
                    print(f"Repair chunk failed: conv={conv_idx} turn={turn_idx} error={err_msg}")
                continue

            for item in results:
                conv_idx = item["conversation_index"]
                turn_idx = item["turn_index"]
                updated_turn = item["updated_turn"]
                error = item.get("error")
                if error:
                    turn = dataset[conv_idx]["turns"][turn_idx]
                    turn["_embedding_error"] = error
                    completed += 1
                    print(f"Repair failed {completed}/{len(tasks)}: conv={conv_idx} turn={turn_idx} error={error}")
                else:
                    dataset[conv_idx]["turns"][turn_idx] = updated_turn
                    dataset[conv_idx]["turns"][turn_idx].pop("_embedding_error", None)
                    completed += 1
                    print(f"Repaired {completed}/{len(tasks)}: conv={conv_idx} turn={turn_idx}")

    write_json(output_path, dataset)
    validate_embedded_dataset(output_path)
    print("Repair stage completed; updated dataset written to:", output_path)
    return output_path


def main() -> None:
    output_path = embed_dataset()
    print(f"Embedded dataset written to: {output_path}")


if __name__ == "__main__":
    main()
