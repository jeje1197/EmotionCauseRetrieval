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
from src.scripts.embed_dataset_helpers import (
    DATA_DIR,
    SELECTED_FILENAME,
    EMBEDDED_FILENAME,
    load_json,
    write_json,
    retry_call,
    validate_embedding,
    validate_vad_scores,
    create_embedding,
    _parse_int,
    score_vad,
    count_embeddings,
    validate_embedded_dataset,
)


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
        # We'll process one conversation at a time and use batch APIs per conversation when possible.
        total_conversations = len({conv_idx for conv_idx, _, _ in tasks})
        print(f"Processing {total_conversations} conversations for embedding (one-at-a-time batching).")

        # Group tasks by conversation
        conv_map: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for conv_idx, turn_idx, turn in tasks:
            conv_map.setdefault(conv_idx, []).append((turn_idx, turn))

        def _process_conversation(conv_idx: int, turns_list: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
            nonlocal semantic_duration_total, semantic_calls, vad_duration_total, vad_calls

            conv = dataset[conv_idx]
            conv_id = conv.get("conversation_id", str(conv_idx))

            embed_texts: list[str] = []
            embed_positions: list[int] = []
            vad_texts: list[str] = []
            vad_positions: list[int] = []
            for turn_idx, turn in turns_list:
                if not turn.get("semantic_embedding"):
                    embed_texts.append(str(turn.get("utterance", "")))
                    embed_positions.append(turn_idx)
                if not turn.get("vad_scores"):
                    vad_texts.append(str(turn.get("utterance", "")))
                    vad_positions.append(turn_idx)

            client = GeminiClient()
            local_failed: list[dict[str, Any]] = []

            embedding_results = None
            if embed_texts:
                embed_start = time.perf_counter()
                try:
                    embedding_results = client.embed_texts(embed_texts)
                    embedding_time = time.perf_counter() - embed_start
                    print(
                        f"Inline embeddings completed for conversation {conv_id} ({len(embed_texts)} vectors) "
                        f"in {embedding_time:.2f}s"
                    )
                except Exception as exc:
                    embedding_time = time.perf_counter() - embed_start
                    err = str(exc)
                    for turn_idx in embed_positions:
                        turn = dataset[conv_idx]["turns"][turn_idx]
                        turn["_embedding_error"] = err
                        local_failed.append({"conversation_index": conv_idx, "turn_index": turn_idx, "error": err})
                    embedding_results = None

            if embedding_results:
                flat_embs: list[list[float]] = []
                if isinstance(embedding_results, list) and embedding_results and isinstance(embedding_results[0], list):
                    flat_embs = embedding_results
                else:
                    try:
                        flat_embs = [list(item) for item in embedding_results]
                    except Exception:
                        flat_embs = []

                for pos, turn_idx in enumerate(embed_positions):
                    if pos < len(flat_embs):
                        dataset[conv_idx]["turns"][turn_idx]["semantic_embedding"] = flat_embs[pos]
                        dataset[conv_idx]["turns"][turn_idx].pop("_embedding_error", None)
                        print(f"Assigned embedding for conv={conv_id} turn={turn_idx}")
                    else:
                        err = "Missing embedding result"
                        turn = dataset[conv_idx]["turns"][turn_idx]
                        turn["_embedding_error"] = err
                        local_failed.append({"conversation_index": conv_idx, "turn_index": turn_idx, "error": err})

            vad_results: list[dict[str, float] | dict[str, str]] = []
            for text in vad_texts:
                vad_start = time.perf_counter()
                try:
                    vad_res = score_vad(client, text)
                    vad_duration_total += time.perf_counter() - vad_start
                    vad_results.append(vad_res)
                except Exception as exc:
                    vad_duration_total += time.perf_counter() - vad_start
                    vad_results.append({"error": str(exc)})

            if vad_results:
                for pos, turn_idx in enumerate(vad_positions):
                    if pos < len(vad_results):
                        res = vad_results[pos]
                        if isinstance(res, dict) and "error" in res:
                            dataset[conv_idx]["turns"][turn_idx]["_embedding_error"] = res["error"]
                            local_failed.append({"conversation_index": conv_idx, "turn_index": turn_idx, "error": res["error"]})
                        else:
                            dataset[conv_idx]["turns"][turn_idx]["vad_scores"] = res
                            dataset[conv_idx]["turns"][turn_idx].pop("_embedding_error", None)
                            print(f"Assigned VAD for conv={conv_id} turn={turn_idx}")
                    else:
                        err = "Missing VAD result"
                        turn = dataset[conv_idx]["turns"][turn_idx]
                        turn["_embedding_error"] = err
                        local_failed.append({"conversation_index": conv_idx, "turn_index": turn_idx, "error": err})

            return local_failed

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            future_to_conv = {
                executor.submit(_process_conversation, conv_idx, turns_list): conv_idx
                for conv_idx, turns_list in conv_map.items()
            }
            for future in concurrent.futures.as_completed(future_to_conv):
                conv_idx = future_to_conv[future]
                try:
                    result = future.result()
                    failed_tasks.extend(result)
                except Exception as exc:
                    err = str(exc)
                    for turn_idx, _ in conv_map[conv_idx]:
                        turn = dataset[conv_idx]["turns"][turn_idx]
                        turn["_embedding_error"] = err
                        failed_tasks.append({"conversation_index": conv_idx, "turn_index": turn_idx, "error": err})
                    print(f"Conversation processing failed conv={conv_idx}: {err}")
                completed += 1
                print(f"Finished conversation batch {completed}/{total_conversations}")

        print("All conversations processed.")

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
    semantic_avg = semantic_duration_total / semantic_calls if semantic_calls else 0.0
    vad_avg = vad_duration_total / vad_calls if vad_calls else 0.0
    print(
        f"Embedding stage duration: {stage_duration:.2f}s | "
        f"semantic calls: {semantic_calls}, avg {semantic_avg:.2f}s, "
        f"vad calls: {vad_calls}, avg {vad_avg:.2f}s"
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
