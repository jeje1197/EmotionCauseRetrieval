"""Gemini client wrapper for structured Gemini SDK calls."""

import json
from typing import Any, Dict, Iterable, List

from google import genai
from google.genai import types


class GeminiClient:
    def __init__(self, model: str = "gemini-3.6-flash") -> None:
        self.client = genai.Client()
        self.model = model

    def generate_structured_json(self, prompt: str, schema: Dict[str, Any], model: str | None = None) -> Any:
        """Generate structured JSON using the interactions API.

        The `model` parameter can override the client's default model for
        use-cases like structured VAD scoring.
        """
        interaction = self.client.interactions.create(
            model=model or self.model,
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
        )

        output_text = getattr(interaction, "output_text", None)
        if output_text is None:
            output_text = getattr(interaction, "output", None)

        if output_text is None:
            raise RuntimeError("Gemini interaction did not return any output text.")

        if isinstance(output_text, str):
            return json.loads(output_text)

        return output_text

    def embed_texts(self, texts: Iterable[str], model: str | None = None) -> List[List[float]]:
        """Embed a list of texts. Use `model` to override the client's default model.

        Default embedding model: `gemini-embedding-1`.
        """
        # Use the GenAI model resource name for embeddings if not provided.
        # Common available embedding models include 'models/gemini-embedding-2' or 'models/gemini-embedding-001'.
        use_model = model or "models/gemini-embedding-001"
        response = self.client.models.embed_content(
            model=use_model,
            contents=list(texts),
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        if not getattr(response, "embeddings", None):
            raise RuntimeError("Embedding request did not return embeddings.")
        return [list(embedding.values) for embedding in response.embeddings]

    # ---- Batch helpers (best-effort wrappers) ----
    def create_embeddings_batch(self, model: str, inlined_texts: list[str], display_name: str | None = None) -> str:
        """Create an embeddings batch job using inline requests. Returns the job name."""
        config = {"display_name": display_name or "embeddings-batch"}
        # Build simple inline requests expected by the batch API
        inlined_requests = [{"contents": [{"parts": [{"text": text}] , "role": "user"}]} for text in inlined_texts]
        job = self.client.batches.create_embeddings(model=model, src={"inlined_requests": inlined_requests}, config=config)
        return job.name

    def get_batch(self, job_name: str) -> Any:
        return self.client.batches.get(name=job_name)

    def download_file(self, file_name: str) -> bytes:
        return self.client.files.download(file=file_name)
